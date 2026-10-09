"""Model, hardware and step-time model for the KV preemption simulator.

Model shapes come from the public HuggingFace config.json files in data/configs (fetch_data.sh).
Parameter counts are derived from the shapes and checked against published totals in
test_kvsim.py. GPU numbers are datasheet values. Kernel efficiencies are fitted in calibrate.py
to Vidur's measured H100 profiles (Agrawal et al., MLSys'24) and extrapolated with a roofline:
decode attention is linear in KV bytes read, prefill attention in query-key pairs.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONFIGS = Path(__file__).resolve().parent.parent / "data" / "configs"


@dataclass
class Model:
    name: str
    layers: int
    d_model: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    params_total: float
    params_active: float
    layer_dense: float           # non-expert params of all layers (attention, dense MLP, shared experts, router)
    expert_params: float         # one routed expert summed over MoE layers
    head_params: float           # LM head (read every step, computed only for sampled tokens)
    n_experts: int
    top_k: int
    kv_bytes_per_token: float    # all layers, at kv_dtype_bytes
    attn_prefill_flops_pair: float   # FLOPs per (query, key) pair, all layers, all heads
    attn_decode_flops_pair: float
    mla: bool = False


def _cfg(repo):
    return json.loads((CONFIGS / (repo.replace("/", "_") + ".json")).read_text())


def load_model(repo, kv_dtype_bytes=2):
    c = _cfg(repo)
    L, d, H = c["num_hidden_layers"], c["hidden_size"], c["num_attention_heads"]
    V = c["vocab_size"]
    emb, head = V * d, (0 if c.get("tie_word_embeddings") else V * d)
    mla = "kv_lora_rank" in c
    if mla:
        r, rope, nope, vd = c["kv_lora_rank"], c["qk_rope_head_dim"], c["qk_nope_head_dim"], c["v_head_dim"]
        qr = c.get("q_lora_rank") or 1536
        attn = d * qr + qr * H * (nope + rope) + d * (r + rope) + r * H * (nope + vd) + H * vd * d
        kv_tok = L * (r + rope) * kv_dtype_bytes
        kvh, hd = 1, r + rope
        # prefill runs as MHA (qk dim nope+rope, v dim vd); decode runs absorbed MQA over the latent
        pre_pair = L * H * 2 * ((nope + rope) + vd)
        dec_pair = L * H * 2 * ((r + rope) + r)
    else:
        kvh = c["num_key_value_heads"]
        hd = c.get("head_dim") or d // H
        attn = d * H * hd + 2 * d * kvh * hd + H * hd * d
        kv_tok = 2 * L * kvh * hd * kv_dtype_bytes
        pre_pair = dec_pair = L * H * 2 * 2 * hd
    n_exp = c.get("n_routed_experts") or c.get("num_experts") or c.get("num_local_experts") or 0
    top_k = c.get("num_experts_per_tok") or 0
    if n_exp:
        mi = c["moe_intermediate_size"]
        dense_layers = c.get("first_k_dense_replace", 0)
        moe_layers = L - dense_layers
        one_exp = 3 * d * mi
        layer_dense = (L * attn + dense_layers * 3 * d * c["intermediate_size"]
                       + moe_layers * (c.get("n_shared_experts", 0) * one_exp + d * n_exp))
        expert = moe_layers * one_exp
    else:
        layer_dense = L * (attn + 3 * d * c["intermediate_size"])
        expert = 0.0
    total = emb + head + layer_dense + n_exp * expert
    active = emb + head + layer_dense + top_k * expert
    return Model(repo, L, d, H, kvh, hd, V, total, active, layer_dense, expert, head, n_exp, top_k,
                 kv_tok, pre_pair, dec_pair, mla)


@dataclass
class GPU:
    name: str
    flops: float        # dense BF16 FLOP/s
    hbm_bw: float       # B/s
    hbm_cap: float      # B
    host_bw: float      # B/s per direction, GPU to host memory


# NVIDIA datasheets: H100 SXM5 (989 TFLOPS, 3.35 TB/s, 80 GB), H200 SXM (989, 4.8, 141), HGX B200
# (2.25 PFLOPS, 8 TB/s, 180 GB). Host link PCIe Gen5 x16: 64 GB/s raw per direction, about
# 50 GB/s for pinned copies.
GPUS = {
    "H100": GPU("H100", 989e12, 3.35e12, 80e9, 50e9),
    "H200": GPU("H200", 989e12, 4.8e12, 141e9, 50e9),
    "B200": GPU("B200", 2250e12, 8.0e12, 180e9, 50e9),
    # RTX PRO 6000 Blackwell Server Edition: 1,000 TFLOPS BF16 with 2:4 sparsity (500 dense),
    # 1,597 GB/s, 96 GB; used only to cross-check against LLMServingSim's shipped profile
    "PRO6000": GPU("PRO6000", 500e12, 1.597e12, 96e9, 16e9),
}


@dataclass
class Eff:
    """Fitted in calibrate.py; see there for provenance."""
    gemm: float = 0.647            # of peak FLOP/s, compute-bound GEMMs
    mem: float = 0.712             # of HBM bandwidth, weight-bound GEMMs
    elem_const: float = 1.7e-5     # s per layer, elementwise ops
    elem_tok: float = 4.0e-8       # s per layer per token, replicated part (layernorm, add)
    elem_tok_shard: float = 1.5e-7 # s per layer per token / tp, sharded part (act, rope)
    attn_decode: float = 0.93      # of HBM bandwidth
    attn_decode_const: float = 1.6e-5   # s per layer per step with decodes
    attn_prefill_max: float = 0.60      # asymptotic fraction of peak (FA3-class kernel)
    attn_prefill_w0: float = 2.0e10     # FLOPs per layer per rank at half the asymptotic rate
    ar_lat: dict = field(default_factory=lambda: {2: 7e-6, 4: 15e-6, 8: 15e-6})
    ar_bw: dict = field(default_factory=lambda: {2: 300e9, 4: 230e9, 8: 211e9})
    step_overhead: float = 0.5e-3  # s per engine step (CPU, launch, sampling)


@dataclass
class Instance:
    model: Model
    gpu: GPU
    tp: int
    weight_bytes: float = 2.0       # bytes per parameter
    mem_util: float = 0.90          # vLLM gpu_memory_utilization
    act_reserve: float = 4e9        # per GPU: activations, CUDA graphs, workspace
    block_size: int = 16
    eff: Eff = field(default_factory=Eff)

    @property
    def rank_kv_bytes(self):
        """KV bytes per token held by one TP rank. GQA shards KV heads over ranks (replicating
        them when kv_heads < tp); MLA keeps the whole latent on every rank."""
        m = self.model
        return m.kv_bytes_per_token if m.mla else m.kv_bytes_per_token / min(self.tp, m.kv_heads)

    @property
    def free_per_gpu(self):
        return max(self.gpu.hbm_cap * self.mem_util - self.model.params_total * self.weight_bytes / self.tp
                   - self.act_reserve, 0)

    @property
    def num_blocks(self):
        return int(self.free_per_gpu // (self.rank_kv_bytes * self.block_size))

    @property
    def kv_capacity_bytes(self):
        """Unique KV bytes the instance holds (replicas not counted)."""
        return self.num_blocks * self.block_size * self.model.kv_bytes_per_token

    @property
    def block_bytes(self):
        return self.model.kv_bytes_per_token * self.block_size

    def _linear(self, n, n_sampled):
        m, g, e, tp = self.model, self.gpu, self.eff, self.tp
        if m.n_experts:
            touched = m.n_experts * (1 - (1 - m.top_k / m.n_experts) ** n)
            w = m.layer_dense + touched * m.expert_params
            act = m.layer_dense + m.top_k * m.expert_params
        else:
            w = act = m.layer_dense
        t = max(2 * act * n / (tp * g.flops * e.gemm), w * self.weight_bytes / (tp * g.hbm_bw * e.mem))
        t += max(2 * m.head_params * n_sampled / (tp * g.flops * e.gemm),
                 m.head_params * self.weight_bytes / (tp * g.hbm_bw * e.mem))
        t += m.layers * (e.elem_const + n * (e.elem_tok + e.elem_tok_shard / tp))
        if tp > 1:
            t += 2 * m.layers * (e.ar_lat[tp] + n * m.d_model * 2 / e.ar_bw[tp])
        return t

    def _decode_attn(self, n_dec, ctx_sum):
        m, g, e, tp = self.model, self.gpu, self.eff, self.tp
        if not n_dec:
            return 0.0
        mem = ctx_sum * self.rank_kv_bytes / (g.hbm_bw * e.attn_decode)
        cmp = ctx_sum * m.attn_decode_flops_pair / (tp * g.flops * e.gemm)
        return m.layers * e.attn_decode_const + max(mem, cmp)

    def _prefill_attn(self, chunk, ctx_before):
        m, g, e, tp = self.model, self.gpu, self.eff, self.tp
        pairs = chunk * (ctx_before + (chunk + 1) / 2)
        w = pairs * m.attn_prefill_flops_pair / (tp * m.layers)
        eff = e.attn_prefill_max * w / (w + e.attn_prefill_w0)
        return m.layers * w / (g.flops * eff)

    def step_time(self, n_dec, ctx_sum, chunks):
        """One engine step: n_dec decoding requests with total context ctx_sum, and prefill chunks
        given as (chunk_tokens, context_before, samples_a_token)."""
        n = n_dec + sum(c for c, _, _ in chunks)
        if n == 0:
            return 0.0
        n_sampled = n_dec + sum(1 for _, _, s in chunks if s)
        t = self.eff.step_overhead + self._linear(n, n_sampled) + self._decode_attn(n_dec, ctx_sum)
        for c, b, _ in chunks:
            t += self._prefill_attn(c, b)
        return t

    def decode_step_times(self, n, ctx_sums):
        """step_time for decode-only steps with n requests, vectorized over ctx_sums."""
        m, g, e, tp = self.model, self.gpu, self.eff, self.tp
        base = e.step_overhead + self._linear(n, n) + m.layers * e.attn_decode_const
        mem = ctx_sums * self.rank_kv_bytes / (g.hbm_bw * e.attn_decode)
        cmp = ctx_sums * m.attn_decode_flops_pair / (tp * g.flops * e.gemm)
        return base + np.maximum(mem, cmp)

    def kv_transfer_time(self, tokens, link_bw):
        """GPU-to-tier copy of `tokens` tokens: every TP rank moves its own share over its own link
        in parallel, so the time is one rank's share over one link."""
        return tokens * self.rank_kv_bytes / link_bw


MODELS = {
    "llama70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "llama8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "qwq32b": "Qwen/QwQ-32B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "qwen3-235b": "Qwen/Qwen3-235B-A22B",
    "qwen3-30b": "Qwen/Qwen3-30B-A3B",
    "dsr1": "deepseek-ai/DeepSeek-R1",
}


if __name__ == "__main__":
    for k, r in MODELS.items():
        m = load_model(r)
        print(f"{k:11s} total {m.params_total / 1e9:6.1f}B active {m.params_active / 1e9:5.1f}B "
              f"KV {m.kv_bytes_per_token / 1024:6.1f} KiB/token mla={m.mla}")
