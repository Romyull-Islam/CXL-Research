"""Real-vLLM test of the first-order model where recomputation is large (document QA).

Workload: D documents of L tokens, Q questions per document (a distinct 32-token suffix each),
answers of `out` tokens, all submitted at once in round-robin order over documents, so a
document's KV has left a small GPU cache before its next question. Three configurations:
  ideal       GPU KV cache large enough for every document (no recomputation)
  recompute   small GPU KV cache, evicted documents are recomputed
  cpu_offload the same small cache plus vLLM's native CPU offloading tier
Each run records wall time, generated tokens and the tokens the scheduler computed.

Model check. The ideal run only confirms the token accounting: with a small GPU cache fewer
requests fit at once, so decoding is slower per token than in the ideal run, and the comparison is
between recompute and cpu_offload at the same cache size. With G generated tokens, X_r the tokens
the recompute run computes beyond the offload run, and X_l the tokens the offload run loads instead
(X_l = X_r when every reuse hits the tier), the first-order model says
    T_recompute = G c_d + X_r c_r        T_offload = G c_d + X_l c_l
where c_r is the GPU time to recompute a token, taken from the independent prefill microbenchmark
on this GPU (results/a100/micro_*.json, interpolated to the document length), and c_l is the KV
bytes per token over the measured pinned host-to-GPU bandwidth. c_d comes from the offload run, and
the model then predicts the tier's gain, R (c_r - c_l) / (c_d + R c_l) with R = X_r / G, which is
compared with the measured gain T_recompute / T_offload - 1.

  python analysis/reuse_validate.py run --out 4 64 512 --reps 2
  python analysis/reuse_validate.py report
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from a100_validate import ENV, GpuMonitor, _count_scheduled, _generate, _llm, _num_blocks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "reuse"
D, L, Q, QLEN = 16, 4096, 8, 32
SMALL_BLOCKS, BIG_BLOCKS = 2048, 6144          # 32K tokens (8 documents) and 96K tokens (all)


def workload(out):
    import numpy as np
    rng = np.random.default_rng(1)
    docs = [rng.integers(1000, 30000, L).tolist() for _ in range(D)]
    prompts = []
    for q in range(Q):
        for d in range(D):
            prompts.append({"prompt_token_ids": docs[d] + rng.integers(1000, 30000, QLEN).tolist()})
    return prompts, [out] * len(prompts)


def kv_bytes_per_token(model):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model)
    head = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    return 2 * c.num_hidden_layers * c.num_key_value_heads * head * 2      # K and V, BF16


def h2d_bandwidth():
    import torch
    x = torch.empty(1 << 30, dtype=torch.uint8).pin_memory()
    y = torch.empty_like(x, device="cuda")
    y.copy_(x, non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        y.copy_(x, non_blocking=True)
    torch.cuda.synchronize()
    return 5 * x.numel() / (time.time() - t0)


def one(a):
    box = _count_scheduled()
    mon = GpuMonitor("0")
    mon.start()
    extra = dict(kv_offloading_size=a.offload_gb, kv_offloading_backend="native") \
        if a.config == "cpu_offload" else {}
    extra["max_model_len"] = 8192          # a request is at most 4,096 + 32 + 512 tokens
    a.kv_blocks = BIG_BLOCKS if a.config == "ideal" else SMALL_BLOCKS
    llm = _llm(a, **extra)
    prompts, outs = workload(a.out)
    box[0] = 0
    wall, gen = _generate(llm, prompts, outs)
    mon.stop = True
    res = dict(config=a.config, out=a.out, rep=a.rep, wall_s=wall, gen_tokens=gen,
               scheduled_tokens=box[0], num_blocks=_num_blocks(llm), model=a.model,
               gpu_util=[s["util"] for s in mon.samples])
    if a.config == "ideal" and a.out == a.first_out and a.rep == 0:
        res.update(h2d_Bps=h2d_bandwidth(), kv_bytes_per_token=kv_bytes_per_token(a.model))
    print("RESULT " + json.dumps(res), flush=True)


def run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "runs.json"
    res = json.loads(path.read_text()) if path.exists() else []
    for rep in range(a.reps):
        for out in a.outs:
            for cfg in ("ideal", "recompute", "cpu_offload"):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", **ENV)
                cmd = [sys.executable, __file__, "_one", "--config", cfg, "--out", str(out),
                       "--rep", str(rep), "--first-out", str(a.outs[0]), "--model", a.model,
                       "--mem-util", str(a.mem_util), "--offload-gb", str(a.offload_gb)]
                r = subprocess.run(cmd, capture_output=True, text=True, env=env)
                line = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")]
                if not line:
                    print(r.stdout[-1500:], r.stderr[-3000:])
                    raise SystemExit("run failed")
                res.append(json.loads(line[-1][7:]))
                path.write_text(json.dumps(res, indent=1))
                x = res[-1]
                print(f"rep {rep} out {out:4d} {cfg:11s} {x['wall_s']:7.2f} s  generated {x['gen_tokens']}"
                      f"  computed {x['scheduled_tokens']}", flush=True)


def recompute_cost(model, L):
    """Recompute time per token at document length L from the prefill microbenchmark on this GPU,
    interpolated in log2(L) between the measured contexts."""
    import numpy as np
    path = ROOT / "results" / "a100" / f"micro_{model.split('/')[-1]}_tp1.json"
    rows = sorted(json.loads(path.read_text())["rows"], key=lambda r: r["ctx"])
    x = np.log2([r["ctx"] for r in rows])
    return float(np.interp(np.log2(L), x, [r["c_pre"] for r in rows]))


def report():
    import pandas as pd
    runs = json.loads((OUT / "runs.json").read_text())
    df = pd.DataFrame(runs)
    meta = next(r for r in runs if "h2d_Bps" in r)
    c_l = meta["kv_bytes_per_token"] / meta["h2d_Bps"]
    c_r = recompute_cost(meta["model"], L)
    rows = []
    for out, g in df.groupby("out"):
        m = g.groupby("config").agg(wall=("wall_s", "median"), sched=("scheduled_tokens", "median"),
                                    gen=("gen_tokens", "median"), n=("wall_s", "size"),
                                    spread=("wall_s", lambda s: (s.max() - s.min()) / s.median()))
        Tr, To = m.loc["recompute", "wall"], m.loc["cpu_offload", "wall"]
        X = m.loc["recompute", "sched"] - m.loc["cpu_offload", "sched"]   # recomputed instead of loaded
        G = m.loc["cpu_offload", "gen"]
        c_d = (To - X * c_l) / G
        R = X / G
        rows.append(dict(out=out, R=R, rho=c_d / c_r, c_d_ms=c_d * 1e3, gain_measured=Tr / To - 1,
                         gain_predicted=R * (c_r - c_l) / (c_d + R * c_l),
                         T_recompute=Tr, T_offload=To, T_ideal=m.loc["ideal", "wall"],
                         offload_extra_vs_ideal=(m.loc["cpu_offload", "sched"] - m.loc["ideal", "sched"]),
                         max_spread=m.spread.max(), reps=int(m.n.min())))
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "summary.csv", index=False)
    print(f"c_r = {c_r * 1e6:.2f} us/token (prefill microbenchmark at {L} tokens), c_l = "
          f"{c_l * 1e6:.3f} us/token ({meta['kv_bytes_per_token']} B/token at {meta['h2d_Bps'] / 1e9:.1f} GB/s)")
    print(rep.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "report", "_one"])
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--outs", type=int, nargs="+", default=[4, 64, 512])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--mem-util", type=float, default=0.6)
    ap.add_argument("--offload-gb", type=float, default=4.0)
    ap.add_argument("--config", default="ideal")
    ap.add_argument("--out", type=int, default=4)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--first-out", type=int, default=4)
    a = ap.parse_args()
    a.tp, a.kv_blocks = 1, 0
    if a.mode == "_one":
        one(a)
    elif a.mode == "run":
        run(a)
    else:
        report()


if __name__ == "__main__":
    main()
