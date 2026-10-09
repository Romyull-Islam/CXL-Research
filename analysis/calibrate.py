"""Fit the step-time model's efficiencies to measured H100 kernel times and report the error.

Data: Vidur's profiling CSVs (github.com/microsoft/vidur, data/profiling, commit abae7f6),
copied to data/vidur by fetch_data.sh. They hold median per-layer times of each operator, measured
on H100 with vLLM-era kernels (FlashAttention 2), for KV lengths up to 4K tokens and batches up to
128. The fit uses Llama-2-70B; the other profiled models are held out as validation.

Prefill attention efficiency grows with work per launch and Vidur's FA2-era kernels reach only
10 to 25% of peak in the profiled range. Current vLLM on Hopper uses FlashAttention 3, which
reports about 600 TFLOP/s (0.6 of peak) for long causal BF16 sequences (Shah et al., NeurIPS'24).
Two calibrations are kept: "fa2" (fitted saturation curve) and "fa3" (same curve shape with a 0.6
asymptote). fa2 makes recompute more expensive, so it is the setting that favors keeping KV.

Run: python analysis/calibrate.py
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from models import Eff

PROF = Path(__file__).resolve().parent.parent / "data" / "vidur" / "profiling"
F, BW = 989e12, 3.35e12
FIT_MODEL = "meta-llama/Llama-2-70b-hf"
GEMM_OPS = ["attn_pre_proj", "attn_post_proj", "mlp_up_proj", "mlp_down_proj"]
ELEM_OPS = ["input_layernorm", "attn_rope", "post_attention_layernorm", "mlp_act", "add"]


def _models():
    return sorted(str(p.parent.relative_to(PROF / "compute" / "h100"))
                  for p in (PROF / "compute" / "h100").glob("*/*/mlp.csv"))


def _mlp(model):
    m = pd.read_csv(PROF / "compute" / "h100" / model / "mlp.csv")
    m["tp"] = m.num_tensor_parallel_workers
    m["n"] = m.num_tokens
    hd = m.n_embd / m.n_head
    m["P"] = (m.n_embd * m.n_head * hd * 2 + 2 * m.n_embd * m.n_kv_head * hd
              + np.where(m.use_gated_mlp, 3, 2) * m.n_embd * m.n_expanded_embd)
    m["t_gemm"] = sum(m[f"time_stats.{o}.median"] for o in GEMM_OPS) / 1e3
    m["t_elem"] = sum(m[f"time_stats.{o}.median"] for o in ELEM_OPS if f"time_stats.{o}.median" in m) / 1e3
    return m


def _attn(model):
    a = pd.read_csv(PROF / "compute" / "h100" / model / "attention.csv")
    a["tp"] = a.num_tensor_parallel_workers
    a["hd"] = a.n_embd / a.n_q_head
    a["kvh_rank"] = np.maximum(a.n_kv_head / a.tp, 1)
    a["qh_rank"] = a.n_q_head / a.tp
    return a


def gemm_time(p, P, n, tp):
    eg, em = p
    return np.maximum(2 * P * n / tp / (F * eg), P * 2 / tp / (BW * em))


def elem_time(p, n, tp):
    c, u, s = p
    return c + n * (u + s / tp)


def decode_time(p, a):
    eff, c = p
    nbytes = a.batch_size * a.kv_cache_size * 2 * a.kvh_rank * a.hd * 2
    return c + nbytes / (BW * eff)


def prefill_flops(a):
    ch, kv = a.prefill_chunk_size, a.kv_cache_size
    return 4 * a.qh_rank * a.hd * ch * (kv + (ch + 1) / 2)


def prefill_time(p, w):
    emax, w0 = p
    return w / (F * emax * w / (w + w0))


def fit():
    m = _mlp(FIT_MODEL)
    g = least_squares(lambda p: (gemm_time(p, m.P, m.n, m.tp) - m.t_gemm) / m.t_gemm, [0.7, 0.8],
                      bounds=([0.05, 0.05], [1, 1])).x
    e = least_squares(lambda p: (elem_time(p, m.n, m.tp) - m.t_elem) / m.t_elem, [1e-5, 1e-8, 1e-7],
                      bounds=([0, 0, 0], [1e-3, 1e-5, 1e-5])).x
    a = _attn(FIT_MODEL)
    d = a[~a.is_prefill.astype(bool)].copy()
    d["t"] = d["time_stats.attn_decode.median"] / 1e3
    d = d[d.batch_size * d.kv_cache_size > 0]
    dec = least_squares(lambda p: (decode_time(p, d) - d.t) / d.t, [0.8, 1e-5],
                        bounds=([0.05, 0], [1, 1e-3])).x
    pp = a[a.is_prefill.astype(bool)].copy()
    pp["t"] = pp["time_stats.attn_prefill.median"] / 1e3
    pp["w"] = prefill_flops(pp)
    pp = pp[pp.w > 0]
    pre = least_squares(lambda p: (prefill_time(p, pp.w) - pp.t) / pp.t, [0.3, 1e10],
                        bounds=([0.01, 1e6], [1, 1e13])).x
    ar = {}
    net = pd.read_csv(PROF / "network" / "h100_dgx" / "all_reduce.csv")
    for nw, x in net.groupby("num_workers"):
        x = x.groupby("size")[["time_stats.all_reduce.min", "time_stats.all_reduce.median"]].median() / 1e3
        big = x.index > 4e6
        slope = np.polyfit(x.index[big], x["time_stats.all_reduce.median"][big], 1)[0]
        # latency: lower envelope at small sizes (vLLM custom all-reduce sits near it)
        ar[int(nw)] = (float(x["time_stats.all_reduce.min"].iloc[:5].median()), 1 / slope)
    return dict(gemm=g, elem=e, dec=dec, pre=pre, ar=ar)


def make_eff(p, kernel="fa3", step_overhead=0.5e-3):
    emax = 0.60 if kernel == "fa3" else p["pre"][0]
    return Eff(gemm=p["gemm"][0], mem=p["gemm"][1], elem_const=p["elem"][0], elem_tok=p["elem"][1],
               elem_tok_shard=p["elem"][2], attn_decode=p["dec"][0], attn_decode_const=p["dec"][1],
               attn_prefill_max=emax, attn_prefill_w0=p["pre"][1],
               ar_lat={k: v[0] for k, v in p["ar"].items()}, ar_bw={k: v[1] for k, v in p["ar"].items()},
               step_overhead=step_overhead)


_CACHE = {}


def calibrated_eff(gpu="H100", kernel="fa3", step_overhead=0.5e-3):
    """Efficiencies for the step-time model. Fitted on H100; other GPUs reuse the fractions."""
    if "p" not in _CACHE:
        _CACHE["p"] = fit() if PROF.exists() else None
    if _CACHE["p"] is None:
        return replace(Eff(), step_overhead=step_overhead)
    return make_eff(_CACHE["p"], kernel, step_overhead)


def _err(pred, meas):
    r = np.abs(pred / meas - 1)
    return f"median {r.median() * 100:5.1f}%  p90 {r.quantile(0.9) * 100:5.1f}%  n={len(r)}"


def report():
    p = fit()
    print("fitted on", FIT_MODEL)
    print(f"  GEMM: {p['gemm'][0]:.3f} of peak FLOP/s, {p['gemm'][1]:.3f} of HBM bandwidth")
    print(f"  elementwise per layer: {p['elem'][0] * 1e6:.1f} us + n x ({p['elem'][1] * 1e9:.1f} + "
          f"{p['elem'][2] * 1e9:.1f}/tp) ns")
    print(f"  decode attention: {p['dec'][0]:.3f} of HBM bandwidth + {p['dec'][1] * 1e6:.1f} us per layer")
    print(f"  prefill attention: asymptote {p['pre'][0]:.3f} of peak, half rate at {p['pre'][1]:.2e} FLOPs per layer-rank")
    for k, (lat, bw) in sorted(p["ar"].items()):
        print(f"  all-reduce tp{k}: {lat * 1e6:.1f} us + bytes / {bw / 1e9:.0f} GB/s")
    print("\nper-operator error (Llama-2-70B is the fit; the rest are held out)")
    for model in _models():
        m = _mlp(model)
        s = f"{model:32s} GEMM {_err(gemm_time(p['gemm'], m.P, m.n, m.tp), m.t_gemm)}"
        s += f" | elem {_err(elem_time(p['elem'], m.n, m.tp), m.t_elem)}"
        print(s)
        a = _attn(model)
        d = a[~a.is_prefill.astype(bool)].copy()
        d["t"] = d["time_stats.attn_decode.median"] / 1e3
        d = d[d.batch_size * d.kv_cache_size >= 32 * 1024]
        pp = a[a.is_prefill.astype(bool)].copy()
        pp["t"] = pp["time_stats.attn_prefill.median"] / 1e3
        pp["w"] = prefill_flops(pp)
        pp = pp[pp.prefill_chunk_size >= 512]
        print(f"{'':32s} decode attn {_err(decode_time(p['dec'], d), d.t)}"
              f" | prefill attn (chunk>=512) {_err(prefill_time(p['pre'], pp.w), pp.t)}")
    # whole layer: GEMM + elementwise + attention for decode batches and prefill chunks
    print("\nwhole-layer error, Llama-2-70B and CodeLlama-34B, tp 1/2/4/8")
    for model in [FIT_MODEL, "codellama/CodeLlama-34b-Instruct-hf"]:
        m, a = _mlp(model), _attn(model)
        rows = []
        dd = a[~a.is_prefill.astype(bool)]
        for _, r in dd[(dd.batch_size % 16 == 0) & (dd.kv_cache_size % 512 == 0) & (dd.kv_cache_size > 0)].iterrows():
            mm = m[(m.tp == r.tp) & (m.n == r.batch_size)]
            if mm.empty:
                continue
            mm = mm.iloc[0]
            meas = mm.t_gemm + mm.t_elem + r["time_stats.attn_decode.median"] / 1e3
            pred = (gemm_time(p["gemm"], mm.P, mm.n, mm.tp) + elem_time(p["elem"], mm.n, mm.tp)
                    + decode_time(p["dec"], r))
            rows.append(("decode", pred, meas))
        pp = a[a.is_prefill.astype(bool) & (a.prefill_chunk_size >= 512)]
        for _, r in pp.iterrows():
            mm = m[(m.tp == r.tp) & (m.n == r.prefill_chunk_size)]
            if mm.empty:
                continue
            mm = mm.iloc[0]
            meas = mm.t_gemm + mm.t_elem + r["time_stats.attn_prefill.median"] / 1e3
            pred = (gemm_time(p["gemm"], mm.P, mm.n, mm.tp) + elem_time(p["elem"], mm.n, mm.tp)
                    + prefill_time(p["pre"], prefill_flops(r)))
            rows.append(("prefill", pred, meas))
        df = pd.DataFrame(rows, columns=["kind", "pred", "meas"])
        for k, x in df.groupby("kind"):
            print(f"  {model:36s} {k:8s} {_err(x.pred, x.meas)}  bias {((x.pred / x.meas).median() - 1) * 100:+.1f}%")
    return p


if __name__ == "__main__":
    report()
