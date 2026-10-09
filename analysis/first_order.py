"""First-order model of what keeping preempted KV can save, checked against the simulator.

Let R be recomputed tokens per generated token and rho the GPU time of one generated (decode)
token over that of one recomputed (prefill) token. Then the recompute share of GPU time is
w = R / (rho + R), and a tier that removes all recomputation raises throughput by w / (1 - w).
rho is large because prefill is compute-bound while decode with a full KV cache is bound by HBM
bandwidth; under vLLM's FCFS preemption R stays below 1 (each preempted request recomputes its
context once per preemption and resumes first).

Input: results/gate2_saturation.csv. Run: python analysis/first_order.py
"""
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"


def table():
    d = pd.read_csv(RESULTS / "gate2_saturation.csv")
    r = d[d.policy == "recompute"].copy()
    c_pre = r.recompute_time / r.recomputed_tokens
    c_dec = (r.busy_time - r.recompute_time) / r.out_tokens
    r["rho"] = c_dec / c_pre
    r["R"] = r.recomputed_per_out
    w = r.R / (r.rho + r.R)
    r["pred_gain"] = w / (1 - w)
    swap = d[d.policy == "swap"].set_index(["deploy", "wname", "kernel"]).throughput
    r["sim_gain"] = swap.loc[list(zip(r.deploy, r.wname, r.kernel))].values / r.throughput.values - 1
    return r[["deploy", "wname", "kernel", "R", "rho", "recompute_frac", "pred_gain", "sim_gain"]]


if __name__ == "__main__":
    t = table()
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))
    print(f"\ncorrelation {t.sim_gain.corr(t.pred_gain):.3f}, mean abs error "
          f"{(t.sim_gain - t.pred_gain).abs().mean() * 100:.2f} points")
    lo = t.rho.min()
    print(f"bound with the largest R ({t.R.max():.2f}) and smallest rho ({lo:.1f}): "
          f"gain <= {t.R.max() / lo:.1%}")
