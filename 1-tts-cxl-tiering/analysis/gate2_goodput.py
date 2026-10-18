"""Gate 2, part 6: goodput, the highest load that still meets a latency SLO, with and without a
tier for preempted KV. Fine load grid on the Bailian trace (real arrivals) and best-of-16.
SLOs: p99 TTFT <= 10 s and <= 60 s, and p90 time per output token <= 100 ms.
"""
import itertools

import numpy as np
import pandas as pd

from gate2_online import BAILIAN_OUT, BAILIAN_RATE, PICK
from gate2_saturation import DEPLOY
from runner import RESULTS, run_all

LOADS = tuple(np.round(np.arange(0.2, 0.81, 0.05), 2))


def configs():
    sat = pd.read_csv(RESULTS / "gate2_saturation.csv")
    sat = sat[(sat.policy == "recompute") & (sat.kernel == "fa3")].set_index(["deploy", "wname"])
    for dep, m, g, tp, wb, mc in DEPLOY:
        if dep not in PICK:
            continue
        thr_b = sat.loc[(dep, "bailian"), "throughput"]
        thr_m = sat.loc[(dep, "bo16"), "throughput"]
        out_m = sat.loc[(dep, "bo16"), "out_tokens"] / sat.loc[(dep, "bo16"), "n_done"]
        for load, pol in itertools.product(LOADS, ("recompute", "swap")):
            base = dict(deploy=dep, model=m, gpu=g, tp=tp, weight_bytes=wb, kernel="fa3", policy=pol,
                        tier_gb=1e5, load=float(load))
            yield {**base, "wname": "bailian",
                   "workload": dict(kind="bailian", speed=load * thr_b / BAILIAN_OUT / BAILIAN_RATE)}
            yield {**base, "wname": "bo16", "workload": dict(kind="matharena", model_config=mc, n_requests=256,
                                                              n_samples=16, rate=load * thr_m / (16 * out_m), seed=2)}


def goodput(df):
    rows = []
    for (dep, w, pol), g in df.groupby(["deploy", "wname", "policy"]):
        g = g.sort_values("load")
        row = dict(deploy=dep, wname=w, policy=pol)
        for name, ok in (("ttft10", g.ttft_p99 <= 10), ("ttft60", g.ttft_p99 <= 60)):
            # highest load such that it and every lower load meet the SLO
            bad = np.where(~ok.to_numpy())[0]
            row[name] = g.load.iloc[bad[0] - 1] if len(bad) and bad[0] > 0 else (g.load.max() if not len(bad) else 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_all(list(configs()), "gate2_goodput", workers=12)
    gp = goodput(df)
    gp.to_csv(RESULTS / "gate2_goodput_summary.csv", index=False)
    print(gp.to_string())
