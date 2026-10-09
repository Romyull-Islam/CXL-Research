"""Gate 2, part 3: online load. Bailian thinking trace (real arrivals, time-scaled) and best-of-16
problems arriving as a Poisson process, at fractions of the recompute saturation throughput.
Timelines are kept for the pooling analysis.
"""
import itertools

import pandas as pd

from gate2_saturation import DEPLOY
from runner import RESULTS, run_all

PICK = ("70B-8xH100", "70B-8xH200", "QwQ32B-4xH100", "Qwen3-235B-FP8-8xH100", "R1-FP8-8xH200")
LOADS = (0.5, 0.7, 0.85, 0.95, 1.05)
BAILIAN_RATE, BAILIAN_OUT = 10812 / 7191.7, 3651.0


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
            speed = load * thr_b / BAILIAN_OUT / BAILIAN_RATE
            base = dict(deploy=dep, model=m, gpu=g, tp=tp, weight_bytes=wb, kernel="fa3", policy=pol,
                        tier_gb=1e5, load=load, keep_timeline=True)
            yield {**base, "wname": "bailian", "workload": dict(kind="bailian", speed=speed)}
            rate = load * thr_m / (16 * out_m)
            yield {**base, "wname": "bo16", "workload": dict(kind="matharena", model_config=mc, n_requests=256,
                                                              n_samples=16, rate=rate, seed=2)}


if __name__ == "__main__":
    df = run_all(list(configs()), "gate2_online", workers=12)
    print(df[["deploy", "wname", "load", "policy", "throughput", "ttft_p99", "e2e_p50", "tier_peak_gb"]].to_string())
