"""Gate 2, part 4: pooled vs per-instance capacity for preempted KV.

Eight instances behind a round-robin balancer. Bailian: the trace is split eight ways, so
cluster-wide bursts reach every instance at once. Best-of-16: independent Poisson streams. Each
instance keeps preempted KV in an unlimited swap tier; its occupancy is sampled every 10 s.
Per-instance provisioning needs the sum of the instances' peaks, a shared pool needs the peak of
the sum.
"""
import itertools
import json

import numpy as np
import pandas as pd

from gate2_online import BAILIAN_OUT, BAILIAN_RATE
from gate2_saturation import DEPLOY
from runner import RESULTS, run_all

PICK = ("70B-8xH100", "QwQ32B-4xH100", "R1-FP8-8xH200")
LOADS = (0.85, 0.95)
M = 8


def configs():
    sat = pd.read_csv(RESULTS / "gate2_saturation.csv")
    sat = sat[(sat.policy == "recompute") & (sat.kernel == "fa3")].set_index(["deploy", "wname"])
    for (dep, m, g, tp, wb, mc), load, i in itertools.product(DEPLOY, LOADS, range(M)):
        if dep not in PICK:
            continue
        base = dict(deploy=dep, model=m, gpu=g, tp=tp, weight_bytes=wb, kernel="fa3", policy="swap",
                    tier_gb=1e5, load=load, inst=i, keep_timeline=True)
        thr_b = sat.loc[(dep, "bailian"), "throughput"]
        speed = load * thr_b / BAILIAN_OUT / BAILIAN_RATE * M
        yield {**base, "wname": "bailian", "workload": dict(kind="bailian", speed=speed, shard=i, n_shards=M)}
        thr_m = sat.loc[(dep, "bo16"), "throughput"]
        out_m = sat.loc[(dep, "bo16"), "out_tokens"] / sat.loc[(dep, "bo16"), "n_done"]
        yield {**base, "wname": "bo16", "workload": dict(kind="matharena", model_config=mc, n_requests=128,
                                                          n_samples=16, rate=load * thr_m / (16 * out_m), seed=100 + i)}


def analyze(df):
    tl = pd.read_json(RESULTS / "gate2_pool_timelines.json")
    df = df.assign(timeline=tl.timeline.values)
    rows = []
    for (dep, w, load), g in df.groupby(["deploy", "wname", "load"]):
        t_end = min(json.loads(x)[-1][0] if isinstance(x, str) else x[-1][0] for x in g.timeline)
        grid = np.arange(0, t_end, 10.0)
        series = []
        for x, gb_per_block in zip(g.timeline, g.block_gb):
            x = np.array(json.loads(x) if isinstance(x, str) else x)
            idx = np.searchsorted(x[:, 0], grid, side="right") - 1
            series.append(x[np.maximum(idx, 0), 5] * gb_per_block)
        s = np.array(series)
        tot = s.sum(0)
        rows.append(dict(deploy=dep, wname=w, load=load, hbm_kv_gb=g.hbm_kv_gb.iloc[0],
                         per_inst_max_gb=s.max(1).mean(), sum_of_max_gb=s.max(1).sum(), pool_max_gb=tot.max(),
                         sum_of_p99_gb=np.percentile(s, 99, axis=1).sum(), pool_p99_gb=np.percentile(tot, 99),
                         saving_max=1 - tot.max() / s.max(1).sum(),
                         saving_p99=1 - np.percentile(tot, 99) / np.percentile(s, 99, axis=1).sum()))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_all(list(configs()), "gate2_pool", workers=12)
    out = analyze(df)
    out.to_csv(RESULTS / "gate2_pool_summary.csv", index=False)
    print(out.round(3).to_string())
