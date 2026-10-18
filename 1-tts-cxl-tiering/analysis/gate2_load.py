"""Gate 2, part 2: latency and tier occupancy under open-loop load, and tier capacity curves.

Load sweep: the Bailian thinking trace with its real arrival times, sped up by `speed` (1.0 is
the trace's own rate, 1.5 requests/s). Each deployment is swept from light load to past its
saturation point; recompute is compared with an unlimited swap tier.

Capacity curves: at a load near saturation, the swap tier and the vLLM-style LRU write-through
tier are given capacities from 0 to 4x the instance's HBM KV capacity; we report how much of the
recompute work is removed at each size.
"""
import sys

import numpy as np
import pandas as pd

from runner import RESULTS, run_all

DEPLOY = {
    "70B-8xH100": dict(model="llama70b", gpu="H100", tp=8, weight_bytes=2),
    "QwQ32B-2xH100": dict(model="qwq32b", gpu="H100", tp=2, weight_bytes=2),
    "QwQ32B-4xH100": dict(model="qwq32b", gpu="H100", tp=4, weight_bytes=2),
    "Qwen3-235B-FP8-8xH100": dict(model="qwen3-235b", gpu="H100", tp=8, weight_bytes=1),
    "R1-FP8-8xH200": dict(model="dsr1", gpu="H200", tp=8, weight_bytes=1),
}
TRACE_RATE = 10812 / 7191.7      # requests per second in the Bailian thinking trace
MEAN_OUT = 3651.0                # mean output tokens per request in the trace


def saturation_speed():
    """Trace speed at which each deployment saturates, from gate2_saturation.csv (fa3)."""
    s = pd.read_csv(RESULTS / "gate2_saturation.csv")
    s = s[(s.wname == "bailian") & (s.kernel == "fa3") & (s.policy == "recompute")]
    return {r.deploy: r.throughput / MEAN_OUT / TRACE_RATE for r in s.itertuples()}


def load_configs():
    sat = saturation_speed()
    for dep, d in DEPLOY.items():
        for frac in (0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3):
            for pol in ("recompute", "swap"):
                yield dict(deploy=dep, **d, policy=pol, tier_gb=1e5, load=frac, keep_timeline=True,
                           workload=dict(kind="bailian", speed=round(sat[dep] * frac, 4)))


def capacity_configs():
    sat = saturation_speed()
    for dep, d in DEPLOY.items():
        for frac in (0.9, 1.1):
            w = dict(kind="bailian", speed=round(sat[dep] * frac, 4))
            yield dict(deploy=dep, **d, policy="recompute", tier_gb=0, load=frac, tier_x=0.0, workload=w)
            for pol in ("swap", "lru"):
                for x in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
                    yield dict(deploy=dep, **d, policy=pol, tier_x=x, load=frac, workload=w)


def with_tier_gb(cfgs):
    from runner import build_instance
    out = []
    for c in cfgs:
        if "tier_x" in c and c["policy"] != "recompute":
            c["tier_gb"] = c["tier_x"] * build_instance(c).kv_capacity_bytes / 1e9
        out.append(c)
    return out


if __name__ == "__main__":
    part = sys.argv[1]
    if part == "load":
        df = run_all(list(load_configs()), "gate2_load", workers=12)
    else:
        # near-saturation online capacity curve; gate2_capacity.py holds the saturation one
        df = run_all(with_tier_gb(list(capacity_configs())), "gate2_load_capacity", workers=12)
    print(df.drop(columns=[c for c in df if c == "workload"]).round(3).to_string())
