"""Gate 2, part 7: sensitivity of the saturation gain to settings that change the cost ratio.

FP8 KV cache (halves KV bytes read per decode step, making decode cheaper relative to recompute),
3 ms per-step overhead, a slower tier link (25 GB/s per rank, a CXL x8 device shared by two GPUs
gives about that), and the FA2-era prefill calibration combined with FP8 KV (the setting most
favorable to keeping KV).
"""
import itertools

from gate2_saturation import DEPLOY, workloads
from runner import run_all

PICK = ("70B-8xH100", "70B-8xH200", "70B-8xB200", "QwQ32B-4xH100", "R1-FP8-8xH200")
VARIANTS = {
    "base": dict(),
    "fp8kv": dict(kv_bytes=1),
    "overhead3ms": dict(overhead=3e-3),
    "tier25GBs": dict(tier_bw=25e9),
    "fp8kv_fa2": dict(kv_bytes=1, kernel="fa2"),
}


def configs():
    for (dep, m, g, tp, wb, mc), (vname, v), pol in itertools.product(DEPLOY, VARIANTS.items(), ("recompute", "swap")):
        if dep not in PICK or (vname == "tier25GBs" and pol == "recompute"):
            continue
        for wname, w in workloads(mc):
            if wname == "bo1":
                continue
            c = dict(deploy=dep, model=m, gpu=g, tp=tp, weight_bytes=wb, workload=w, wname=wname,
                     kernel="fa3", policy=pol, tier_gb=1e5, variant=vname)
            c.update(v)
            yield c


if __name__ == "__main__":
    df = run_all(list(configs()), "gate2_sensitivity", workers=12)
    print(df[["deploy", "wname", "variant", "policy", "throughput", "recompute_frac"]].to_string())
