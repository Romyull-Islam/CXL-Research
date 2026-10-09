"""Gate 2, part 2: how much tier capacity keeps the gain, at saturation.

Swap (store on preemption) and LRU write-through (the vLLM OffloadingConnector CPU tier) with
capacity given as a multiple of the instance's HBM KV capacity.
"""
import itertools

from gate2_saturation import DEPLOY, workloads
from models import GPUS, MODELS, Instance, load_model
from runner import run_all

PICK = ("70B-8xH100", "70B-8xH200", "QwQ32B-4xH100", "R1-FP8-8xH200")
MULT = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)


def configs():
    for dep, m, g, tp, wb, mc in DEPLOY:
        if dep not in PICK:
            continue
        hbm = Instance(load_model(MODELS[m]), GPUS[g], tp, weight_bytes=wb).kv_capacity_bytes / 1e9
        for (wname, w), pol, x in itertools.product(list(workloads(mc)), ("swap", "lru"), MULT):
            if wname == "bo1":
                continue
            yield dict(deploy=dep, model=m, gpu=g, tp=tp, weight_bytes=wb, workload=w, wname=wname,
                       kernel="fa3", policy=pol, tier_mult=x, tier_gb=x * hbm)


if __name__ == "__main__":
    df = run_all(list(configs()), "gate2_capacity", workers=12)
    print(df[["deploy", "wname", "policy", "tier_mult", "throughput", "recompute_frac", "wall_s"]].to_string())
