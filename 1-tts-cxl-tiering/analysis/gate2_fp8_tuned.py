"""Gate 2, part 8: the most favorable case (FP8 KV cache, offline best-of-N on 70B) against a
recompute baseline with tuned max_num_seqs, plus the tier capacity that case needs."""
import itertools

from gate2_saturation import DEPLOY, workloads
from models import GPUS, MODELS, Instance, load_model
from runner import run_all

PICK = ("70B-8xH100", "70B-8xH200", "70B-8xB200")


def configs():
    for dep, m, g, tp, wb, mc in DEPLOY:
        if dep not in PICK:
            continue
        hbm = Instance(load_model(MODELS[m], kv_dtype_bytes=1), GPUS[g], tp, weight_bytes=wb).kv_capacity_bytes / 1e9
        for wname, w in workloads(mc):
            if wname not in ("bo16", "bo64"):
                continue
            base = dict(deploy=dep, model=m, gpu=g, tp=tp, weight_bytes=wb, workload=w, wname=wname,
                        kernel="fa3", kv_bytes=1)
            for seqs in (64, 96, 128, 192, 256, 512, 1024):
                yield {**base, "policy": "recompute", "max_num_seqs": seqs, "tier_gb": 0}
            for pol, x in itertools.product(("swap", "lru"), (0.25, 0.5, 1.0, 2.0, 4.0, 100.0)):
                yield {**base, "policy": pol, "max_num_seqs": 1024, "tier_mult": x, "tier_gb": x * hbm}


if __name__ == "__main__":
    df = run_all(list(configs()), "gate2_fp8_tuned", workers=12)
    print(df[["deploy", "wname", "policy", "max_num_seqs", "tier_gb", "throughput"]].to_string())
