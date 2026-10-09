"""Gate 2, part 5: can admission control alone remove the preemption cost?

Offline (all requests at t=0) with recompute preemption and smaller max_num_seqs, compared with
the default max_num_seqs=1024 plus an unlimited swap tier.
"""
import itertools

from gate2_saturation import DEPLOY, workloads
from runner import run_all

PICK = ("70B-8xH100", "70B-8xH200", "QwQ32B-4xH100", "R1-FP8-8xH200")
SEQS = (32, 48, 64, 96, 128, 192, 256, 512, 1024)


def configs():
    for dep, m, g, tp, wb, mc in DEPLOY:
        if dep not in PICK:
            continue
        for (wname, w), seqs in itertools.product(list(workloads(mc)), SEQS):
            if wname == "bo1":
                continue
            yield dict(deploy=dep, model=m, gpu=g, tp=tp, weight_bytes=wb, workload=w, wname=wname,
                       kernel="fa3", policy="recompute", max_num_seqs=seqs)


if __name__ == "__main__":
    df = run_all(list(configs()), "gate2_maxseqs", workers=12)
    print(df[["deploy", "wname", "max_num_seqs", "throughput", "preemptions", "recompute_frac"]].to_string())
