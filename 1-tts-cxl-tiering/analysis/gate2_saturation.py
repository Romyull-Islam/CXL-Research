"""Gate 2, part 1: saturation throughput with recompute vs keeping preempted KV (swap tier).

Every request arrives at t=0 (offline batch), the regime with the most preemption. The gain of the
swap tier over recompute here bounds what a CXL (or host DRAM) tier for preempted KV can buy.
"""
import itertools

from runner import run_all

DEPLOY = [  # name, model, gpu, tp, weight bytes, MathArena config with the same architecture
    ("70B-8xH100", "llama70b", "H100", 8, 2, "deepseek/deepseek_distill_70b"),
    ("70B-8xH200", "llama70b", "H200", 8, 2, "deepseek/deepseek_distill_70b"),
    ("70B-8xB200", "llama70b", "B200", 8, 2, "deepseek/deepseek_distill_70b"),
    ("QwQ32B-2xH100", "qwq32b", "H100", 2, 2, "qwen/qwq"),
    ("QwQ32B-4xH100", "qwq32b", "H100", 4, 2, "qwen/qwq"),
    ("Qwen3-235B-FP8-8xH100", "qwen3-235b", "H100", 8, 1, "qwen/qwen3_235b_a22b"),
    ("R1-FP8-8xH200", "dsr1", "H200", 8, 1, "deepseek/deepseek_r1_0528"),
]


def workloads(mc):
    yield "bailian", dict(kind="bailian", offline=True)
    for n in (1, 16, 64):
        yield f"bo{n}", dict(kind="matharena", model_config=mc, n_requests=max(2048 // n, 32), n_samples=n,
                             rate=1.0, seed=1, offline=True)


def configs():
    for (dep, m, g, tp, wb, mc), kernel, pol in itertools.product(DEPLOY, ("fa3", "fa2"), ("recompute", "swap")):
        for wname, w in workloads(mc):
            yield dict(deploy=dep, model=m, gpu=g, tp=tp, weight_bytes=wb, workload=w, wname=wname,
                       kernel=kernel, policy=pol, tier_gb=1e5)


if __name__ == "__main__":
    df = run_all(list(configs()), "gate2_saturation", workers=16)
    print(df[["deploy", "wname", "kernel", "policy", "throughput", "recompute_frac", "wall_s"]].to_string())
