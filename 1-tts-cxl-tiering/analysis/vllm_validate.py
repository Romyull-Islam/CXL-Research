"""Real-system check on vLLM V1: preemption count, recomputed tokens and the throughput gain of the
native CPU KV offloading tier, against kvsim's prediction for the same run.

Setup: Qwen2.5-0.5B-Instruct (BF16) on one RTX 4500 Ada, KV cache fixed with
num_gpu_blocks_override, max_num_seqs 256, max_num_batched_tokens 8192, prefix caching on.
Workload: the first 300 requests of the Bailian thinking trace with prompts divided by 8 and
outputs by 4 (total under 8192 tokens), all submitted at once, random prompt tokens, output
lengths forced with ignore_eos. Runs: low KV pressure (15,000 blocks, what fits at 0.2 GPU memory), recompute (default) and CPU
offloading. FlashInfer's sampler is disabled (its JIT build fails on this host).

kvsim's step time is not calibrated for this GPU, so the comparison uses counts only (preemptions,
recomputed and reloaded tokens); with every request submitted at once the schedule depends on
steps, not on time.

Needs a vLLM environment (tested with vLLM 0.31.0, CUDA 13). Run:
  python analysis/vllm_validate.py run      (in the vLLM venv; writes results/vllm_validate.json)
  python analysis/vllm_validate.py compare  (in the analysis venv; runs kvsim on the same trace)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACE = ROOT / "data" / "xcheck" / "vllm_trace.csv"
OUT = ROOT / "results" / "vllm_validate.json"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_REQ, MAX_LEN, BLOCKS = 300, 8192, 4096
COMMON = dict(dtype="bfloat16", max_model_len=MAX_LEN, max_num_seqs=256, max_num_batched_tokens=8192,
              enable_prefix_caching=True, gpu_memory_utilization=0.2, seed=0, disable_log_stats=False)
RUNS = {
    "low_pressure": dict(num_gpu_blocks_override=15000),
    "recompute": dict(num_gpu_blocks_override=BLOCKS),
    "cpu_offload": dict(num_gpu_blocks_override=BLOCKS, kv_offloading_size=4.0, kv_offloading_backend="native"),
}


def make_trace():
    import pandas as pd
    d = pd.read_json(ROOT / "data" / "bailian" / "qwen_thinking_blksz_16.jsonl", lines=True)
    d = d.sort_values("timestamp").head(N_REQ)
    p = (d.input_length // 8).clip(lower=16)
    o = (d.output_length // 4).clip(lower=1)
    o = o.where(p + o <= MAX_LEN, MAX_LEN - p)
    TRACE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prompt": p.values, "out": o.values}).to_csv(TRACE, index=False)


def one(name):
    import numpy as np
    import pandas as pd
    from vllm import LLM, SamplingParams
    from vllm.v1.core.sched import scheduler as vsched
    # count every token the engine computes (fresh and recomputed); needs the in-process engine
    scheduled = [0]
    orig = vsched.Scheduler.schedule

    def counting(self, *a, **k):
        out = orig(self, *a, **k)
        scheduled[0] += out.total_num_scheduled_tokens
        return out
    vsched.Scheduler.schedule = counting
    tr = pd.read_csv(TRACE)
    rng = np.random.default_rng(0)
    prompts = [{"prompt_token_ids": rng.integers(1000, 100000, int(p)).tolist()} for p in tr.prompt]
    params = [SamplingParams(max_tokens=int(o), min_tokens=int(o), ignore_eos=True, temperature=0.0)
              for o in tr.out]
    llm = LLM(model=MODEL, **COMMON, **RUNS[name])
    t0 = time.time()
    outs = llm.generate(prompts, params, use_tqdm=False)
    wall = time.time() - t0
    gen = sum(len(x.outputs[0].token_ids) for x in outs)
    metrics = {}
    for m in llm.get_metrics():
        if hasattr(m, "value"):
            metrics[m.name] = metrics.get(m.name, 0) + m.value
        elif hasattr(m, "sum"):
            metrics[m.name + ":sum"] = metrics.get(m.name + ":sum", 0) + m.sum
            metrics[m.name + ":count"] = metrics.get(m.name + ":count", 0) + m.count
    return dict(run=name, wall_s=wall, out_tokens=gen, throughput=gen / wall, scheduled_tokens=scheduled[0],
                metrics=metrics)


def run_all():
    make_trace()
    res = []
    for name in RUNS:
        env = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0", VLLM_ENABLE_V1_MULTIPROCESSING="0")
        r = subprocess.run([sys.executable, __file__, "one", name], capture_output=True, text=True, env=env)
        line = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")]
        if not line:
            print(r.stderr[-3000:])
            raise SystemExit(f"run {name} failed")
        res.append(json.loads(line[-1][7:]))
        print(name, {k: res[-1][k] for k in ("wall_s", "out_tokens", "throughput")})
    OUT.write_text(json.dumps(res, indent=1))


def compare():
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).parent))
    from kvsim import Req, Sim, summarize
    from models import GPUS, Instance, load_model
    res = {r["run"]: r for r in json.loads(OUT.read_text())}
    tr = pd.read_csv(TRACE)
    m = load_model(MODEL)
    for name, pol in (("recompute", "recompute"), ("cpu_offload", "lru")):
        reqs = [Req(i, 0.0, int(p), int(o)) for i, (p, o) in enumerate(zip(tr.prompt, tr.out))]
        sim = Sim(Instance(m, GPUS["H100"], 1), reqs, policy=pol, tier_bytes=4 * 2**30, num_blocks=BLOCKS,
                  max_num_seqs=256, max_batched_tokens=8192)
        sim.run()
        s = summarize(sim)
        mt = res[name]["metrics"]
        computed = res[name]["scheduled_tokens"]
        fresh = int(tr.prompt.sum() + tr.out.sum() - len(tr))
        print(f"{name:12s} preemptions kvsim {s['preemptions']} vLLM {mt.get('vllm:num_preemptions')}; "
              f"recomputed tokens kvsim {s['recomputed_tokens']} vLLM "
              f"{None if computed is None else int(computed - fresh)}; "
              f"loaded tokens kvsim {s['loaded_tokens']}")
    base = res["recompute"]["throughput"]
    print(f"measured throughput: low pressure {res['low_pressure']['throughput']:.0f}, recompute {base:.0f}, "
          f"cpu offload {res['cpu_offload']['throughput']:.0f} tok/s; offload gain "
          f"{(res['cpu_offload']['throughput'] / base - 1) * 100:.1f}%")


if __name__ == "__main__":
    if sys.argv[1] == "one":
        print("RESULT " + json.dumps(one(sys.argv[2])))
    elif sys.argv[1] == "run":
        run_all()
    else:
        compare()
