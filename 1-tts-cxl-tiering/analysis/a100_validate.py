"""Real-system check on an A100 server: is keeping preempted KV worth it on real vLLM?

Measures on real vLLM, and predicts with kvsim calibrated on Vidur's A100 profiles:
1. rho, the GPU time of one decoded token over that of one recomputed (prefilled) token, the key
   parameter of the first-order model (microbench);
2. preemptions, recomputed tokens and throughput with recompute versus vLLM's native CPU
   offloading tier, on reasoning traces with every request submitted at once (run).

The server needs only this repository (without data/), a vLLM venv and the model weights; the
traces are generated here and kept in results/a100/. On a shared server every run samples
nvidia-smi and records other users' processes on the chosen GPUs, so contended runs can be
dropped. Recompute and offload runs alternate to spread drift over both.

  here:    python analysis/a100_validate.py traces
  server:  python analysis/a100_validate.py check
           python analysis/a100_validate.py microbench --model M --tp 1 --gpus 0
           python analysis/a100_validate.py run --model M --tp 1 --gpus 0 --trace bailian
  here:    python analysis/a100_validate.py compare
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "a100"
MAX_LEN = 32768
SEQS, BUDGET = 256, 8192
ENV = dict(VLLM_USE_FLASHINFER_SAMPLER="0", VLLM_ENABLE_V1_MULTIPROCESSING="0")
TRACES = {  # name: (source, MathArena config whose lengths are used)
    "bailian": ("bailian", None),
    "bo16_llama": ("matharena", "deepseek/deepseek_distill_70b"),
    "bo16_qwq": ("matharena", "qwen/qwq"),
}


def make_traces():
    sys.path.insert(0, str(Path(__file__).parent))
    import pandas as pd
    import workloads
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (src, mc) in TRACES.items():
        if src == "bailian":
            reqs = workloads.bailian(offline=True, max_len=MAX_LEN)[:400]
        else:
            reqs = workloads.matharena(mc, n_requests=12, n_samples=16, rate=1.0, seed=7, offline=True,
                                       max_len=MAX_LEN)
        pd.DataFrame({"prompt": [r.prompt for r in reqs], "out": [r.out for r in reqs],
                      "group": [r.group for r in reqs]}).to_csv(OUT / f"trace_{name}.csv", index=False)
        print(name, len(reqs), "requests,", sum(r.out for r in reqs), "output tokens")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def check():
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(
        gpus=sh("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader"),
        processes=sh("nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader"),
        topology=sh("nvidia-smi topo -m"),
        driver=sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
        host_mem=sh("free -g"), cpus=sh("nproc"), disk=sh("df -h ."),
        python=sys.version.split()[0],
    )
    try:
        import torch
        import vllm
        env.update(vllm=vllm.__version__, torch=torch.__version__, cuda=torch.version.cuda)
    except Exception as e:  # noqa: BLE001
        env["vllm"] = f"not importable: {e}"
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    for k, v in env.items():
        print(f"== {k}\n{v}")
    print("\nPick GPUs with near-zero memory use and utilization; pass them with --gpus.")


class GpuMonitor(threading.Thread):
    """Samples utilization, memory and other users' processes on the chosen GPUs."""

    def __init__(self, gpus):
        super().__init__(daemon=True)
        self.gpus, self.samples, self.stop = gpus, [], False
        self.mine = {os.getpid()}

    def run(self):
        uuids = sh(f"nvidia-smi -i {self.gpus} --query-gpu=uuid --format=csv,noheader").split()
        while not self.stop:
            util = sh(f"nvidia-smi -i {self.gpus} --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits")
            procs = [p.split(", ") for p in sh(
                "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader").splitlines() if p]
            foreign = sorted({int(p[1]) for p in procs if len(p) == 2 and p[0] in uuids and p[1].isdigit()
                              and int(p[1]) not in self.mine and not _is_child(int(p[1]))})
            self.samples.append(dict(t=time.time(), util=util, foreign=foreign))
            time.sleep(2)

    def summary(self):
        foreign = sorted({p for s in self.samples for p in s["foreign"]})
        return dict(n_samples=len(self.samples), foreign_pids=foreign, contended=bool(foreign))


def _is_child(pid):
    ppid = sh(f"ps -o ppid= -p {pid}").strip()
    return ppid.isdigit() and int(ppid) == os.getpid()


def _llm(a, **extra):
    from vllm import LLM
    kw = dict(model=a.model, tensor_parallel_size=a.tp, dtype="bfloat16", max_model_len=MAX_LEN,
              max_num_seqs=SEQS, max_num_batched_tokens=BUDGET, enable_prefix_caching=True,
              gpu_memory_utilization=a.mem_util, seed=0, disable_log_stats=False)
    if a.kv_blocks:
        kw["num_gpu_blocks_override"] = a.kv_blocks
    kw.update(extra)
    return LLM(**kw)


def _num_blocks(llm):
    for path in ("llm_engine.vllm_config.cache_config.num_gpu_blocks",
                 "llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool.num_gpu_blocks"):
        obj = llm
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            if obj:
                return int(obj)
        except AttributeError:
            continue
    return None


def _metrics(llm):
    out = {}
    for m in llm.get_metrics():
        if hasattr(m, "value"):
            out[m.name] = out.get(m.name, 0) + m.value
        elif hasattr(m, "sum"):
            out[m.name + ":sum"] = out.get(m.name + ":sum", 0) + m.sum
            out[m.name + ":count"] = out.get(m.name + ":count", 0) + m.count
    return out


def _count_scheduled():
    from vllm.v1.core.sched import scheduler as vsched
    box = [0]
    orig = vsched.Scheduler.schedule

    def counting(self, *x, **k):
        o = orig(self, *x, **k)
        box[0] += o.total_num_scheduled_tokens
        return o
    vsched.Scheduler.schedule = counting
    return box


def _generate(llm, prompts, outs):
    from vllm import SamplingParams
    params = [SamplingParams(max_tokens=int(o), min_tokens=int(o), ignore_eos=True, temperature=0.0) for o in outs]
    t0 = time.time()
    res = llm.generate(prompts, params, use_tqdm=False)
    return time.time() - t0, sum(len(r.outputs[0].token_ids) for r in res)


def _prompts(lengths, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    return [{"prompt_token_ids": rng.integers(1000, 30000, int(p)).tolist()} for p in lengths]


def one_run(a):
    """Child process: one configuration of `run` or `microbench`."""
    import pandas as pd
    box = _count_scheduled()
    mon = GpuMonitor(a.gpus)
    mon.start()
    extra = {}
    if a.config == "cpu_offload":
        extra = dict(kv_offloading_size=a.offload_gb, kv_offloading_backend="native")
    llm = _llm(a, **extra)
    blocks = _num_blocks(llm)
    res = dict(model=a.model, tp=a.tp, gpus=a.gpus, config=a.config, num_blocks=blocks, offload_gb=a.offload_gb,
               gpu_name=sh(f"nvidia-smi -i {a.gpus.split(',')[0]} --query-gpu=name,memory.total --format=csv,noheader"))
    if a.mode == "run":
        tr = pd.read_csv(OUT / f"trace_{a.trace}.csv")
        if a.limit:
            tr = tr.head(a.limit)
        box[0] = 0
        wall, gen = _generate(llm, _prompts(tr.prompt, 0), tr.out)
        res.update(trace=a.trace, limit=a.limit, rep=a.rep, wall_s=wall, out_tokens=gen, throughput=gen / wall,
                   scheduled_tokens=box[0], fresh_tokens=int(tr.prompt.sum() + tr.out.sum() - len(tr)),
                   metrics=_metrics(llm))
    else:
        cap = (blocks or 0) * 16
        rows = []
        for L in a.ctx:
            B = max(1, min(SEQS, int(0.8 * cap / (L + 160))))
            p = _prompts([L] * B, L)
            t1, _ = _generate(llm, p, [1] * B)          # prefill of B x L tokens
            # the same plus 128 decode steps, on new prompts so the prefix cache cannot help
            t3, _ = _generate(llm, _prompts([L] * B, L + 1), [129] * B)
            rows.append(dict(ctx=L, batch=B, prefill_s=t1, prefill_decode_s=t3,
                             c_pre=t1 / (B * L), step_s=(t3 - t1) / 128, c_dec=(t3 - t1) / 128 / B))
            rows[-1]["rho"] = rows[-1]["c_dec"] / rows[-1]["c_pre"]
            print(rows[-1], flush=True)
        res.update(rows=rows)
    mon.stop = True
    res["gpu_monitor"] = mon.summary()
    print("RESULT " + json.dumps(res), flush=True)


def launch(a, extra_args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus, **ENV)
    cmd = [sys.executable, __file__, "_one"] + extra_args
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    line = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")]
    if not line:
        print(r.stdout[-2000:], r.stderr[-4000:])
        raise SystemExit("run failed")
    return json.loads(line[-1][7:])


def tag(a):
    return f"{a.model.split('/')[-1]}_tp{a.tp}"


def run(a, argv):
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"run_{tag(a)}_{a.trace}.json"
    res = json.loads(out.read_text()) if out.exists() else []
    for rep in range(a.reps):
        for cfg in ("recompute", "cpu_offload"):
            r = launch(a, argv + ["--config", cfg, "--rep", str(rep), "--mode", "run"])
            res.append(r)
            out.write_text(json.dumps(res, indent=1))
            print(f"rep {rep} {cfg}: {r['throughput']:.0f} tok/s, preemptions "
                  f"{r['metrics'].get('vllm:num_preemptions')}, recomputed {r['scheduled_tokens'] - r['fresh_tokens']}, "
                  f"contended {r['gpu_monitor']['contended']}", flush=True)


def microbench(a, argv):
    OUT.mkdir(parents=True, exist_ok=True)
    r = launch(a, argv + ["--config", "recompute", "--mode", "micro"])
    (OUT / f"micro_{tag(a)}.json").write_text(json.dumps(r, indent=1))
    for row in r["rows"]:
        print(f"ctx {row['ctx']:6d} batch {row['batch']:4d}  recompute {row['c_pre'] * 1e6:7.2f} us/token  "
              f"decode {row['c_dec'] * 1e6:8.2f} us/token  rho {row['rho']:6.1f}")


def compare():
    sys.path.insert(0, str(Path(__file__).parent))
    import pandas as pd
    from calibrate import calibrated_eff
    from kvsim import Req, Sim, summarize
    from models import GPUS, Instance, load_model

    def inst(r):
        gpu = "A100-40" if "40" in r["gpu_name"].split(",")[-1] and "80" not in r["gpu_name"] else "A100"
        return Instance(load_model(r["model"]), GPUS[gpu], r["tp"], eff=calibrated_eff(gpu))

    for f in sorted(OUT.glob("micro_*.json")):
        r = json.loads(f.read_text())
        i = inst(r)
        print(f"\n{f.name} ({r['gpu_name']}), contended={r['gpu_monitor']['contended']}")
        for row in r["rows"]:
            B, L = row["batch"], row["ctx"]
            t = []
            for out in (1, 129):
                s = Sim(i, [Req(k, 0.0, L, out) for k in range(B)], num_blocks=r["num_blocks"], max_num_seqs=SEQS,
                        max_batched_tokens=BUDGET)
                s.run()
                t.append(s.t)
            c_pre, c_dec = t[0] / (B * L), (t[1] - t[0]) / 128 / B
            print(f"  ctx {L:6d} B {B:4d}  rho measured {row['rho']:6.1f} kvsim {c_dec / c_pre:6.1f}   "
                  f"recompute us/token {row['c_pre'] * 1e6:6.2f} vs {c_pre * 1e6:6.2f}   "
                  f"decode step ms {row['step_s'] * 1e3:6.2f} vs {(t[1] - t[0]) / 128 * 1e3:6.2f}")
    for f in sorted(OUT.glob("run_*.json")):
        runs = json.loads(f.read_text())
        df = pd.DataFrame([dict(config=r["config"], thr=r["throughput"], pre=r["metrics"].get("vllm:num_preemptions"),
                                rec=r["scheduled_tokens"] - r["fresh_tokens"], contended=r["gpu_monitor"]["contended"])
                           for r in runs])
        clean = df[~df.contended]
        print(f"\n{f.name}: {len(df)} runs, {len(df) - len(clean)} contended (dropped)")
        print(clean.groupby("config").agg(thr=("thr", "mean"), thr_sd=("thr", "std"), pre=("pre", "mean"),
                                          rec=("rec", "mean"), n=("thr", "size")).round(1).to_string())
        r0 = runs[0]
        tr = pd.read_csv(OUT / f"trace_{r0['trace']}.csv")
        if r0.get("limit"):
            tr = tr.head(r0["limit"])
        i = inst(r0)
        pred = {}
        for cfg, pol in (("recompute", "recompute"), ("cpu_offload", "lru")):
            blocks = next(r["num_blocks"] for r in runs if r["config"] == cfg)
            s = Sim(i, [Req(k, 0.0, int(p), int(o)) for k, (p, o) in enumerate(zip(tr.prompt, tr.out))], policy=pol,
                    tier_bytes=r0["offload_gb"] * 2**30, tier_bw=GPUS["A100"].host_bw, num_blocks=blocks,
                    max_num_seqs=SEQS, max_batched_tokens=BUDGET)
            s.run()
            pred[cfg] = summarize(s)
            print(f"  kvsim {cfg:11s}: {pred[cfg]['throughput']:.0f} tok/s, preemptions {pred[cfg]['preemptions']}, "
                  f"recomputed {pred[cfg]['recomputed_tokens']}")
        m = clean.groupby("config").thr.mean()
        if {"recompute", "cpu_offload"} <= set(m.index):
            print(f"  offload gain measured {(m['cpu_offload'] / m['recompute'] - 1) * 100:+.1f}%, kvsim (ideal LRU) "
                  f"{(pred['cpu_offload']['throughput'] / pred['recompute']['throughput'] - 1) * 100:+.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["traces", "check", "microbench", "run", "compare", "_one"])
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--trace", default="bailian", choices=list(TRACES))
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N requests of the trace")
    ap.add_argument("--mem-util", type=float, default=0.9)
    ap.add_argument("--kv-blocks", type=int, default=0, help="fix the KV cache size (blocks of 16 tokens)")
    ap.add_argument("--offload-gb", type=float, default=32.0)
    ap.add_argument("--ctx", type=int, nargs="+", default=[2048, 8192, 16384])
    ap.add_argument("--config", default="recompute")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--mode", dest="sub", default="run")
    a, _ = ap.parse_known_args()
    argv = [x for x in sys.argv[2:]]
    if a.mode == "_one":
        a.mode = a.sub
        one_run(a)
    elif a.mode == "traces":
        make_traces()
    elif a.mode == "check":
        check()
    elif a.mode == "microbench":
        microbench(a, argv)
    elif a.mode == "run":
        run(a, argv)
    else:
        compare()


if __name__ == "__main__":
    main()
