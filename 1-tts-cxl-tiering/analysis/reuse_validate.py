"""Real-vLLM test of the first-order model where recomputation is large (document QA).

Workload: D documents, Q questions per document, answers of `out` tokens, all submitted at once in
round-robin order over documents, so a document's KV has left a small GPU cache before its next
question. Two benchmarks: docqa (16 random-token documents of 4,096 tokens, a distinct 32-token
suffix per question) and longbench (16 real LongBench documents of 1,500 to 4,000 words from five
tasks, eight real questions each). A third, agent, replays real SWE-agent sessions (agent_sessions):
round k submits every session's k-th model turn at once, with the conversation so far as the prompt,
so a session's context leaves the GPU cache while the other sessions run. Three configurations:
  ideal       GPU KV cache large enough for every document (no recomputation)
  recompute   small GPU KV cache, evicted documents are recomputed
  cpu_offload the same small cache plus vLLM's native CPU offloading tier
Each run records wall time, generated tokens and the tokens the scheduler computed.

Model check. The ideal run only confirms the token accounting: with a small GPU cache fewer
requests fit at once, so decoding is slower per token than in the ideal run, and the comparison is
between recompute and cpu_offload at the same cache size. With G generated tokens, X_r the tokens
the recompute run computes beyond the offload run, and X_l the tokens the offload run loads instead
(X_l = X_r when every reuse hits the tier), the first-order model says
    T_recompute = G c_d + X_r c_r        T_offload = G c_d + X_l c_l
where c_r is the GPU time to recompute a token, taken from the independent prefill microbenchmark
on this GPU (results/a100/micro_*.json, interpolated to the document length), and c_l is the KV
bytes per token over the measured pinned host-to-GPU bandwidth. c_d comes from the offload run, and
the model then predicts the tier's gain, R (c_r - c_l) / (c_d + R c_l) with R = X_r / G, which is
compared with the measured gain T_recompute / T_offload - 1.

  python analysis/reuse_validate.py run --model M --bench docqa --outs 4 64 512 --reps 2
  python analysis/reuse_validate.py run --model M --bench agent --outs 0 --reps 2
  python analysis/reuse_validate.py report         all models and benchmarks into summary.csv
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from a100_validate import ENV, GpuMonitor, _count_scheduled, _generate, _llm, _num_blocks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("REUSE_DIR", ROOT / "results" / "reuse"))   # results/reuse_a100 for the A100 runs
D, L, Q, QLEN = 16, 4096, 8, 32
SMALL_BLOCKS = 2048                             # 32K tokens: about half of the documents
LONGBENCH = Path(os.environ.get("LONGBENCH_DIR", Path.home() / (
    ".cache/huggingface/hub/datasets--zai-org--LongBench/snapshots/"
    "5e628be450b7e67fb7ae6e201bd6d8f7056f7672/longbench_extracted/data")))
LB_TASKS = {"qasper": 4, "multifieldqa_en": 3, "2wikimqa": 3, "gov_report": 3, "lcc": 3}
AGENT = Path(os.environ.get("AGENT_DIR", ROOT / "data" / "agent"))   # nebius/SWE-agent-trajectories shard
AG_SESSIONS, AG_MAXCTX, AG_MAXOUT, AG_BLOCKS = 24, 24576, 512, 8192   # 8,192 blocks: 131K tokens


def short(model):
    return model.split("/")[-1]


def _longbench_docs(model):
    """16 real documents (1,500 to 4,000 words, the first by _id in each task) and eight questions per
    document: its own and seven others of the same task. Tokenized with the model's tokenizer."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    docs, questions = [], []
    for task, n in LB_TASKS.items():
        rows = sorted((json.loads(x) for x in open(LONGBENCH / f"{task}.jsonl")), key=lambda r: r["_id"])
        pool = [r for r in rows if 1500 <= r["length"] <= 4000][:n + Q]
        for i in range(n):
            ids = tok(pool[i]["context"], add_special_tokens=False)["input_ids"][:6144]
            qs = [pool[(i + k) % len(pool)]["input"] or "Write a one-page summary." for k in range(Q)]
            docs.append(ids)
            questions.append([tok("\n\nQuestion: " + q + "\nAnswer:", add_special_tokens=False)["input_ids"][:QLEN * 2]
                              for q in qs])
    return docs, questions


def agent_sessions(model):
    """Real SWE-agent sessions (swe-agent-llama-70b, submitted, by instance_id): the first AG_SESSIONS
    with at least 8 model turns. Each model turn is a call whose prompt is every earlier message,
    tokenized message by message with the model's tokenizer so that each prompt extends the previous
    one; its output length is the recorded turn's (at most AG_MAXOUT). A session ends before the call
    whose context would exceed AG_MAXCTX. Returns, per session, the list of (prompt ids, out)."""
    import pandas as pd
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    d = pd.read_parquet(AGENT / "swe0.parquet")
    d = d[(d.model_name == "swe-agent-llama-70b") & d.exit_status.str.startswith("submitted")]
    sessions = []
    for _, r in d.sort_values("instance_id").iterrows():
        ids, calls = [], []
        for m in r.trajectory:
            text = m["system_prompt"] if m["role"] == "system" and m["system_prompt"] else m["text"]
            t = tok(f"\n<{m['role']}>\n{text}", add_special_tokens=False)["input_ids"]
            if m["role"] == "ai":
                out = min(len(t), AG_MAXOUT)
                if len(ids) + out > AG_MAXCTX:
                    break
                calls.append((list(ids), out))
            ids += t
        if len(calls) >= 8:
            sessions.append(calls)
        if len(sessions) == AG_SESSIONS:
            break
    return sessions


def workload(out, bench="docqa", model=None):
    """Prompts in submission order and the document token lengths."""
    import numpy as np
    rng = np.random.default_rng(1)
    if bench == "agent":
        sessions = agent_sessions(model)
        rounds = [[s[k] for s in sessions if k < len(s)] for k in range(max(map(len, sessions)))]
        return rounds, None, [len(p) for s in sessions for p, _ in s], [len(s[-1][0]) + s[-1][1] for s in sessions]
    if bench == "docqa":
        docs = [rng.integers(1000, 30000, L).tolist() for _ in range(D)]
        questions = [[rng.integers(1000, 30000, QLEN).tolist() for _ in range(Q)] for _ in range(D)]
    else:
        docs, questions = _longbench_docs(model)
    prompts = [{"prompt_token_ids": docs[d] + questions[d][q]} for q in range(Q) for d in range(len(docs))]
    return prompts, [out] * len(prompts), [len(x) for x in docs], None


def kv_bytes_per_token(model):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model)
    c = getattr(c, "text_config", c)
    head = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    return 2 * c.num_hidden_layers * c.num_key_value_heads * head * 2      # K and V, BF16


def h2d_bandwidth(ndev=1):
    """Pinned host-to-GPU copy rate summed over ndev GPUs copying at once (with tensor parallelism
    each GPU loads its share of the KV over its own link)."""
    import torch
    xs = [torch.empty(1 << 30, dtype=torch.uint8).pin_memory() for _ in range(ndev)]
    ys = [torch.empty_like(x, device=f"cuda:{i}") for i, x in enumerate(xs)]

    def copy():
        for x, y in zip(xs, ys):
            y.copy_(x, non_blocking=True)
        for i in range(ndev):
            torch.cuda.synchronize(i)
    copy()
    t0 = time.time()
    for _ in range(5):
        copy()
    return 5 * ndev * xs[0].numel() / (time.time() - t0)


def _log_steps(path):
    """STEPLOG=<file>: record every scheduler step (time, prefill tokens, decode tokens, decoding
    requests), for the step-time regression in analysis/step_fit.py."""
    from vllm.v1.core.sched import scheduler as vsched
    steps = []
    orig = vsched.Scheduler.schedule

    def logged(self, *x, **k):
        o = orig(self, *x, **k)
        n = list(o.num_scheduled_tokens.values())
        steps.append((time.perf_counter(), sum(v for v in n if v > 1), sum(1 for v in n if v == 1)))
        return o
    vsched.Scheduler.schedule = logged
    return steps


def one(a):
    box = _count_scheduled()
    steps = _log_steps(os.environ["STEPLOG"]) if os.environ.get("STEPLOG") else None
    mon = GpuMonitor(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    mon.start()
    prompts, outs, lens, final = workload(a.out, a.bench, a.model)
    kvb = kv_bytes_per_token(a.model)
    held = sum(final) if final else sum(lens)       # tokens the GPU would hold to never recompute
    offload_gb = a.offload_gb or max(2.0, round(1.3 * held * kvb / 2**30 * 2) / 2)
    extra = dict(kv_offloading_size=offload_gb, kv_offloading_backend="native") \
        if a.config == "cpu_offload" else {}
    # documents: a request is at most 6,144 + 64 + 512 tokens; agent calls at most AG_MAXCTX
    extra["max_model_len"] = AG_MAXCTX + 16 if final else 8192
    # ideal: every document (session) plus the running requests fit (the 1.5B docqa runs used 6,144 blocks)
    a.kv_blocks = (held // 16 + 1024) if a.config == "ideal" else (AG_BLOCKS if final else SMALL_BLOCKS)
    llm = _llm(a, **extra)
    box[0] = 0
    if final:
        wall = gen = 0
        for rnd in prompts:
            w, g = _generate(llm, [{"prompt_token_ids": p} for p, _ in rnd], [o for _, o in rnd])
            wall, gen = wall + w, gen + g
    else:
        wall, gen = _generate(llm, prompts, outs)
    mon.stop = True
    res = dict(config=a.config, out=a.out, rep=a.rep, wall_s=wall, gen_tokens=gen, bench=a.bench,
               scheduled_tokens=box[0], num_blocks=_num_blocks(llm), model=a.model, doc_lens=lens, tp=a.tp,
               offload_gb=offload_gb if a.config == "cpu_offload" else 0,
               gpu_util=[s["util"] for s in mon.samples])
    if a.config == "ideal" and a.out == a.first_out and a.rep == 0:
        res.update(h2d_Bps=h2d_bandwidth(a.tp), kv_bytes_per_token=kvb)
    if steps is not None:
        Path(os.environ["STEPLOG"]).write_text(json.dumps(dict(config=a.config, out=a.out, model=a.model,
                                                               bench=a.bench, steps=steps)))
    print("RESULT " + json.dumps(res), flush=True)


def mixbench(a):
    """Recompute cost when prefill shares steps with decodes, as in the document runs: 7 requests
    decode 512 tokens over 4,096-token contexts, alone and with 16 extra 4,096-token prefills
    (one output token each); the added time per prefill token is c_r,mixed. Fresh random tokens
    every time, so the prefix cache never hits."""
    import numpy as np
    from a100_validate import _generate, _llm
    a.kv_blocks = 0
    llm = _llm(a, max_model_len=8192)
    rng = np.random.default_rng(7)

    def p(n):
        return [{"prompt_token_ids": rng.integers(1000, 30000, L).tolist()} for _ in range(n)]
    _generate(llm, p(4), [8] * 4)                                   # warm-up
    rows = []
    for rep in range(3):
        t_dec, _ = _generate(llm, p(7), [512] * 7)
        t_mix, _ = _generate(llm, p(7) + p(16), [512] * 7 + [1] * 16)
        rows.append(dict(rep=rep, t_dec=t_dec, t_mix=t_mix, c_r_mixed=(t_mix - t_dec) / (16 * L)))
    res = dict(model=a.model, rows=rows, c_r_mixed=float(np.median([r["c_r_mixed"] for r in rows])))
    (OUT / short(a.model)).mkdir(parents=True, exist_ok=True)
    (OUT / short(a.model) / "mixbench.json").write_text(json.dumps(res, indent=1))
    print(res, flush=True)


def run(a):
    d = OUT / short(a.model) / a.bench
    d.mkdir(parents=True, exist_ok=True)
    path = d / "runs.json"
    res = json.loads(path.read_text()) if path.exists() else []
    for rep in range(a.reps):
        for out in a.outs:
            for cfg in ("ideal", "recompute", "cpu_offload"):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("GPU", "0"), **ENV)   # GPU=<index> to pick one
                cmd = [sys.executable, __file__, "_one", "--config", cfg, "--out", str(out),
                       "--rep", str(rep), "--first-out", str(a.outs[0]), "--model", a.model,
                       "--mem-util", str(a.mem_util), "--offload-gb", str(a.offload_gb), "--bench", a.bench,
                       "--tp", str(a.tp)]
                r = subprocess.run(cmd, capture_output=True, text=True, env=env)
                line = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")]
                if not line:
                    print(r.stdout[-1500:], r.stderr[-3000:])
                    raise SystemExit("run failed")
                res.append(json.loads(line[-1][7:]))
                path.write_text(json.dumps(res, indent=1))
                x = res[-1]
                print(f"rep {rep} out {out:4d} {cfg:11s} {x['wall_s']:7.2f} s  generated {x['gen_tokens']}"
                      f"  computed {x['scheduled_tokens']}", flush=True)


def recompute_cost(model, lens, tp=1):
    """Recompute time per token from the prefill microbenchmark on this GPU, interpolated in log2 of
    the context between the measured contexts at each document length and weighted by length."""
    import numpy as np
    path = ROOT / "results" / "a100" / f"micro_{short(model)}_tp{tp}.json"
    rows = sorted(json.loads(path.read_text())["rows"], key=lambda r: r["ctx"])
    x = np.log2([r["ctx"] for r in rows])
    c = np.interp(np.log2(lens), x, [r["c_pre"] for r in rows])
    return float(np.dot(c, lens) / np.sum(lens))


def report():
    import numpy as np
    import pandas as pd
    rows = []
    for path in sorted(OUT.glob("*/*/runs.json")):
        runs = json.loads(path.read_text())
        df = pd.DataFrame(runs)
        if not {"ideal", "recompute", "cpu_offload"} <= set(df.config):
            print(f"skipped, incomplete: {path}")
            continue
        meta = next(r for r in runs if "h2d_Bps" in r)
        lens = meta.get("doc_lens") or [L] * D
        c_l = meta["kv_bytes_per_token"] / meta["h2d_Bps"]
        c_r = recompute_cost(meta["model"], lens, meta.get("tp", 1))
        mb = OUT / short(meta["model"]) / "mixbench.json"
        c_m = json.loads(mb.read_text())["c_r_mixed"] if mb.exists() else None
        for out, g in df.groupby("out"):
            if not {"ideal", "recompute", "cpu_offload"} <= set(g.config):
                continue                                   # this answer length is still running
            m = pd.DataFrame({c: dict(wall=x.wall_s.median(), sched=x.scheduled_tokens.median(),
                                      gen=x.gen_tokens.median(), n=len(x),
                                      spread=(x.wall_s.max() - x.wall_s.min()) / x.wall_s.median())
                              for c, x in g.groupby("config")}).T   # explicit: named aggregations
                                                                    # misbehave in newer pandas
            Tr, To = m.loc["recompute", "wall"], m.loc["cpu_offload", "wall"]
            X = m.loc["recompute", "sched"] - m.loc["cpu_offload", "sched"]   # recomputed instead of loaded
            G = m.loc["cpu_offload", "gen"]
            c_d = (To - X * c_l) / G
            R = X / G
            rows.append(dict(model=short(meta["model"]), bench=path.parent.name, out=out, R=R, rho=c_d / c_r,
                             c_r_us=c_r * 1e6, c_l_us=c_l * 1e6, c_d_ms=c_d * 1e3,
                             gain_measured=Tr / To - 1, gain_predicted=R * (c_r - c_l) / (c_d + R * c_l),
                             c_r_mixed_us=c_m * 1e6 if c_m else None,
                             gain_predicted_mixed=R * (c_m - c_l) / (c_d + R * c_l) if c_m else None,
                             T_recompute=Tr, T_offload=To, T_ideal=m.loc["ideal", "wall"], X=X,
                             offload_extra_vs_ideal=(m.loc["cpu_offload", "sched"] - m.loc["ideal", "sched"]),
                             max_spread=m.spread.max(), reps=int(m.n.min())))
    rep = pd.DataFrame(rows)
    rep["error"] = rep.gain_predicted / rep.gain_measured - 1
    rep["error_mixed"] = rep.gain_predicted_mixed / rep.gain_measured - 1
    # Load cost in place: the tier's time beyond the ideal cache per loaded token in the shortest-answer
    # run (decoding negligible, same tokens computed), applied to the longer answers; the agent runs
    # take it from the same model's docqa run, so nothing is fitted to them.
    rep["c_l_insitu_us"] = (rep.T_offload - rep.T_ideal) / rep.X * 1e6
    first = rep.loc[rep.groupby(["model", "bench"]).out.idxmin()].set_index(["model", "bench"]).c_l_insitu_us
    cl = np.array([first.get((m, "docqa" if b == "agent" else b), np.nan)
                   for m, b in zip(rep.model, rep.bench)]) * 1e-6
    rep["c_r_insitu_us"] = (rep.T_recompute - rep.T_ideal) / rep.X * 1e6
    rep["error_insitu"] = rep.X * (rep.c_r_us * 1e-6 - cl) / rep.T_offload / rep.gain_measured - 1
    rep.to_csv(OUT / "summary.csv", index=False)
    print(rep.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "report", "_one", "mixbench"])
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--outs", type=int, nargs="+", default=[4, 64, 512])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--mem-util", type=float, default=0.6)
    ap.add_argument("--offload-gb", type=float, default=0, help="0: 1.3 times the documents' KV")
    ap.add_argument("--bench", default="docqa", choices=["docqa", "longbench", "agent"])
    ap.add_argument("--config", default="ideal")
    ap.add_argument("--out", type=int, default=4)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--first-out", type=int, default=4)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel GPUs (GPU=0,1 for 2)")
    a = ap.parse_args()
    a.kv_blocks = 0
    if a.mode == "_one":
        one(a)
    elif a.mode == "run":
        run(a)
    elif a.mode == "mixbench":
        mixbench(a)
    else:
        report()


if __name__ == "__main__":
    main()
