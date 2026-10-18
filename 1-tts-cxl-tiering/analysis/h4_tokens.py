"""Per-sample token counts from the HuggingFace H4 test-time-scaling completions (MATH-500).

Reads the best-of-N, beam-search and DVTS completions of Llama-3.2-1B/3B-Instruct from data/h4/
(see fetch_data.sh) and writes data/h4/h4_tokens.parquet with one row per
(model, strategy, n, seed, problem, sample). Prompt tokens use the search-and-learn system prompt
and chat template (src/sal/config.py); vLLM tokenizes the templated string with a BOS token.
completion_tokens is vLLM's count stored in the data (best-of-N, beam search; includes EOS) or,
for DVTS, which stores -1, the sum of per-step token counts from the Llama-3.2 tokenizer.
Steps are the "\\n\\n"-terminated segments that beam search and DVTS generate one per iteration.

Usage: python analysis/h4_tokens.py [--summary-only]
"""
import glob
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = "data/h4"
OUT = f"{ROOT}/h4_tokens.parquet"
STRATEGY = {"best-of-N": "best_of_n", "beam-search": "beam_search", "DVTS": "dvts"}

# search-and-learn src/sal/config.py, Config.system_prompt
SYSTEM_PROMPT = (
    "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\n"
    "Provide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\n"
    "Use this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n"
    "## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\n"
    "Regardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. "
    "I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."
)


def prompt_text(problem):
    """Output of Config.custom_chat_template for [system, user] with add_generation_prompt=True."""
    return ("<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n"
            "Today Date: 26 Jul 2024\n\n" + SYSTEM_PROMPT.strip() + "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n" + problem + "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")


def load_tokenizer():
    path = hf_hub_download("unsloth/Llama-3.2-1B-Instruct", "tokenizer.json",
                           local_dir=f"{ROOT}/tokenizer/llama32", token=False)
    tok = Tokenizer.from_file(path)
    assert tok.get_vocab_size() == 128256 and tok.token_to_id("<|begin_of_text|>") == 128000
    return tok


def split_steps(text):
    """Segments ending in "\\n\\n" (the stop string), plus a final segment without it."""
    parts = text.split("\n\n")
    steps = [p + "\n\n" for p in parts[:-1]]
    if parts[-1]:
        steps.append(parts[-1])
    return steps


def configs():
    for repo in sorted(glob.glob(f"{ROOT}/Llama-3.2-*-Instruct-*-completions")):
        name = os.path.basename(repo)
        model = name.split("-")[2]
        strategy = STRATEGY[re.search(r"Instruct-(.+)-completions", name).group(1)]
        for cfg in sorted(glob.glob(f"{repo}/HuggingFaceH4_MATH-500*")):
            c = os.path.basename(cfg)
            if c.endswith(("--evals", "--levels")) or "n-merged" in c:
                continue
            n = int(re.search(r"--n-(\d+)--", c).group(1))
            seed = int(re.search(r"--seed-(\d+)--", c).group(1))
            yield repo, model, strategy, n, seed, sorted(glob.glob(f"{cfg}/*.parquet"))


def model_levels(repo_glob):
    files = glob.glob(f"{ROOT}/{repo_glob}/*--levels/*.parquet")
    if not files:
        return {}
    t = pq.read_table(files[0], columns=["problem", "level_model_all"]).to_pydict()
    return dict(zip(t["problem"], t["level_model_all"]))


def tree_stats(steps_list, step_tok_list):
    """Per sample: is an exact duplicate of an earlier sample, tokens of its longest step prefix shared
    with another distinct sample, and tokens it adds to the prefix trie of the problem's samples."""
    keys = []
    for steps in steps_list:
        h, ks = 0, []
        for s in steps:
            h = hash((h, s))
            ks.append(h)
        keys.append(ks)
    full = [k[-1] if k else 0 for k in keys]
    seen_full, dup = set(), []
    for f in full:
        dup.append(f in seen_full)
        seen_full.add(f)
    cnt = Counter(k for ks, d in zip(keys, dup) if not d for k in ks)
    seen, shared, new = set(), [], []
    for ks, st, d in zip(keys, step_tok_list, dup):
        cum = np.cumsum([0] + st)
        depth = 0
        for i, k in enumerate(ks):
            if cnt[k] >= 2 or d:
                depth = i + 1
        shared.append(int(cum[depth]))
        added = 0
        for k, t in zip(ks, st):
            if k not in seen:
                seen.add(k)
                added += t
        new.append(added)
    return dup, shared, new


SCHEMA = pa.schema([
    ("model", pa.string()), ("strategy", pa.string()), ("n", pa.int32()), ("seed", pa.int16()),
    ("problem_id", pa.string()), ("level", pa.int8()), ("level_model", pa.int8()),
    ("sample_idx", pa.int32()), ("prompt_tokens", pa.int32()), ("completion_tokens", pa.int32()),
    ("completion_tokens_vllm", pa.int32()), ("retok_tokens", pa.int32()), ("n_steps", pa.int16()),
    ("step_tokens", pa.list_(pa.int32())), ("prm_last", pa.float32()), ("is_dup", pa.bool_()),
    ("shared_prefix_tokens", pa.int32()), ("trie_new_tokens", pa.int32()),
])


def process_batch(rb, tok, meta, levels, has_vllm):
    model, strategy, n, seed = meta
    d = rb.to_pydict()
    rows = {k: [] for k in SCHEMA.names}
    ptoks = [len(e.ids) for e in tok.encode_batch([prompt_text(p) for p in d["problem"]])]
    for j in range(rb.num_rows):
        comps = d["completions"][j]
        scores = d["scores"][j]
        m = len(comps)
        vllm = d["completion_tokens"][j] if has_vllm else None
        if strategy == "best_of_n":
            steps_list = None
            step_toks = [None] * m
            retok = [None] * m
            n_steps = [len(split_steps(c)) for c in comps]
            dup = [False] * m
            seen = set()
            for i, c in enumerate(comps):
                dup[i] = c in seen
                seen.add(c)
            shared = new = [None] * m
        else:
            steps_list = [split_steps(c) for c in comps]
            flat = [s for st in steps_list for s in st]
            lens = [len(e.ids) for e in tok.encode_batch(flat, add_special_tokens=False)] if flat else []
            step_toks, pos = [], 0
            for st in steps_list:
                step_toks.append(lens[pos:pos + len(st)])
                pos += len(st)
            retok = [sum(s) for s in step_toks]
            n_steps = [len(st) for st in steps_list]
            dup, shared, new = tree_stats(steps_list, step_toks)
        for i in range(m):
            v = vllm[i] if vllm is not None and vllm[i] >= 0 else None
            rows["model"].append(model)
            rows["strategy"].append(strategy)
            rows["n"].append(n)
            rows["seed"].append(seed)
            rows["problem_id"].append(d["unique_id"][j])
            rows["level"].append(d["level"][j])
            rows["level_model"].append(levels.get(d["problem"][j]))
            rows["sample_idx"].append(i)
            rows["prompt_tokens"].append(ptoks[j])
            rows["completion_tokens"].append(v if v is not None else retok[i])
            rows["completion_tokens_vllm"].append(v)
            rows["retok_tokens"].append(retok[i])
            rows["n_steps"].append(n_steps[i])
            rows["step_tokens"].append(step_toks[i])
            rows["prm_last"].append(scores[i][-1] if scores[i] else None)
            rows["is_dup"].append(dup[i])
            rows["shared_prefix_tokens"].append(shared[i])
            rows["trie_new_tokens"].append(new[i])
    return pa.table(rows, schema=SCHEMA)


def build():
    tok = load_tokenizer()
    levels = {m: model_levels(f"Llama-3.2-{m}-Instruct-best-of-N-completions") for m in ("1B", "3B")}
    writer = pq.ParquetWriter(OUT + ".tmp", SCHEMA, compression="zstd")
    for repo, model, strategy, n, seed, files in configs():
        cols = ["problem", "unique_id", "level", "completions", "scores"]
        has_vllm = strategy != "dvts"
        if has_vllm:
            cols.append("completion_tokens")
        rows = 0
        for f in files:
            pf = pq.ParquetFile(f)
            for rb in pf.iter_batches(batch_size=25 if n <= 256 else 8, columns=cols):
                t = process_batch(rb, tok, (model, strategy, n, seed), levels[model], has_vllm)
                writer.write_table(t)
                rows += t.num_rows
        print(f"{model} {strategy:11s} n={n:5d} seed={seed}: {rows} rows", flush=True)
    writer.close()
    os.replace(OUT + ".tmp", OUT)


def check_examples():
    """Hand check of three counts against vLLM's stored completion_tokens and the chat template."""
    tok = load_tokenizer()
    f = sorted(glob.glob(f"{ROOT}/Llama-3.2-1B-Instruct-beam-search-completions/*n-4--*seed-0*last/*.parquet"))[0]
    t = pq.read_table(f, columns=["problem", "completions", "completion_tokens"]).slice(0, 3).to_pydict()
    for p, comps, ct in zip(t["problem"], t["completions"], t["completion_tokens"]):
        whole = len(tok.encode(comps[0], add_special_tokens=False).ids)
        steps = sum(len(tok.encode(s, add_special_tokens=False).ids) for s in split_steps(comps[0]))
        print(f"prompt {len(tok.encode(prompt_text(p)).ids)} tokens; completion vLLM {ct[0]}, "
              f"retokenized whole {whole}, sum of steps {steps}, steps {len(split_steps(comps[0]))}")


def boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    i += 7
    depth, j = 1, i
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return s[i:j - 1]


def norm(a):
    if a is None:
        return None
    a = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", a)
    a = a.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("dfrac", "frac")
    a = a.replace("tfrac", "frac").replace("^\\circ", "").replace("^{\\circ}", "").replace("\\$", "")
    a = re.sub(r"\s+", "", a).rstrip(".")
    return a


def accuracy_check():
    """Recompute weighted best-of-N (sum of last-step PRM score per answer) at N=256, 1B, seed 0."""
    f = glob.glob(f"{ROOT}/Llama-3.2-1B-Instruct-best-of-N-completions/*n-256*seed-0--agg_strategy-last/*.parquet")[0]
    ev = glob.glob(f"{ROOT}/Llama-3.2-1B-Instruct-best-of-N-completions/*n-256*seed-0*evals/*.parquet")[0]
    t = pq.read_table(f, columns=["answer", "completions", "agg_scores", "pred_weighted@16",
                                  "pred_weighted@256"]).to_pydict()
    for n in (16, 256):
        mine = stored = 0
        for ans, comps, sc, pw in zip(t["answer"], t["completions"], t["agg_scores"], t[f"pred_weighted@{n}"]):
            tally = Counter()
            for c, s in zip(comps[:n], sc[:n]):
                a = norm(boxed(c))
                if a is not None:
                    tally[a] += s
            pick = tally.most_common(1)[0][0] if tally else None
            mine += pick == norm(ans)
            stored += norm(boxed(pw)) == norm(ans)
        print(f"1B best-of-N seed 0 weighted@{n}: recomputed {mine / 5:.1f}%, stored pred {stored / 5:.1f}% "
              f"(string match)")
    e = pq.read_table(ev).to_pandas().set_index("n").sort_index()
    print("evals parquet (Qwen2.5-Math grader):", e.acc_weighted.to_dict())


def summary():
    df = pd.read_parquet(OUT, columns=["model", "strategy", "n", "seed", "problem_id", "prompt_tokens",
                                       "completion_tokens", "is_dup", "trie_new_tokens", "n_steps"])
    g = df.groupby(["model", "strategy", "n"])
    s = g.completion_tokens.quantile([.5, .9, .99]).unstack()
    s.columns = ["p50", "p90", "p99"]
    s["max"] = g.completion_tokens.max()
    s["seeds"] = g.seed.nunique()
    s["problems"] = g.problem_id.nunique()
    per = df.groupby(["model", "strategy", "n", "seed", "problem_id"]).agg(
        k=("completion_tokens", "size"), uniq=("is_dup", lambda x: (~x).sum()),
        tot=("completion_tokens", "sum"), trie=("trie_new_tokens", "sum"), steps=("n_steps", "median"))
    pg = per.groupby(["model", "strategy", "n"])
    s["k"] = pg.k.median()
    s["uniq_k"] = pg.uniq.median()
    s["steps_p50"] = pg.steps.median()
    s["sum_p50"] = pg.tot.median()
    s["sum_p90"] = pg.tot.quantile(.9)
    s["trie_p50"] = pg.trie.median().where(s.index.get_level_values("strategy") != "best_of_n")
    s["prompt_p50"] = g.prompt_tokens.median()
    pd.set_option("display.width", 220)
    print(s.round(0).to_string())


if __name__ == "__main__":
    if "--summary-only" not in sys.argv:
        check_examples()
        build()
    summary()
    accuracy_check()
