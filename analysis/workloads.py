"""Request traces for kvsim from public data.

bailian: Qwen-Bailian thinking trace (alibaba-edu/qwen-bailian-usagetraces-anon), real arrival
times and token counts of a production reasoning service, two hours. Arrival times are divided by
`speed` to set the load on one instance.

matharena: per-sample input and output tokens of reasoning models on AIME 2025 (MathArena). Each
request is one problem solved with best-of-n parallel samples: the n samples arrive together.
Samples are drawn from the 4 real samples of a (model, problem) pair; with n > 4 they are reused,
so within-problem spread is limited to what 4 samples show.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from kvsim import Req

DATA = Path(__file__).resolve().parent.parent / "data"


def _clip(prompt, out, max_len):
    prompt = min(int(prompt), max_len - 1)
    return max(prompt, 1), max(min(int(out), max_len - prompt), 1)


def bailian(speed=1.0, max_len=131072, name="qwen_thinking_blksz_16", t_max=None, t_min=0.0,
            offline=False, shard=0, n_shards=1):
    """n_shards > 1 keeps every n_shards-th request (round-robin load balancing)."""
    d = pd.read_json(DATA / "bailian" / f"{name}.jsonl", lines=True)
    d = d.drop(columns="hash_ids").sort_values("timestamp")
    d = d[d.timestamp >= t_min]
    if t_max is not None:
        d = d[d.timestamp < t_max]
    d = d.iloc[shard::n_shards]
    out = []
    for k, row in enumerate(d.itertuples(index=False)):
        p, o = _clip(row.input_length, row.output_length, max_len)
        out.append(Req(k, 0.0 if offline else (row.timestamp - t_min) / speed, p, o))
    return out


def matharena_lengths(min_median=4000):
    """(model_config, problem) -> arrays of input and output tokens, open-weight reasoning models."""
    df = pd.read_parquet(DATA / "tts" / "matharena_aime25.parquet",
                         columns=["model_config", "problem_idx", "input_tokens", "output_tokens"])
    df = df[(df.output_tokens > 0) & (df.input_tokens > 0)]
    med = df.groupby("model_config").output_tokens.median()
    df = df[df.model_config.isin(med[med >= min_median].index)]
    return df


def matharena(model_config, n_requests, n_samples, rate, seed=0, max_len=131072, offline=False):
    """n_requests problems arriving as a Poisson process of `rate` problems per second."""
    df = matharena_lengths(0)
    df = df[df.model_config == model_config]
    groups = [g for _, g in df.groupby("problem_idx")]
    rng = np.random.default_rng(seed)
    t, out, k = 0.0, [], 0
    for q in range(n_requests):
        t = 0.0 if offline else t + rng.exponential(1 / rate)
        g = groups[rng.integers(len(groups))]
        pick = rng.choice(len(g), size=n_samples, replace=n_samples > len(g))
        for j in pick:
            p, o = _clip(g.input_tokens.iloc[j], g.output_tokens.iloc[j], max_len)
            out.append(Req(k, t, p, o, group=q))
            k += 1
    return out
