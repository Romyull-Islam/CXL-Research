"""KV memory of large-N test-time scaling from the HuggingFace H4 completions (N up to 2048).

Input: data/h4/h4_tokens.parquet from h4_tokens.py (Llama-3.2 1B/3B on MATH-500; best-of-N,
beam search and DVTS). One group is one problem in one run.

Best-of-N: the N samples start together, decode one token per step and share the prompt KV
(prefix cache). At time t (in tokens) the group holds P + t * alive(t) tokens of KV. Reported per
group: peak and mean KV, byte-seconds held by the longest 10% of samples, and the straggler tail
(share of the group's lifetime with at most 10% of samples alive). Beam search and DVTS store only
their final samples, so for them we report the prefix-sharing factor of those samples (sum of
tokens over tokens in their prefix trie), a lower bound on sharing in the live tree.

Run: python analysis/h4_memory.py  (writes results/h4_memory.csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KEYS = ["model", "strategy", "n", "seed", "problem_id"]


def group_stats(o, p):
    o = np.sort(o.astype(np.float64))
    n = len(o)
    life = o[-1]
    alive = n - np.arange(n)                  # samples alive just before the k-th finishes
    peak = p + (o * alive).max()
    bs = p * life + (o ** 2 / 2).sum()          # prompt held once until the last sample ends
    k = max(int(np.ceil(0.1 * n)), 1)
    per = p * o + o ** 2 / 2                    # each sample's own byte-seconds (prompt unshared)
    top10 = per[-k:].sum() / per.sum()
    last10 = o[n - k - 1] if n > k else 0.0
    return dict(n_samples=n, median=np.median(o), max=life, spread=life / max(np.median(o), 1),
                peak_kv=peak, mean_kv=bs / life, peak_to_mean=peak * life / bs, top10_share=top10,
                tail_frac=(life - last10) / life, kv_sum_noshare=(p + o).sum(),
                bytesec_share_saving=1 - bs / per.sum())


def main():
    d = pd.read_parquet(ROOT / "data" / "h4" / "h4_tokens.parquet",
                        columns=KEYS + ["prompt_tokens", "completion_tokens", "is_dup", "trie_new_tokens"])
    rows = []
    for key, g in d.groupby(KEYS, sort=False):
        s = dict(zip(KEYS, key))
        if key[1] == "best_of_n":
            s.update(group_stats(g.completion_tokens.to_numpy(), g.prompt_tokens.iloc[0]))
        else:
            u = g[~g.is_dup]
            s.update(n_samples=len(g), n_unique=len(u), median=u.completion_tokens.median(),
                     max=u.completion_tokens.max(),
                     share_factor=u.completion_tokens.sum() / max(u.trie_new_tokens.sum(), 1))
        rows.append(s)
    st = pd.DataFrame(rows)
    out = ROOT / "results" / "h4_memory.csv"
    out.parent.mkdir(exist_ok=True)
    st.to_csv(out, index=False)
    agg = st.groupby(["model", "strategy", "n"]).median(numeric_only=True).drop(columns=["seed"])
    pd.set_option("display.width", 220)
    print(agg.round(3).to_string())
    # mixed batch: all best-of-N samples of one run pooled, as a server would see them
    b = d[d.strategy == "best_of_n"]
    for (m, n), g in b.groupby(["model", "n"]):
        g = g[g.seed == g.seed.min()]
        o = g.completion_tokens.to_numpy(np.float64)
        per = np.sort(g.prompt_tokens.to_numpy() * o + o ** 2 / 2)[::-1]
        print(f"{m} best_of_n N={n}: across all {len(o)} samples the longest 10% hold "
              f"{per[:len(per) // 10].sum() / per.sum():.2f} of KV byte-seconds")


if __name__ == "__main__":
    main()
