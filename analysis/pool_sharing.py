"""Shared KV pool versus per-host caches, on the Mooncake and Qwen-Bailian traces.

N prefill hosts. Per-host schemes give each host an LRU cache of C blocks:
- random: requests go to hosts uniformly at random;
- affinity: requests go to hash(first block) mod N, so a conversation's prefix stays on one host;
- prefix-aware: requests go to the host with the longest cached prefix, unless that host's
  assigned compute (miss tokens) exceeds the mean by more than `slack`, in which case the least
  loaded host is used (a simple form of Mooncake's cache-aware, load-balanced scheduling).
The pooled scheme has one LRU cache of N x C blocks that every host reads.

Reported: prefix-hit share of block references, cache insertions (blocks written to the cache
tier, which become NAND writes on a hybrid device) and load imbalance (max over mean of the miss
tokens per host).

Run: python analysis/pool_sharing.py  (writes results/pool_sharing.csv)
"""
import itertools
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from nand_stream import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRACES = {
    "mooncake_conversation": "mooncake/conversation_trace.jsonl",
    "mooncake_toolagent": "mooncake/toolagent_trace.jsonl",
    "bailian_A_chat": "bailian/qwen_traceA_blksz_16.jsonl",
    "bailian_B_api": "bailian/qwen_traceB_blksz_16.jsonl",
}


class LRU:
    def __init__(self, cap):
        self.cap, self.d, self.inserts = cap, OrderedDict(), 0

    def prefix(self, seq):
        n = 0
        for b in seq:
            if b in self.d:
                n += 1
            else:
                break
        return n

    def use(self, seq, hit):
        for b in seq[:hit]:
            self.d.move_to_end(b)
        for b in seq[hit:]:
            if b in self.d:
                self.d.move_to_end(b)
            else:
                self.d[b] = None
                self.inserts += 1
                if len(self.d) > self.cap:
                    self.d.popitem(last=False)


def simulate(seqs, n_hosts, cap, scheme, slack=0.25, seed=0):
    rng = np.random.default_rng(seed)
    refs = sum(len(s) for s in seqs)
    if scheme == "pooled":
        c = LRU(n_hosts * cap)
        hits = 0
        for s in seqs:
            h = c.prefix(s)
            hits += h
            c.use(s, h)
        return dict(hit_share=hits / refs, inserts=c.inserts, imbalance=1.0)
    caches = [LRU(cap) for _ in range(n_hosts)]
    load = np.zeros(n_hosts)
    hits = 0
    for s in seqs:
        if scheme == "random":
            k = int(rng.integers(n_hosts))
        elif scheme == "affinity":
            k = int(s[0]) % n_hosts if len(s) else 0
        else:
            pre = np.array([c.prefix(s) for c in caches])
            k = int(pre.argmax())
            if load[k] > (1 + slack) * max(load.mean(), 1):
                k = int(load.argmin())
        h = caches[k].prefix(s)
        hits += h
        caches[k].use(s, h)
        load[k] += len(s) - h
    return dict(hit_share=hits / refs, inserts=sum(c.inserts for c in caches),
                imbalance=load.max() / max(load.mean(), 1))


def one(cfg):
    name, n, frac, scheme = cfg
    tr = load(ROOT / "data" / TRACES[name])
    seqs = [s.tolist() for s in tr["seqs"]]
    cap = max(int(frac * tr["n_blocks"] / n), 1)
    r = simulate(seqs, n, cap, scheme)
    r.update(trace=name, hosts=n, cap_frac=frac, cap_blocks_per_host=cap, scheme=scheme)
    print(r, flush=True)
    return r


def main():
    cfgs = list(itertools.product(TRACES, (4, 8, 16), (0.02, 0.05, 0.1, 0.2, 0.5),
                                  ("pooled", "random", "affinity", "prefix-aware")))
    with ProcessPoolExecutor(4) as ex:
        rows = list(ex.map(one, cfgs, chunksize=4))
    out = ROOT / "results" / "pool_sharing.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
