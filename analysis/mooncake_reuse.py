"""Reuse time and LRU stack distance of prompt blocks in the Mooncake traces."""
import json
import sys

import numpy as np

BLOCK_TOKENS = 512
KV_BYTES_PER_TOKEN = {               # K and V, all layers, 16-bit
    "Qwen2.5-7B (GQA)": 2 * 28 * 4 * 128 * 2,
    "DeepSeek-V3 (MLA)": (512 + 64) * 61 * 2,
    "Llama-3.1-70B (GQA)": 2 * 80 * 8 * 128 * 2,
}


class Fenwick:
    def __init__(self, n):
        self.t = np.zeros(n + 1, dtype=np.int64)

    def add(self, i, v):
        i += 1
        while i < len(self.t):
            self.t[i] += v
            i += i & -i

    def prefix(self, i):
        i += 1
        s = 0
        while i > 0:
            s += self.t[i]
            i -= i & -i
        return s


def analyze(path):
    reqs = [json.loads(line) for line in open(path)]
    refs = [(r["timestamp"] / 1000.0, h) for r in reqs for h in r["hash_ids"]]
    n = len(refs)
    last_pos, last_time = {}, {}
    fw = Fenwick(n)
    reuse_s, stack = [], []
    for pos, (t, h) in enumerate(refs):
        if h in last_pos:
            p = last_pos[h]
            stack.append(fw.prefix(pos - 1) - fw.prefix(p))   # distinct blocks touched since last use
            reuse_s.append(t - last_time[h])
            fw.add(p, -1)
        fw.add(pos, 1)
        last_pos[h], last_time[h] = pos, t
    reuse_s, stack = np.array(reuse_s), np.array(stack)
    print(f"== {path.split('/')[-1]}: {n} block refs, {len(reuse_s)} reuses")
    pct = [10, 25, 50, 75, 90, 99]
    print("   reuse time (s)  " + "  ".join(f"p{p}={np.percentile(reuse_s, p):.1f}" for p in pct))
    bands = [(0, 1), (1, 10), (10, 60), (60, 600), (600, 3600)]
    print("   reuse time bands " + "  ".join(f"[{a},{b})s:{np.mean((reuse_s >= a) & (reuse_s < b)):.2f}" for a, b in bands))
    max_hits = len(stack)
    print("   LRU capacity needed (blocks) for a share of all achievable hits: " +
          "  ".join(f"{q}%->{int(np.percentile(stack, q)) + 1}" for q in (50, 80, 90, 95, 99)))
    for model, bpt in KV_BYTES_PER_TOKEN.items():
        gb = lambda blocks: blocks * BLOCK_TOKENS * bpt / 1e9
        caps = "  ".join(f"{q}%->{gb(np.percentile(stack, q) + 1):,.0f} GB" for q in (50, 80, 90, 95))
        print(f"   {model:22} {caps}")
    return reuse_s, stack, n


for path in sys.argv[1:]:
    analyze(path)
