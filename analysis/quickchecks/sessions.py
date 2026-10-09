"""Quick checks on Mooncake traces: (1) can sessions be reconstructed by prefix-extension of hash_ids,
(2) inter-step gaps of reconstructed chains, (3) write vs read block volume under admit-all LRU."""
import json, sys
from collections import OrderedDict
import numpy as np

def load(p):
    return [json.loads(l) for l in open(p)]

def chains(reqs):
    # map full hash tuple -> latest request idx; child = request whose hash_ids extend a previous request's
    # full block list (excluding last partial block of parent, which may differ)
    by_prefix = {}
    parent = [-1]*len(reqs)
    for i, r in enumerate(reqs):
        h = r["hash_ids"]
        # find longest k such that tuple(h[:k]) is a registered parent key
        best = -1
        for k in range(len(h)-1, 1, -1):
            j = by_prefix.get(tuple(h[:k]))
            if j is not None:
                best = j; break
        parent[i] = best
        # register this request's prefix minus last (possibly partial) block
        if len(h) > 2:
            by_prefix[tuple(h[:-1])] = i
    return parent

def lru_rw(reqs, cap_blocks):
    c = OrderedDict(); hits = misses = 0
    for r in reqs:
        for h in r["hash_ids"]:
            if h in c:
                c.move_to_end(h); hits += 1
            else:
                misses += 1; c[h] = 1
                if len(c) > cap_blocks: c.popitem(last=False)
    return hits, misses

for p in sys.argv[1:]:
    reqs = load(p)
    ts = np.array([r["timestamp"] for r in reqs])/1000.0
    par = chains(reqs)
    linked = [i for i, x in enumerate(par) if x >= 0]
    gaps = np.array([ts[i]-ts[par[i]] for i in linked])
    print(f"== {p.split('/')[-1]}: {len(reqs)} reqs, {len(linked)} ({len(linked)/len(reqs):.2f}) extend a previous request's full prefix")
    if len(gaps):
        print("   parent->child gap s: " + " ".join(f"p{q}={np.percentile(gaps,q):.1f}" for q in (10,25,50,75,90,99)))
    nb = sum(len(r["hash_ids"]) for r in reqs)
    for cap in (10_000, 50_000, 200_000):
        h, m = lru_rw(reqs, cap)
        print(f"   LRU cap {cap} blocks: hit {h/nb:.2f}, miss(=admit write) {m/nb:.2f}, write/read ratio {m/max(h,1):.2f}")
