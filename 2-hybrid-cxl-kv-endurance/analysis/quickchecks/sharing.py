"""Shared pool vs per-instance caches with routing, Mooncake traces. Prefix-hit = longest cached prefix of hash_ids."""
import json, sys, random
from collections import OrderedDict

class LRU:
    def __init__(s, cap): s.c = OrderedDict(); s.cap = cap
    def prefix_len(s, h):
        k = 0
        for x in h:
            if x in s.c: k += 1
            else: break
        return k
    def access(s, h):
        k = s.prefix_len(h)
        for x in h:
            if x in s.c: s.c.move_to_end(x)
            else:
                s.c[x] = 1
                if len(s.c) > s.cap: s.c.popitem(last=False)
        return k

def run(reqs, n_inst, total_cap, policy, slack=1.5):
    caches = [LRU(total_cap // n_inst) for _ in range(n_inst)]
    load = [0]*n_inst; hits = tot = 0; rr = 0
    for r in reqs:
        h = r["hash_ids"]; tot += len(h)
        if policy == "rr":
            i = rr % n_inst; rr += 1
        else:  # affinity with load cap: best prefix among instances whose load <= slack*mean
            mean = (sum(load)+1)/n_inst
            cand = [j for j in range(n_inst) if load[j] <= slack*mean] or list(range(n_inst))
            i = max(cand, key=lambda j: (caches[j].prefix_len(h), -load[j]))
        load[i] += r["input_length"]
        hits += caches[i].access(h)
    return hits/tot

if __name__ == "__main__":
  for p in sys.argv[1:]:
    reqs = [json.loads(l) for l in open(p)]
    for cap in (20_000, 100_000):
        shared = run(reqs, 1, cap, "rr")
        out = [f"shared={shared:.3f}"]
        for n in (4, 8):
            out.append(f"N{n}: rr={run(reqs, n, cap, 'rr'):.3f} aff={run(reqs, n, cap, 'aff'):.3f}")
        print(f"{p.split('/')[-1]} cap={cap} blocks: " + "  ".join(out))
