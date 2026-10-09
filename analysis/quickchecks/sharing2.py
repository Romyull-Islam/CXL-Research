import json, sys
sys.path.insert(0, '.')
from sharing import LRU
def run(reqs, n, cap, slack):
    caches=[LRU(cap//n) for _ in range(n)]; load=[0]*n; hits=tot=0; cnt=[0]*n
    for r in reqs:
        h=r["hash_ids"]; tot+=len(h)
        mean=(sum(load)+1)/n
        cand=[j for j in range(n) if load[j]<=slack*mean] or list(range(n))
        i=max(cand,key=lambda j:(caches[j].prefix_len(h),-load[j]))
        load[i]+=r["input_length"]; cnt[i]+=1; hits+=caches[i].access(h)
    return hits/tot, [round(x/sum(load),3) for x in load]
