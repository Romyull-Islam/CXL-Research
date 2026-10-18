import json, sys
from sharing2 import run
from sharing import run as run1
p=sys.argv[1]; reqs=[json.loads(l) for l in open(p)]
print("shared", round(run1(reqs,1,20000,"rr"),3))
for n in (4,8):
    hr, ld = run(reqs, n, 20000, 1.05); print(f"N={n} aff(slack1.05) hit={hr:.3f} maxshare={max(ld)}")
