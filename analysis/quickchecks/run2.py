import json, sys
from sharing2 import run
for p in sys.argv[1:]:
    reqs=[json.loads(l) for l in open(p)]
    for slack in (1.05, 1.5):
        for n in (4, 8):
            hr, ld = run(reqs, n, 20000, slack)
            print(p.split('/')[-1], f"slack={slack} N={n} hit={hr:.3f} load_share={ld}")
