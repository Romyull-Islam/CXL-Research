"""Basic shape of the Mooncake FAST'25 traces: sizes, timing and prefix-block reuse."""
import json
import sys
from collections import Counter

import numpy as np

BLOCK_TOKENS = 512


def load(path):
    return [json.loads(line) for line in open(path)]


def summarize(name, reqs):
    ts = np.array([r["timestamp"] for r in reqs], dtype=float)
    inp = np.array([r["input_length"] for r in reqs])
    out = np.array([r["output_length"] for r in reqs])
    refs = Counter(h for r in reqs for h in r["hash_ids"])
    total = sum(refs.values())
    print(f"== {name}: {len(reqs)} requests, timestamps {ts.min():.0f}..{ts.max():.0f} "
          f"(sorted: {bool(np.all(np.diff(ts) >= 0))})")
    print(f"   input tokens  p50 {np.median(inp):.0f}  p90 {np.percentile(inp, 90):.0f}  max {inp.max()}")
    print(f"   output tokens p50 {np.median(out):.0f}  p90 {np.percentile(out, 90):.0f}  max {out.max()}")
    print(f"   block refs {total}, unique blocks {len(refs)}, "
          f"reuse fraction (infinite cache) {1 - len(refs) / total:.3f}")
    once = sum(1 for c in refs.values() if c == 1)
    print(f"   blocks referenced once {once / len(refs):.3f}; top-10 block ref counts {[c for _, c in refs.most_common(10)]}")
    roots = Counter(r["hash_ids"][0] for r in reqs if r["hash_ids"])
    print(f"   distinct first blocks {len(roots)}; most common first block share "
          f"{roots.most_common(1)[0][1] / len(reqs):.3f}")


for path in sys.argv[1:]:
    summarize(path.split("/")[-1], load(path))
