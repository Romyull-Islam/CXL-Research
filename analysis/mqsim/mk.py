"""Make an MQSim trace, SSD config and workload file for one case.

usage: mk.py CASE_DIR STREAM(.npy or uniform:L:N:seed) L OP DELTA_BLOCKS PREFIX_WRITES [GAP_NS]
One logical unit = one 4 KB flash page = 8 sectors. One channel/chip/die/plane.
Blocks per plane = ceil(L*(1+op)/64) (same as ftl.py) + DELTA_BLOCKS.
"""
import math
import os
import sys

import numpy as np

SECT = 8          # sectors per page (4 KB page)
PPB = 64          # pages per block

case, src, L, op, delta, prefix = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
gap = int(sys.argv[7]) if len(sys.argv) > 7 else 1_000_000
os.makedirs(case, exist_ok=True)

if src.startswith("uniform:"):
    _, uL, n, seed = src.split(":")
    s = np.random.default_rng(int(seed)).integers(0, int(uL), size=int(n), dtype=np.int64)
else:
    s = np.load(src, mmap_mode="r")
s = np.asarray(s[:prefix], dtype=np.int64)
assert s.min() >= 0 and s.max() < L

n_eb = max(int(np.ceil(L * (1 + op) / PPB)), L // PPB + 2 + 2)
blocks = n_eb + delta
P = blocks * PPB
r = (P - L) / P - 1e-9
assert int(P * (1 - r)) >= L and r >= 0.05, (r, P, L)

tag = os.path.basename(src).replace(".npy", "").replace(":", "_")
tpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "traces", f"{tag}_{prefix}" + ("" if gap == 1_000_000 else f"_gap{gap}") + ".txt"))
t = (np.arange(len(s), dtype=np.int64) + 1) * gap
if not os.path.exists(tpath):
  with open(tpath, "w") as f:
    step = 1_000_000
    for i in range(0, len(s), step):
        a, b = t[i:i + step], s[i:i + step] * SECT
        f.write("".join(f"{x} 0 {y} {SECT} 0\n" for x, y in zip(a.tolist(), b.tolist())))

tmpl = open(os.path.join(os.path.dirname(__file__), "ssdconfig.tmpl")).read()
open(os.path.join(case, "ssdconfig.xml"), "w").write(
    tmpl.replace("@OP@", f"{r:.9f}").replace("@BLOCKS@", str(blocks)).replace("@PPB@", str(PPB)).replace("@PAGE@", str(SECT * 512)))
wl = open(os.path.join(os.path.dirname(__file__), "workload.tmpl")).read()
open(os.path.join(case, "workload.xml"), "w").write(wl.replace("@TRACE@", tpath))
print(f"{case}: L={L} op={op} n_eb={n_eb} blocks={blocks} P={P} OP_ratio={r:.6f} writes={len(s)} ({len(s)/P:.2f}x physical)")
