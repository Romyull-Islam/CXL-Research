"""Write amplification depends on how KV dies: LRU prefix cache versus request-scoped KV.

Prefix cache: the NAND write stream logged by nand_stream.replay, where every write overwrites the
slot of the block LRU just evicted, so KV dies roughly in the order it was written.

Request-scoped KV: decode KV of reasoning requests (Qwen-Bailian thinking trace, real arrivals and
output lengths) is written to flash one 16-token block at a time while the request decodes at
`rate` tokens/s and is all freed when the request ends. Concurrent requests interleave their
blocks in every erase block and end at different times, which is the case HBFlex (arXiv
2609.18675) reports about 30x GC write amplification for.
Freed blocks are either trimmed at once (an NVMe-style deallocate), or, as on CXL.mem which has no
TRIM, stay valid in the FTL until the host reuses their slot (FIFO free list).

The FTL's logical space is the peak live KV; physical space adds op. Erase blocks hold 64 KV
blocks. Run: python analysis/lifetime_waf.py
"""
import heapq
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ftl import FTL  # noqa: E402
from nand_stream import load, oracle_flags, replay  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
B = 64


def request_scoped_stream(rate=30.0, block=16, speed=1.0):
    d = pd.read_json(ROOT / "data" / "bailian" / "qwen_thinking_blksz_16.jsonl", lines=True)
    d = d.drop(columns="hash_ids").sort_values("timestamp")
    ev = []
    for rid, r in enumerate(d.itertuples(index=False)):
        t0 = r.timestamp / speed
        n = max(int(r.output_length) // block, 1)
        for k in range(n):
            ev.append((t0 + (k + 1) * block / rate, 0, rid, k))          # write block k
        ev.append((t0 + max(r.output_length, n * block) / rate + 1e-6, 1, rid, n))  # free all n after the last write
    ev.sort()
    return ev


def run_scoped(op, trim, rate=30.0):
    ev = request_scoped_stream(rate)
    # assign logical slots with a FIFO free list; logical size = peak live blocks
    live, peak, slots = 0, 0, {}
    for _, kind, rid, k in ev:
        live += 1 if kind == 0 else -k
        peak = max(peak, live)
    f = FTL(peak, op, B)
    free = list(range(peak))
    heapq.heapify(free)
    from collections import deque
    fifo = deque(range(peak))
    owned = {}
    host = 0
    warm = 2 * peak
    h0 = g0 = None
    for _, kind, rid, k in ev:
        if kind == 0:
            lu = fifo.popleft()
            owned.setdefault(rid, []).append(lu)
            f.write(lu)
            host += 1
            if host == warm:
                h0, g0 = f.host, f.gc
        else:
            for lu in owned.pop(rid, []):
                if trim:
                    f.trim(lu)
                fifo.append(lu)
    return dict(policy="request-scoped", trim=trim, op=op, waf=(f.host - h0 + f.gc - g0) / (f.host - h0),
                logical_blocks=peak, host_writes=f.host)


def run_prefix(op, trace="bailian/qwen_thinking_blksz_16.jsonl", admit="all", scale=0.02):
    tr = load(ROOT / "data" / trace)
    orc = oracle_flags(tr["seqs"]) if admit == "oracle" else None
    nand = max(int(scale * tr["n_blocks"]), B * 8)
    st = replay(tr, max(nand // 40, 1), nand, admit, orc, log=True)
    f = FTL(nand, op, B)
    r = f.run(st["writes"], warmup=2 * nand)
    return dict(policy=f"prefix LRU ({admit})", trim=False, op=op, waf=r["waf"], logical_blocks=nand,
                host_writes=len(st["writes"]))


def main():
    rows = []
    for op in (0.07, 0.28):
        for admit in ("all", "oracle"):
            rows.append(run_prefix(op, admit=admit))
            print(rows[-1], flush=True)
        for trim in (True, False):
            rows.append(run_scoped(op, trim))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "lifetime_waf.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
