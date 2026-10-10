"""Can the host tell a hybrid CXL device that request-scoped KV is dead, and what does it save?

Request-scoped KV (lifetime_waf.request_scoped_stream): decode KV of the Bailian thinking trace,
written one 16-token block at a time at 30 tokens/s and freed when the request ends. The device
has an LRU DRAM cache of `dram` blocks (write-back for dirty blocks) in front of a page-mapped FTL
(ftl.FTL, 7% over-provisioning, 64 blocks per erase block).

Host and device options:
  reuse    "fifo": freed slots are reused oldest first; "lifo": most recently freed first.
  discard  False: CXL.mem as today, the device never learns that KV is dead.
           True: at request end the host queues one discard command for the request's slots on
           a single serial command channel (CXL mailbox: one command at a time), each taking
           `lat` seconds. When the command completes, the device drops the slots' DRAM copies
           without writing them and trims them in the FTL, then acknowledges. A slot is reused
           only after its discard is acknowledged, so a late discard can never hit new data.
  reads    after each block written, the request reads `reads` of its earlier blocks chosen at
           random (sparse attention over offloaded KV). A read that misses DRAM fetches the block
           from NAND into DRAM as a clean copy; clean blocks are dropped on eviction.

Logical space = peak live KV plus `slack` (room for slots waiting for an acknowledgement); if no
slot is free the host would stall, which is counted. Outputs per host block written: NAND writes,
write amplification and flash programs (NAND writes x WAF), plus NAND reads, stalls and the most
slots waiting for an acknowledgement. lat = 0 with reads = 0 reproduces the first version.
Run: python analysis/kv_discard.py
"""
import heapq
import os
import random
import sys
from collections import OrderedDict, deque
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ftl import FTL  # noqa: E402
from lifetime_waf import request_scoped_stream  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
B = 64


def peak_live(ev):
    live = peak = 0
    for _, kind, _, k in ev:
        live += 1 if kind == 0 else -k
        peak = max(peak, live)
    return peak


def run(args):
    dram_frac, reuse, discard, lat, reads, op = args[:6]
    slack = args[6] if len(args) > 6 else 0.01
    rng = random.Random(1)
    ev = request_scoped_stream()
    peak = peak_live(ev)
    logical = peak + int(slack * peak)
    dram_n = int(dram_frac * peak)
    f = FTL(logical, op, B)
    free = deque(range(logical))
    owned = {}
    cache = OrderedDict()                   # slot -> dirty
    pending = []                            # (completion time, seq, slots)
    chan_free_at = 0.0
    seq = host = nand = nand_reads = stalls = waiting = max_waiting = 0
    warm, h0 = 2 * peak, None

    def evict_if_full():
        nonlocal nand
        if len(cache) > dram_n:
            old, dirty = cache.popitem(last=False)
            if dirty:
                f.write(old)
                nand += 1

    def put(slot, dirty):
        nonlocal nand
        if dram_n == 0:
            if dirty:
                f.write(slot)
                nand += 1
            return
        if slot in cache:
            cache.move_to_end(slot)
            dirty = dirty or cache[slot]
        cache[slot] = dirty
        evict_if_full()

    def complete(upto):
        nonlocal waiting
        while pending and pending[0][0] <= upto:
            _, _, slots = heapq.heappop(pending)
            for s in slots:
                cache.pop(s, None)
                f.trim(s)
                free.append(s)
            waiting -= len(slots)

    for t, kind, rid, _ in ev:
        complete(t)
        if kind == 0:
            if not free:
                stalls += 1
                complete(pending[0][0])
            slot = free.pop() if reuse == "lifo" else free.popleft()
            mine = owned.setdefault(rid, [])
            mine.append(slot)
            host += 1
            put(slot, True)
            for _ in range(reads if len(mine) > 1 else 0):
                s = mine[rng.randrange(len(mine) - 1)]
                if dram_n and s in cache:
                    cache.move_to_end(s)
                else:
                    nand_reads += 1
                    if dram_n:
                        put(s, False)
            if host == warm:
                h0, n0, r0, f0 = host, nand, nand_reads, f.host + f.gc
        else:
            slots = owned.pop(rid, [])
            if not discard:
                free.extend(slots)
            elif lat == 0:
                for s in slots:
                    cache.pop(s, None)
                    f.trim(s)
                free.extend(slots)
            else:
                chan_free_at = max(chan_free_at, t) + lat
                seq += 1
                heapq.heappush(pending, (chan_free_at, seq, slots))
                waiting += len(slots)
                max_waiting = max(max_waiting, waiting)
    h, n, prog = host - h0, nand - n0, f.host + f.gc - f0
    return dict(dram_frac=dram_frac, reuse=reuse, discard=discard, lat_s=lat, reads=reads, op=op,
                peak_blocks=peak, host_writes=h, nand_per_host=n / h,
                waf=prog / n if n else float("nan"), programs_per_host=prog / h,
                nand_reads_per_host=(nand_reads - r0) / h, stalls=stalls,
                max_waiting_frac=max_waiting / peak)


def main():
    cases = []
    for d in (0.1, 0.25):
        for r in (0, 1, 4):
            cases.append((d, "fifo", False, 0.0, r, 0.07))
            for lat in (0.0, 1e-3, 0.1, 1.0, 10.0):
                cases.append((d, "fifo", True, lat, r, 0.07))
    with Pool(int(os.environ.get("WORKERS", 4))) as p:
        rows = p.map(run, cases)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "kv_discard.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
