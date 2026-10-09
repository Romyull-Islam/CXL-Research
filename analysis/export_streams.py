"""Export NAND write streams (logical unit per host write) for a cross-check of ftl.py with MQSim.
Writes data/streams/<name>.npy and data/streams/index.csv with this FTL's WAF (64 units per erase
block) for each stream. Run: python analysis/export_streams.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ftl import FTL  # noqa: E402
from lifetime_waf import request_scoped_stream  # noqa: E402
from nand_stream import load, oracle_flags, replay  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "streams"
B = 64


def lru_stream(trace, admit, nand_frac=0.02):
    tr = load(ROOT / "data" / trace)
    orc = oracle_flags(tr["seqs"]) if admit == "oracle" else None
    nand = max(int(nand_frac * tr["n_blocks"]), B * 8)
    return nand, replay(tr, max(nand // 40, 1), nand, admit, orc, log=True)["writes"]


def scoped_stream():
    ev = request_scoped_stream()
    live = peak = 0
    for _, kind, _, k in ev:
        live += 1 if kind == 0 else -k
        peak = max(peak, live)
    from collections import deque
    fifo, owned, w = deque(range(peak)), {}, []
    for _, kind, rid, k in ev:
        if kind == 0:
            lu = fifo.popleft()
            owned.setdefault(rid, []).append(lu)
            w.append(lu)
        else:
            fifo.extend(owned.pop(rid, []))
    return peak, np.array(w, dtype=np.int64)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = {"lru_chat_oracle": lambda: lru_stream("bailian/qwen_traceA_blksz_16.jsonl", "oracle"),
             "lru_chat_all": lambda: lru_stream("bailian/qwen_traceA_blksz_16.jsonl", "all"),
             "scoped_notrim": scoped_stream}
    for name, fn in cases.items():
        L, w = fn()
        np.save(OUT / f"{name}.npy", w)
        r = dict(name=name, logical_units=L, writes=len(w))
        for op in (0.07, 0.28):
            r[f"waf_op{int(op * 100)}"] = FTL(L, op, B).run(w, warmup=2 * L)["waf"]
        rows.append(r)
        print(r, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "index.csv", index=False)


if __name__ == "__main__":
    main()
