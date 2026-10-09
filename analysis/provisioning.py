"""Which flash can hold a node's KV cache for five years? Cheapest class that fits, per workload.

Input: results/endurance_sweep_2000GB.csv and results/endurance_sweep_667GB.csv. Required NAND
writes per day = host write rate x write amplification measured by the FTL (7% over-provisioning
for read-intensive drives, 28% for mixed-use and SLC drives). When the FTL could not reach steady
state the write amplification is taken as 1 (marked).

Budgets (5-year ratings, random writes; datasheets collected 2026-10-09):
- TLC-1: 2 TB at 1 DWPD (Samsung PM9A3, Micron 7450 PRO, Kioxia CM7-R)           2 TB/day
- TLC-3: 2 TB at 3 DWPD (Micron 7450 MAX, Kioxia CM7-V)                          6 TB/day
- pSLC: the same NAND in SLC mode, 667 GB at 35 DWPD (Micron XTR, TLC in SLC mode) 23 TB/day
- SLC:  2 TB of native SLC at 50 DWPD (Solidigm D7-P5810)                       100 TB/day
Sequential ratings are 1.7 to 4.4x higher for TLC (Micron 7450 spec, 128 KB writes) and KV is
written in large blocks; `seq` applies the low end of that factor to TLC.

Bandwidth wall: a measured CMM-H sustains about 4.5 to 5 GB/s once the working set exceeds its
DRAM cache (arXiv 2503.22017; HotStorage'25 slides), ITME's miss path 9 GB/s (arXiv 2606.12556,
Table 3). `over_bw` marks streams whose NAND writes (with write amplification) exceed 4.5 GB/s.

Run: python analysis/provisioning.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLASSES = [  # name, sweep capacity, budget TB/day, over-provisioning for WAF
    ("TLC-1", 2000, 2.0, 7),
    ("TLC-3", 2000, 6.0, 28),
    ("pSLC", 667, 0.667 * 35, 28),
    ("SLC", 2000, 100.0, 28),
]
SEQ_TLC = 1.7
NAND_BW = 4.5      # GB/s


def required(row, op):
    waf = row[f"waf_op{op}"]
    return row["host_GBs"] * 86400 / 1e3 * (1.0 if np.isnan(waf) else waf)


def table(seq=False):
    sweeps = {cap: pd.read_csv(ROOT / "results" / f"endurance_sweep_{cap}GB.csv") for cap in (2000, 667)}
    key = ["trace", "model", "precision", "admit", "duty"]
    base = sweeps[2000].set_index(key)
    rows = []
    for k, r in base.iterrows():
        out = dict(zip(key, k))
        fit = None
        for name, cap, budget, op in CLASSES:
            s = sweeps[cap].set_index(key)
            if k not in s.index:
                continue
            need = required(s.loc[k], op)
            if seq and name.startswith("TLC"):
                budget *= SEQ_TLC
            out[f"{name}_TBday"] = need
            if fit is None and need <= budget:
                fit = name
        out["cheapest"] = fit or "none"
        waf7 = r["waf_op7"] if not np.isnan(r["waf_op7"]) else 1.0
        out["nand_GBs"] = r["host_GBs"] * waf7
        out["over_bw"] = out["nand_GBs"] > NAND_BW
        out["waf_estimated"] = bool(np.isnan(r["waf_op7"]))
        rows.append(out)
    return pd.DataFrame(rows)


def main():
    for seq in (False, True):
        t = table(seq)
        t.to_csv(ROOT / "results" / f"provisioning{'_seq' if seq else ''}.csv", index=False)
        print("\n== budgets from", "sequential (TLC x1.7)" if seq else "random", "ratings")
        for duty in (1.0, 0.3):
            x = t[(t.duty == duty) & (t.admit.isin(["2nd hit", "oracle"]))]
            p = x.pivot_table(index=["trace", "model"], columns=["admit", "precision"], values="cheapest",
                              aggfunc="first")
            print(f"-- prefill duty {duty:.0%}")
            print(p.to_string())
        b = t[t.precision == "BF16"].pivot_table(index=["trace", "model"], columns=["admit", "duty"],
                                                 values="nand_GBs").round(2)
        if not seq:
            print("-- NAND write GB/s with write amplification, BF16 (device NAND path about 4.5 GB/s)")
            print(b.to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    main()
