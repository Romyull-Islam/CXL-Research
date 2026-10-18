"""Parked-table size: 512 waiters on a switched device with streaming tenants, tables of 4,096,
256 and 64 entries. Run: python model/run_table.py  (writes model/results_table.csv)"""
import pandas as pd

from experiments import OUT, calibrated, waiters_vs_tenant

rows = []
for table in (4096, 256, 64):
    p = calibrated(600.0, hosts=20, table=table)
    rows.append(dict(table=table, **waiters_vs_tenant(p, 512, "parked")))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "results_table.csv", index=False)
