"""Many waiters sharing one switched CXL device with bandwidth-heavy tenants.
Run: python model/run_contention.py  (writes model/results_contention.csv)"""
import itertools

import pandas as pd

from experiments import OUT, calibrated, waiters_vs_tenant

rows = []
for w, pr in itertools.product((0, 16, 64, 128, 256, 512), ("poll_flag", "poll_lines", "parked")):
    if w == 0 and pr != "parked":
        continue
    rows.append(dict(waiters=w, protocol=pr, **waiters_vs_tenant(calibrated(600.0, hosts=20), w, pr)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "results_contention.csv", index=False)
