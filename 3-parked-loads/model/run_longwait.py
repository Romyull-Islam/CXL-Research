"""Waits longer than the parked-read bound (25 us): parked reads time out, answer empty and re-park.
Same setup as run_contention.py (switched device, 4 streaming tenant hosts) with 256 waiters and
mean waits of 20 to 500 us. Run: python model/run_longwait.py  (writes model/results_longwait.csv)"""
import itertools
from multiprocessing import Pool

import pandas as pd

from experiments import OUT, calibrated, waiters_vs_tenant


def one(args):
    mean_wait, pr = args
    horizon = max(200_000.0, 10 * mean_wait)
    r = waiters_vs_tenant(calibrated(600.0, hosts=20), 256, pr, horizon=horizon, mean_wait=mean_wait)
    r.update(mean_wait_us=mean_wait / 1000, protocol=pr)
    return r


if __name__ == "__main__":
    cases = list(itertools.product((20_000.0, 50_000.0, 100_000.0, 500_000.0),
                                   ("poll_flag", "poll_lines", "parked")))
    with Pool(4) as p:
        rows = p.map(one, cases)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results_longwait.csv", index=False)
    print(df[["mean_wait_us", "protocol", "tenant_GBs", "handoff_ns", "handoff_p99_ns", "timeouts",
              "messages"]].to_string(index=False))
