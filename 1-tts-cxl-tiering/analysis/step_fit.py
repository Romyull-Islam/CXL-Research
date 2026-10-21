"""Fit vLLM step times to the tokens each step runs (logs from reuse_validate.py with STEPLOG).

Step time (to the next scheduling call) = a + b * prefill tokens + c * decode tokens, by least
squares over steps with any work, per run. b is the in-run cost of a recomputed (prefilled) token, to
compare with the standalone microbenchmark's c_r.
  python analysis/step_fit.py results/steps/*.json
"""
import json
import sys

import numpy as np


def fit(path):
    d = json.loads(open(path).read())
    s = np.array(d["steps"], dtype=float)
    dt = np.diff(s[:, 0])
    pre, dec = s[:-1, 1], s[:-1, 2]
    keep = (pre + dec) > 0
    X = np.column_stack([np.ones(keep.sum()), pre[keep], dec[keep]])
    coef, *_ = np.linalg.lstsq(X, dt[keep], rcond=None)
    pred = X @ coef
    r2 = 1 - ((dt[keep] - pred) ** 2).sum() / ((dt[keep] - dt[keep].mean()) ** 2).sum()
    mixed = keep & (pre > 0) & (dec > 0)
    return dict(run=path.split("/")[-1], steps=int(keep.sum()), mixed_steps=int(mixed.sum()),
                prefill_tokens=int(pre.sum()), a_ms=coef[0] * 1e3, b_us_per_prefill_token=coef[1] * 1e6,
                c_ms_per_decode_req=coef[2] * 1e3, r2=r2)


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in fit(f).items()})
