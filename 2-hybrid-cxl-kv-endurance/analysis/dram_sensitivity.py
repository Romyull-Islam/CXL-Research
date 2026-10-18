"""Sensitivity of per-node NAND writes to the device's DRAM cache size (16 to 256 GB).
Run: python analysis/dram_sensitivity.py  (writes results/dram_sensitivity.csv; 4 workers)"""
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import endurance_sweep as es  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def one(cfg):
    dram, rest = cfg[0], cfg[1:]
    es.DRAM_GB = dram
    r = es.one(rest)
    r["dram_GB"] = dram
    return r


def main():
    cfgs = [(d, t, m, "BF16", a, 1.0) for d, t, m, a in itertools.product(
        (16, 48, 128, 256), ("bailian_B_api", "bailian_A_chat", "bailian_thinking"),
        ("Llama-3.1-70B", "DeepSeek-V3"), ("2nd hit", "oracle"))]
    cfgs.sort(key=lambda c: c[1])
    with ProcessPoolExecutor(4) as ex:
        rows = list(ex.map(one, cfgs, chunksize=4))
    pd.DataFrame(rows).to_csv(ROOT / "results" / "dram_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
