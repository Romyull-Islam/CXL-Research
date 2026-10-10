"""When does loading KV from the device beat recomputing it?

Recomputing n tokens of context on one 8-GPU node takes
  t_rc(n) = (2 * P_active * n + L * w * n^2) / (G * F * u)
with linear FLOPs 2 * P_active per token and causal attention over n^2 / 2 query-key pairs at
2 * w FLOPs per pair and layer (w = heads x (qk dim + v dim)), on G GPUs of F dense FLOP/s at
utilization u. Loading takes n * kv_bytes / B over a device path of B bytes/s. Loading wins when
B exceeds the break-even bandwidth B* = n * kv_bytes / t_rc(n), which falls as the context grows
because attention makes recompute superlinear.

Device paths (measured, 48 GB CMM-H, HotStorage'25): DRAM-cache hits about 27 GB/s, NAND misses about 5 GB/s
(arXiv 2503.22017, HotStorage'25), ITME miss path 9 GB/s (arXiv 2606.12556). GPUs: H100 989 and
B200 2250 dense BF16 TFLOP/s; u = 0.5 as in finding 04. The general trade-off is formalized in
"To Restore or To Recompute" (HotStorage'26); here it is applied to the device numbers above.

Run: python analysis/load_vs_recompute.py  (writes results/load_vs_recompute.csv)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# name: KV elements per token, active params, layers, w = heads x (qk dim + v dim)
MODELS = {
    "Llama-3.1-70B": (2 * 80 * 8 * 128, 70.6e9, 80, 64 * 128 * 2),
    "DeepSeek-V3": ((512 + 64) * 61, 37.6e9, 61, 128 * (192 + 128)),
    "Qwen2.5-7B": (2 * 28 * 4 * 128, 7.6e9, 28, 28 * 128 * 2),
}
GPUS = {"8x H100": 8 * 989e12, "8x B200": 8 * 2250e12}
UTIL = 0.5
PREC = {"BF16": 2.0, "FP8": 1.0, "INT4": 0.5}
CTX = (2048, 8192, 32768)


def breakeven(model, gpu, prec, n):
    elems, p, layers, width = MODELS[model]
    flops = 2 * p * n + layers * width * n * n
    t = flops / (GPUS[gpu] * UTIL)
    return n * elems * PREC[prec] / t


def main():
    rows = [dict(model=m, gpu=g, precision=pr, context=n, breakeven_GBs=breakeven(m, g, pr, n) / 1e9)
            for m in MODELS for g in GPUS for pr in PREC for n in CTX]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "load_vs_recompute.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.pivot_table(index=["model", "gpu", "precision"], columns="context", values="breakeven_GBs").round(1))
    print("\nLoading from NAND (5 GB/s, CMM-H) wins where the break-even is below 5 GB/s;")
    print("from the DRAM cache (27 GB/s) wherever the break-even is below 27 GB/s.")


if __name__ == "__main__":
    main()
