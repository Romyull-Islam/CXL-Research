"""Reproduce the setup of Mooncake FAST'25 section 5.3.2: 10 prefill nodes, 3M tokens of cache each
(512-token trace blocks), local caches with a cache-aware scheduler versus one global pool.
Mooncake reports up to a 136% higher hit rate for the global pool. The local-cache result depends
on how strictly the scheduler balances load, so the allowed imbalance (slack) is swept.

Run: python analysis/mooncake_repro.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nand_stream import load  # noqa: E402
from pool_sharing import simulate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N, CAP = 10, 3_000_000 // 512

for name in ("conversation", "toolagent"):
    seqs = [s.tolist() for s in load(ROOT / "data" / "mooncake" / f"{name}_trace.jsonl")["seqs"]]
    pooled = simulate(seqs, N, CAP, "pooled")["hit_share"]
    print(f"{name}: pooled hit share {pooled:.3f}")
    for scheme, slack in [("random", 0)] + [("prefix-aware", s) for s in (0.0, 0.05, 0.25, 1.0, 1e9)]:
        r = simulate(seqs, N, CAP, scheme, slack=slack)
        print(f"  {scheme:12s} slack {slack:>8g}: hit {r['hit_share']:.3f}  pool advantage "
              f"{(pooled / r['hit_share'] - 1) * 100:+5.0f}%  imbalance {r['imbalance']:.2f}")
