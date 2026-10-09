# Hybrid DRAM-over-NAND CXL devices hit an endurance wall as KV caches

Data: Mooncake conversation and toolagent traces (one hour each). Scripts:
analysis/hybrid_device_replay.py (trace replay through device DRAM 16 or 48 GB over 2 TB NAND,
LRU, write-back to NAND on DRAM eviction) and analysis/per_node_endurance.py (per-node rates).
Device numbers: GPU to CXL pool 26 GB/s (Beluga, measured), NAND path 1 or 4.5 GB/s (ITME
prototype miss path; CMM-H measured peak). Recompute: FLOP roofline on H100, 50% utilization.

Admission policies: all (write every block evicted from device DRAM), 2nd hit (write only blocks
referenced at least twice), oracle (write only blocks that will be referenced again).

## Trace-level results (48 GB device DRAM)

| Trace, model | Policy | Share of block refs served | of which from device DRAM | NAND bytes written / read |
|---|---|---|---|---|
| conversation, 70B | all | 23% | 18% | 4.07 |
| conversation, 70B | 2nd hit | 21% | 20% | 0.93 |
| conversation, 70B | oracle | 34% | 12% | 0.54 |
| conversation, 7B | all | 36% | 14% | 2.05 |
| toolagent, 70B | all | 47% | 70% | 3.74 |
| toolagent, 70B | oracle | 55% | 61% | 0.45 |

## Per node (one 2 TB device per 8-GPU node, prefill duty 100% / 30%)

Only 21 to 24% of newly computed KV is ever reused (the oracle's persisted share).

| Model | Policy | NAND write GB/s | Drive writes per day |
|---|---|---|---|
| Llama-3.1-70B | all | 9.3 | 400 / 120 |
| Llama-3.1-70B | oracle | 2.0 to 2.3 | 87 to 98 / 26 to 29 |
| DeepSeek-V3 (MLA) | all | 3.7 | 162 / 48 |
| DeepSeek-V3 (MLA) | oracle | 0.8 to 0.9 | 35 to 39 / 10 to 12 |
| Qwen2.5-7B | all | 14.8 | 641 / 192 |
| Qwen2.5-7B | oracle | 3.2 to 3.6 | 138 to 155 / 41 to 46 |

TLC endurance is about 1 to 3 drive writes per day; pseudo-SLC gives roughly 10x more at a third
of the capacity.

## Load versus recompute (share of requests with a hit where loading is faster)

- DeepSeek-V3 and Qwen2.5-7B: 63 to 92% at a 1 GB/s NAND path, 100% at 4.5 GB/s.
- Llama-3.1-70B: 61 to 89%, identical at 1 and 4.5 GB/s, because only hits served from device
  DRAM win; loading from NAND never beats recomputing on 8 x H100 at 320 KB of KV per token.

## Reading

1. Admitting all KV writes 2 to 4 times more to NAND than is read back, and at node prefill
   rates the write stream (3.7 to 14.8 GB/s) exceeds the device's measured bandwidth.
2. Even oracle admission exceeds TLC endurance by about 3 to 50x per node. The only near-feasible
   corner is an MLA model, partial prefill duty, near-oracle admission and pseudo-SLC.
3. A second-hit filter cuts writes like the oracle but loses about 40% of hits, because the first
   reuse arrives minutes later, after the block has left the 16 to 48 GB device DRAM.
4. Whether NAND-backed KV pays at all depends on KV bytes per FLOP: MLA-class models gain, dense
   GQA 70B-class models do not.

## Caveats

- One hour of one provider's traffic; 512-token blocks; LRU everywhere; no GC or write
  amplification modeled (real WAF only makes the wall higher).
- Prefill rate from a FLOP roofline; attention FLOPs and real utilization ignored.
- One device per node is an assumption; more devices per node divide the drive writes per day.
- Novelty check pending against HBF endurance work (Son et al., HBF Sucks?, HBFlex, Lachesis),
  HiFC (pseudo-SLC KV), KVCache-in-the-Wild and compute-or-load work.

## Second provider: Alibaba Qwen-Bailian (16-token blocks, two hours each)

Script: `per_node_endurance.py 16 data/bailian/qwen_trace{A,B}_blksz_16.jsonl` (48 GB device
DRAM). Drive writes per day per 2 TB device, prefill duty 100% / 30%.

| Trace | Model | Oracle persisted share | All: DWPD | Oracle: DWPD | Hits from device DRAM |
|---|---|---|---|---|---|
| A, to-C chat | Llama-3.1-70B | 0.44 | 382 / 115 | 175 / 53 | 27 to 39% |
| A, to-C chat | DeepSeek-V3 | 0.39 | 154 / 46 | 64 / 19 | 41 to 55% |
| A, to-C chat | Qwen2.5-7B | 0.37 | 611 / 183 | 238 / 72 | 46 to 60% |
| B, to-B API | Llama-3.1-70B | 0.04 | 352 / 105 | 17 / 5 | 68 to 69% |
| B, to-B API | DeepSeek-V3 | 0.04 | 137 / 41 | 7 / 2 | 88 to 89% |
| B, to-B API | Qwen2.5-7B | 0.04 | 542 / 162 | 26 / 8 | 90 to 91% |

Revised reading: the wall is workload dependent.
- Interactive traffic (Bailian chat, Mooncake conversation and tool agents) keeps 21 to 44% of
  computed KV for minutes; even oracle admission is 6 to 80x over TLC endurance.
- API traffic (Bailian to-B) reuses only 4% of KV and mostly within seconds; device DRAM serves
  about 90% of hits and second-hit or oracle admission brings NAND writes to 2 to 26 drive writes
  per day, within reach of TLC for MLA models at partial prefill duty.
- A hybrid device therefore needs to know which regime it is serving; one fixed admission policy
  is either endurance-infeasible (interactive) or wasteful (API).

## Novelty gate (2026-10-09): failed as framed

- "HBF Sucks?" (arXiv 2608.11668, Aug-Sep 2026) already replays the same Qwen-Bailian traces through
  a flash KV tier: 1.14 to 4.90x more writes than reads, 48 to 140 TB/day against a 21.7 TB/day
  TLC budget, and says the result applies to CXL-attached flash.
- Load versus recompute as KV bytes per FLOP is formalized in "To Restore or To Recompute"
  (HotStorage'26) and py-kvcache (arXiv 2609.11744).
- Mooncake FAST'25 (section 5.3) and TokenLake show a global pool beating prefix-routed local
  caches, contradicting our sharing check; that check was too small to stand.
- Counter-evidence on endurance: SPLASH (SLC, 30-minute retention) and HiFC (pseudo-SLC) report
  multi-year lifetimes, so a TLC-based wall can be dismissed as a configuration choice.
- What is left (an oracle-admission bound, the interactive vs API split) is a refinement of
  HBF Sucks?, not a top-venue contribution on its own.
