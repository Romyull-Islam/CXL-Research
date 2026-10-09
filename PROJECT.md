# Endurance limits of LLM KV caching on hybrid DRAM-over-NAND CXL memory

Status (2026-10-09): journal analysis done (findings/05). Novelty re-checked the same day: narrow
but present; no paper models a hybrid DRAM-over-NAND CXL device with per-node scaling, measured
write amplification and flash x precision provisioning, but HBF papers now add admission and
lifetime-aware placement (2609.07175, 2609.39131, HBFlex 2609.18675, Lachesis 2610.08378) and two
use the same Bailian traces. Target: ACM TOS (characterization and provisioning); IEEE TC or TACO
are weaker without a mechanism. Not a top-conference paper.

## Idea

Replay real production LLM traces through a model of a hybrid CXL device (a DRAM cache in front of
NAND flash) to find what such a device can actually sustain as a KV cache. The binding limit turns
out to be flash endurance, not latency: even perfect admission exceeds TLC flash's write budget for
interactive traffic, while API traffic fits. Turn this into provisioning rules for CMM-H-class
devices.

## Problem and gap

Hybrid CXL devices (Samsung CMM-H class: tens of GB of DRAM cache over TB of NAND) are proposed as
cheap KV-cache capacity for LLM serving (HyMCache arXiv 2607.18141, ITME 2606.12556, LM-CXD
2609.26828). These designs assume high prefix-hit rates, read-mostly KV, a device DRAM that holds
the working set, and that loading KV beats recomputing it. Production traces contradict several of
these assumptions, and no study quantifies what a CXL hybrid device can sustain under real traffic.

## Goal

A trace-driven characterization and provisioning study for hybrid CXL KV tiers:
1. write excess and endurance per node under admit-all, second-hit and oracle admission;
2. an oracle-admission bound: the lowest NAND write rate any admission policy can reach;
3. the split between interactive traffic (chat, agents) and API traffic;
4. when loading beats recomputing, as a function of KV bytes per FLOP (GQA versus MLA models);
5. provisioning rules for CMM-H-class devices across TLC, pseudo-SLC and SLC and across BF16, FP8
   and INT4 KV.

## Motivation

- Hybrid CXL devices are the cheapest way to add terabytes next to a GPU server, and several 2026
  preprints build KV caches on them without measuring wear.
- Production evidence (Mooncake and Qwen-Bailian traces replayed through the device model):
  - admitting all KV writes 2 to 4x more bytes to NAND than are read back;
  - only 21 to 44% of newly computed KV is ever reused in interactive traffic, 4% in API traffic;
  - per node, even oracle admission exceeds TLC endurance by about 6 to 80x for interactive
    traffic, while API traffic falls within reach of TLC for MLA models;
  - device DRAM serves 12 to 46% of interactive reuse but about 90% of API reuse;
  - for a dense GQA 70B model, loading KV from NAND never beats recomputing on 8 x H100, while MLA
    models benefit.
- Device makers need provisioning guidance (capacity, flash type, admission) that no paper gives
  for CXL.

## Related work

- Flash endurance for KV: "HBF Sucks?" (2608.11668, write excess on Bailian traces, on-package
  flash, unit write amplification, TLC only); Son et al. and Kyung et al. (IEEE CAL'26, HBF KV
  endurance); Lachesis (2610.08378) and HBFlex (2609.18675, about 30x GC write amplification for
  interleaved lifetimes), lifetime-aware placement; admission for HBF: 2609.07175 (LRU-K, SLC) and
  2609.39131 (admission headroom).
- Hybrid CXL KV systems measured without wear: HyMCache (64 GB DRAM over 2 TB TLC, skips over 15% of
  insertions), ITME (qualitative wear claims only), LM-CXD (emulated Z-NAND), TempoKV (XCENA MX1P over
  a 15.36 TB PM1753); measured CMM-H: 48 GB DRAM cache, about 27 GB/s when the working set fits and
  5 GB/s when it does not.
- Counter-evidence: SPLASH (2609.23816) and HiFC (NeurIPS'25) report multi-year lifetimes with SLC
  or pseudo-SLC.
- Load versus recompute: "To Restore or To Recompute" (HotStorage'26); py-kvcache (2609.11744);
  Cake; HCache (EuroSys'25).
- Reuse and working sets: KVCache-in-the-Wild (ATC'25); KVSET (2609.27746); Mooncake (FAST'25);
  TokenLake (pool sharing).
- Admission filters: Flashield (NSDI'19); SGLang HiCache selective write-through; write-aware LRU-K.
- Hybrid CXL KV systems: HyMCache, ITME, LM-CXD, TempoKV (2609.35065); CMM-H characterization
  (2503.22017); Cylon (FAST'26).

This work adds what those lack: a CXL device model, the oracle bound, the interactive versus API
split and device provisioning rules.

## Evidence so far

Full tables in findings/05-journal-analysis.md (corrected per-node scaling, measured write
amplification, provisioning across flash type, KV precision, admission and prefill duty, shared
pool versus per-host caches) and findings/04-endurance-wall.md (first pass and novelty verdict).
Headlines:
- admit-all never fits TLC or pseudo-SLC; with second-hit or oracle admission, TLC fits only
  API-like traffic at partial prefill duty or MLA models with INT4 KV; interactive traffic needs
  pseudo-SLC or SLC, and at full prefill duty nearly half of its cases exceed 100 TB/day;
- FP8 halves and INT4 quarters NAND writes; MLA models write 2.5 to 2.8x less than a GQA 70B model;
- write amplification of an LRU KV cache is about 1 (FTL validated against Desnoyers' closed
  forms); request-scoped KV reaches 2.2 with TRIM and 6.9 without it, the CXL.mem case;
- a shared pool beats load-balanced cache-aware routing by only 1 to 18%, random routing by 1.05
  to 4.5x; Mooncake's +136% sits inside this range.
Supporting findings: 01 (Mooncake reuse), 02 (Alibaba MaaS pools), 03 (CXL bandwidth by mix).

## Remaining work for a journal version

Done on 2026-10-09 (findings/05): consistent per-node scaling, a validated FTL with garbage
collection, the TLC / pSLC / SLC x BF16 / FP8 / INT4 sweep with datasheet budgets, write
amplification by KV death order (including the no-TRIM CXL case), the shared-pool check at
Mooncake's scale, the Bailian thinking and coder traces, the exact claims of HyMCache, ITME,
LM-CXD and TempoKV (none measures wear), and a bandwidth check against measured CMM-H rates.
Left:
1. Measure write amplification on a validated SSD emulator (FEMU reports WAF in its log page) or a
   real drive by replaying the logged NAND write streams; not on the author's system disk.
2. Write the paper; cite and separate HBF Sucks?, HBFlex, Lachesis, 2609.07175 and 2609.39131.
3. Re-run the novelty search before submission.

## Data and how to run

Mooncake FAST'25 traces, Qwen-Bailian traces, ByteDance EuroSys'24 CXL MLC logs, Alibaba MaaS
app metrics. `./fetch_data.sh` downloads into data/ (not in git). Commands are listed in README.md.
