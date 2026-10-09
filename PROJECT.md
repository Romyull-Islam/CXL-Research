# Endurance limits of LLM KV caching on hybrid DRAM-over-NAND CXL memory

Status (2026-10-09): core analysis done on two providers' production traces. Not novel for a top
conference ("HBF Sucks?", arXiv 2608.11668, reports the flash write excess on the same Alibaba
traces for on-package flash). Viable as a rigorous top-journal paper (IEEE TC, ACM TACO, ACM TOS)
that extends that result to CXL devices with the pieces it lacks.

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
  flash); Son et al. and Kyung et al. (IEEE CAL'26, HBF KV endurance); Lachesis (2610.08378) and
  HBFlex (2609.18675), lifetime-aware placement.
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

Full tables in findings/04-endurance-wall.md (both providers, three models, three policies), with
the novelty verdict at its end. Supporting findings: 01 (Mooncake reuse times and capacity curves),
02 (Alibaba MaaS: reserved KV pools are 1 to 2% used outside decode instances), 03 (real CXL ASIC
bandwidth by read/write mix, for calibration).

## Remaining work for a journal version

1. Sweep TLC, pseudo-SLC and SLC and BF16, FP8 and INT4 KV; state budgets in TB/day.
2. A page-mapped FTL with garbage collection to measure write amplification.
3. Quote HyMCache, ITME and LM-CXD exactly so the comparison is fair.
4. Redo the shared-pool check at Mooncake's scale (analysis/quickchecks is too small and conflicts
   with Mooncake FAST'25).
5. Calibrate against Cylon (FAST'26) or published CMM-H measurements; add the Bailian thinking and
   coder traces.

## Data and how to run

Mooncake FAST'25 traces, Qwen-Bailian traces, ByteDance EuroSys'24 CXL MLC logs, Alibaba MaaS
app metrics. `./fetch_data.sh` downloads into data/ (not in git). Commands are listed in README.md.
