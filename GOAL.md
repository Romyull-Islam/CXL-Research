# Endurance limits of LLM KV caching on hybrid DRAM-over-NAND CXL memory

Status (2026-10-09): core analysis done on two providers' production traces (findings/). Not novel
for a top conference ("HBF Sucks?", arXiv 2608.11668, reports the write excess on the same Alibaba
traces for on-package flash). Viable as a rigorous top-journal paper (IEEE TC, ACM TACO, ACM TOS)
that extends that result to CXL devices with the pieces it lacks.

## Problem and gap

Hybrid CXL devices (Samsung CMM-H class: tens of GB of DRAM cache over TB of NAND) are being
proposed as cheap KV-cache capacity for LLM serving (HyMCache arXiv 2607.18141, ITME 2606.12556,
LM-CXD 2609.26828). These designs assume high prefix-hit rates, read-mostly KV, a device DRAM that
holds the working set, and that loading KV always beats recomputing it. Production traces
contradict several of these assumptions, and no study quantifies what a CXL hybrid device can
sustain under real traffic.

## Goal

A trace-driven characterization and provisioning study for hybrid CXL KV tiers:
- write excess and endurance per node under admit-all, second-hit and oracle admission;
- an oracle-admission bound (the lowest NAND write rate any admission policy can reach);
- the split between interactive traffic (chat, agents) and API traffic;
- when loading beats recomputing, as a function of KV bytes per FLOP (GQA versus MLA);
- provisioning rules for CMM-H-class devices across TLC, pseudo-SLC and SLC and BF16, FP8 and INT4
  KV.

## Motivation and evidence so far

- Admitting all KV writes 2 to 4x more bytes to NAND than are read back.
- Only 21 to 44% of newly computed KV is ever reused in interactive traffic; 4% in API traffic.
- Per node, even oracle admission exceeds TLC endurance by about 6 to 80x for interactive traffic,
  while API traffic falls within reach of TLC for MLA models.
- Device DRAM serves 12 to 46% of interactive reuse but about 90% of API reuse.
- For a dense GQA 70B model, loading KV from NAND never beats recomputing on 8 x H100; MLA models
  benefit.

## Related work and how this differs

HBF Sucks? (2608.11668, flash write excess on Bailian traces, on-package HBF); Son et al. and
Kyung et al. (CAL'26, HBF KV endurance); Lachesis (2610.08378), HBFlex (2609.18675) (lifetime-aware
placement); SPLASH (2609.23816) and HiFC (NeurIPS'25) (SLC or pseudo-SLC lifetimes, the main
counter-evidence); To Restore or To Recompute (HotStorage'26) and py-kvcache (2609.11744)
(load-versus-recompute break-even); KVCache-in-the-Wild (ATC'25), KVSET (2609.27746) (reuse and
working sets); Mooncake (FAST'25) and TokenLake (pool sharing); Flashield (NSDI'19) and SGLang
HiCache selective write-through (admission filters). This work adds the CXL device model, the
oracle bound, the interactive versus API split and device provisioning rules.

## Remaining work for a journal version

1. Sweep TLC, pseudo-SLC and SLC and BF16, FP8 and INT4 KV; state budgets in TB/day.
2. A page-mapped FTL with garbage collection to measure write amplification.
3. Quote HyMCache, ITME and LM-CXD exactly so the comparison is fair.
4. Redo the shared-pool check at Mooncake's scale (the quick check in analysis/quickchecks is too
   small and conflicts with Mooncake FAST'25).
5. Calibrate against Cylon (FAST'26) or published CMM-H measurements; add the Bailian thinking and
   coder traces.
