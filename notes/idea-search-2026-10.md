# Idea search, 2026-10-08

Seven survey agents mapped CXL research up to October 2026, then every candidate idea went to a
separate agent whose only job was to find prior work that already does it. Sixteen ideas were
checked in four rounds. The full conversation, with every agent report, is in
chat/2026-10-08-idea-search.md.

## What the surveys found

- Crowded in 2025 and 2026: KV cache pooling and offload on CXL (Beluga, TraCT, HyMCache, ITME,
  LM-CXD, SAC), near-memory processing for attention, CXL compression (TRACE, IBEX), device-side
  hotness tracking (NeoMem, M5, S-Tiering), learned prefetching, MoE and LoRA offload, ANNS and RAG
  on CXL, and on-package High Bandwidth Flash (six or more papers in July and August 2026).
- One SK hynix and Virginia Tech group has three papers close to SemSched (HyMCache, ITME, TraCT).
- Samsung and Intel hold patents on many obvious CXL device mechanisms.
- Architecture venues accept simulation-only CXL papers when the contribution is new hardware;
  OS tiering-policy papers now draw pushback without a real device.

## Ideas checked and rejected

| Idea | Verdict | Prior work that decided it |
|---|---|---|
| Discard and lifetime hints for CXL flash (TRIM for load/store) | Incremental | Revisiting Trim for CXL Memory (HotStorage'25), LM-CXD (arXiv 2609.26828), Lachesis (2610.08378), HBFlex (2609.18675) |
| Cross-host RowHammer / PRAC interference in pooled devices | Not novel as framed | RogueRFM (2501.06646), LeakyHammer (MICRO'25), PRACLeak (ISCA'25), BreakHammer (MICRO'24) |
| Processing-using-DRAM (RowClone) inside CXL devices | Incremental | Micron row-copy patent US12498856B2, DTL (ISCA'23), CXL sanitize and Write Zeros commands |
| Coherence capacity and a "sealed" state for immutable AI data | Incremental | Cuesta et al. (ISCA'11, TC'13), Cohesion (ISCA'10), Netdev 0x1A (2026), AMD arXiv 2404.03245 |
| CXL shared memory for RL post-training weight sync | Incremental | Laminar, TensorHub, SparseRL-Sync, CCCL; element-level deltas move 6 to 30 times fewer bytes than dirty lines |
| Recomputability as a CXL memory attribute (drop and poison) | Not novel | Midas (NSDI'24), Soft Memory (HotOS'23), LM-CXD, Arm discard-eligible patents; the suffix-order rule was also wrong |
| Cross-host read coalescing and weight carousels | Weak | Fiber Memory (arXiv 2607.08407), combining networks, CENT; per-host link bandwidth makes the setting unrealistic |
| DevLoad telemetry as a cross-host attack surface | Novel only as a defense | ECN and PFC congestion-signal attacks; no shipping host reacts to DevLoad |
| CXL-native memory tagging for cross-host memory safety | Narrow | HPE MaKC patent US10592437, CEP (TACO'19), Intel US10877897, Arm CHI-E |
| Co-selection layout for sparse attention on flash | Not novel | SPLASH (arXiv 2609.23816), Swarm (2603.17803), DynaKV |
| Cooperative exclusive caching between host and hybrid device | Not novel | Samsung US12717485B2, SkyByte (HPCA'25), DEMOTE (ATC'02) |
| GPU load/store to hybrid CXL with device miss merging | Narrow | Sano et al. (SC-W'23), Yang et al. (ATC'23), SkyByte, HBF-Sim (arXiv 2609.29246) |
| Endurance-aware host tiering for hybrid devices | Narrow | RaPP (ICS'11), PDRAM (DAC'09), TMO (ASPLOS'22), SkyByte |
| Coded caching over CXL switch multicast | Not novel | Samsung US12536107B2 and US12229409B2, Adegbija and Tandon (ISVLSI'17) |
| Coded computation against flash stragglers | Not novel, weak premise | Tiny-Tail Flash (FAST'17), IODA (SOSP'21), Lee et al. (2017) |
| Full/empty parked loads plus combining fetch-and-add | Novel with caveats in CXL (chosen, without combining) | In CXL: Intel US12455827 and US20250021228A1, Cxlalloc (ASPLOS'26). Outside CXL: Lustig and Martonosi (HPCA'13), Callback (ISCA'15) |

## Final verification of parked loads (2026-10-09)

- Wu et al., JSA 174 (May 2026): read from the abstract and introduction. Functional simulation of
  CXL shared memory with VMs plus software spin locks and semaphores. Not competing work.
- arXiv: all 154 papers that mention CXL were scanned for synchronization, atomics, waiting,
  notification and doorbells. None proposes reads that wait in the device; the systems use CAS,
  locks or polling (DAXFS, DPC, cMPI, CCCL, TraCT, Cohet, Beluga and others).
- Patents (about 60 queries, USPTO full text plus web): not anticipated in CXL. Closest outside
  CXL: Lustig and Martonosi HPCA'13 (full/empty bits in GPU DRAM, held until the CPU writes),
  Callback ISCA'15 (blocking callback reads in a directory), KnuEdge US10203911B2, ICT-CAS
  US9411778B2, Ericsson US8385354B2. Closest in CXL: Intel US12455827 (queued atomics) and
  Intel US20250021228A1 (pending lock queue in the device).
- Consequence: claim the multi-host CXL realization, not memory-side waiting as such.

## Why parked loads was chosen

It is the only idea whose core mechanism no checker found in any CXL paper or patent (memory-side
waiting exists outside CXL, in HPCA'13 and ISCA'15), it is a hardware and protocol contribution of the kind architecture venues accept in simulation, and it
targets CXL 3.x pods that cannot be studied on real hardware until 2027. Its main risk is
significance: a parked read saves about half a polling round trip per handoff, so the paper has
to be built on latency-bound, fine-grained synchronization. Combining in the switch was dropped
to an ablation because pods of 16 hosts or fewer do not produce the hot-spot saturation that
justified it in the 1980s.

## Significance gate (2026-10-09)

The first-order model in model/ settles it: parked loads beat the best software protocol by 1.5
to 1.7x per small handoff, but that is at most about 2% end to end for AI serving and 1.2 to 1.4x
only for microsecond RPC with very short service times. Parked loads passes the novelty gate and
fails the significance gate, so it is not pursued as a top-venue paper in its current form.
