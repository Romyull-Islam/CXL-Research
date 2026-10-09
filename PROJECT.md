# Parked loads: full/empty synchronization for multi-host CXL memory

Status (2026-10-09): CAL letter draft ready (paper/parked_loads.tex, 4 pages, "Parked Loads:
Protocol-Safe Wait-Until-Written Reads for Multi-Host CXL Memory"). Novelty re-checked the same
day: not anticipated in CXL; a second search added IBM's CLT patent, Microsoft's conditional-read
patent, SSB and QOLB as prior art to cite and separate; the closest new items are Micron US12645581B2 (device defers
requests to busy lines) and Intel US11093398B2 (remote MONITOR/MWAIT); HPCA'13 already names
timeouts as future work, so the claim rests on the CXL-specific design. A calibrated multi-host
simulation replaced the analytical inputs: 1.3 to 4x lower handoff, RPC and barrier latency, and
waiting threads stop consuming a pooled device's bandwidth. Still too small for ISCA/MICRO/HPCA.

## Idea

Let a CXL memory device hold a read of a not-yet-written line until another host writes it. Each
line of a synchronization region carries a full/empty bit; a read of an empty line is parked in
the device and answered when a write fills it, bounded below the CXL.mem completion timeout with an
explicit empty response. This revives the HEP (1978) and Tera MTA full/empty bits and I-structures
(1989) for multi-host CXL pods.

## Problem and gap

CXL 3.x lets 8 to 16 hosts load and store the same memory, but hardware coherence across hosts is
absent or limited to a small region (a few hundred MB, per MEGALON OSDI'26 and Tigon OSDI'25). So
hosts synchronize in software:
- flush and poll with uncached reads: about 600 ns or more per poll, burning link and DRAM bandwidth
  (TraCT, CCCL, cMPI, RPCool);
- move ownership by back-invalidation: over 800 ns per handoff, so a contended counter saturates at
  about one operation per handoff;
- compare-and-swap near memory (Cxlalloc ASPLOS'26, about 2.3 us at the median), still followed by
  polling.

CXL.mem has no way to say "answer this read when the data is ready". Memory-side waiting exists
outside CXL (Lustig and Martonosi HPCA'13, one CPU and one GPU; Callback ISCA'15, on chip), but not
for many independent hosts on CXL, where credits, timeouts, failures and fairness are new problems.

## Goal

1. A CXL.mem extension: F/E state in a sync region (device SRAM or per-line metadata), parked reads
   that do not exhaust credits or deadlock, a bound below the completion timeout with an empty
   response, consuming and non-consuming reads that are safe under speculation, and behavior when a
   host or a device head fails.
2. Device and host support with its cost (parked-request table, core resources).
3. An evaluation against flush polling, per-line valid polling, coherent spinning with
   UMONITOR/UMWAIT, and queued atomics, on microsecond-scale workloads.

## Motivation

- Multi-host CXL systems already pay the polling cost: cMPI measures 2.2 us per flushed access
  (2.8x a cached one) and about 12 us per MPI ping-pong; HydraRPC measures 1.47 us per 64 B RPC; Cxlalloc's allocator drops to about 1% of hardware-coherent throughput under
  near-memory CAS; PNNL (MEMSYS'24) concludes hardware support is needed.
- The device is where every host's requests meet, so it is the only place that sees both the
  waiting read and the releasing write.
- First-order model (model/): parked loads cut one-way handoff latency 1.5 to 1.7x for 64 B to 4 KB
  messages against the best software protocol at device latencies of 250 to 600 ns; end to end, 1.2
  to 1.4x for cross-host RPC with service times of 2 us or less. AI serving gains at most about 2%
  because bulk transfer and compute dominate, so the paper targets microsecond-scale communication.

## Related work

| Work | What it does | What it does not do |
|---|---|---|
| Lustig and Martonosi, HPCA'13 | F/E bits in GPU DRAM; GPU memory controller holds requests until the CPU writes over PCIe | One CPU and one GPU; no independent hosts or CXL constraints |
| Callback, ISCA'15 | Special callback read blocks in an on-chip directory until the next write | On chip, special instruction |
| Intel US12455827 | Queued contended atomics in a multi-host CXL device, granted by completion write | No F/E on plain loads |
| Intel US20250021228A1 | Pending lock queue in a CXL device for independent hosts | Explicit lock requests, no F/E data |
| Cxlalloc (ASPLOS'26); AMD arXiv 2404.03245 | Near-memory CAS or T&S registers | No waiting primitive |
| SynCron (HPCA'21); LRSCwait (DATE'24); Active Memory Operations (ICS'07) | Memory-side synchronization in one system | No fabric, no CXL |
| Wu et al., J. Systems Architecture 2026 | Software spin locks and semaphores on VM-emulated CXL shared memory | No device mechanism |
| cMPI, RPCool, TraCT, CCCL | Cross-host communication over CXL with polling | No hardware support |

Patents also checked: KnuEdge US10203911B2, ICT-CAS US9411778B2, Ericsson US8385354B2 (waiting reads
outside shared CXL memory).

## Research questions

- Protocol: address-range attribute or new opcode; where F/E state lives; credit and deadlock
  rules; timeout bound; failure semantics.
- Host: parked loads hold a reorder-buffer entry and a fill buffer; speculation must never consume a
  FULL value; interaction with UMWAIT and stall-switching cores.
- Device: parked-table size and organization, wakeup order and fairness, whether a fetch-and-add
  unit is also needed.

## Expected reviewer objections

1. Significance: about 300 ns saved per handoff. Answer with microsecond RPC, CPU time not spent
   spinning, and scaling with many waiters.
2. "HPCA'13 or ISCA'15 moved to CXL". Answer with what only multi-host CXL has: independent failure
   and trust domains, credits, the completion-timeout bound, the missing empty response, fairness.
3. Simulation validity: calibrate latencies against published devices and sweep link latency.

## Remaining work for a CAL letter

Done on 2026-10-09: framing (microsecond-scale cross-host communication), CXL-specific design points
(credits, RwD bypass, 25 us bound, empty encoding, speculation, failures, fairness, table cost), and
a calibrated multi-host simulation. Left for the author:
1. Optional: read the full DaMoN'26 paper in a browser (DOI 10.1145/3789237.3809131; open access but
   blocked to scripts). Its verified abstract is already cited in the letter; check its
   write-visibility numbers against the 1.4 us poll cost used in the model.
2. Bibliography: done (24 entries verified, paper/p3.bib).
3. Check VAMOs (MICRO'26, 4 Nov) and the HPCA'27 list (2026-11-06) before submitting.
4. Add affiliation and compile with the IEEE CAL template.

## Files

model/ holds the first-order model (`python3 model/run_model.py`) and the discrete-event simulation
(`parksim.py`, `experiments.py`, `run_contention.py`); paper/ holds the CAL letter and its figures;
notes/idea-search-2026-10.md records the 16 ideas checked before this one was chosen.
