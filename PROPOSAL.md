# Parked loads: full/empty synchronization for multi-host CXL memory

Working title. Idea stage; novelty searched on 2026-10-08 (see "Novelty position" and the
must-read list before claiming anything).

Target venues: ISCA, MICRO, HPCA, ASPLOS. The contribution is a CXL.mem protocol extension plus
device and host support, evaluated in simulation. Architecture venues accept simulation-only
evaluation for hardware that does not exist yet, and CPUs that support multi-host CXL 3.x
sharing are not expected before 2027.

## Problem

CXL 3.x lets several hosts load and store the same memory, but in practice hardware coherence
across hosts is either absent or confined to a small region (vendors and MEGALON, OSDI'26, put
the practical coherent region at a few hundred MB). Systems built on CXL pods therefore signal
across hosts in software:

- flush the cached line, then poll it with uncached reads (TraCT, arXiv 2512.18194; CCCL,
  arXiv 2602.22457; cMPI, arXiv 2510.05476; RPCool, arXiv 2408.11325);
- move line ownership through back-invalidation, at more than 800 ns per handoff
  (arXiv 2502.11046), which caps a contended counter or queue tail at roughly one operation per
  handoff no matter how many hosts participate;
- use a compare-and-swap executed near memory (Cxlalloc, ASPLOS'26: about 2.3 us at the median
  on an FPGA device), still followed by polling.

Every poll costs a round trip of roughly 600 ns or more and consumes link and DRAM bank
bandwidth. A cross-host ping-pong in cMPI takes about 2.2 us with flushes. PNNL's study of
locks on CXL memory (MEMSYS'24) concludes that hardware support for synchronization is needed.

## Idea

The CXL memory device keeps a full/empty (F/E) state per line in a designated synchronization
region.

- A read of a FULL line returns data as usual. A "consuming" read may also set the line EMPTY.
- A read of an EMPTY line is not answered. The device parks it in a table and answers it when a
  write sets the line FULL. One write can release several parked readers.
- A parked read is bounded. Before the CXL.mem completion timeout of the requester would fire,
  the device returns an explicit EMPTY response (not poison), and the host software retries or
  blocks.

This revives the HEP (1978) and Tera MTA full/empty bits, Arvind's I-structures (1989) and
QOLB (1989). They faded because coherent caches made spinning on a cached flag cheap. Inside a
CXL pod that is no longer true, and three properties of CXL make the old mechanism practical:

1. The memory controller is in the device, where the requests of all hosts meet, so the device
   can hold waiters without any host trusting another host.
2. CXL.mem keeps requests and write data on separate channels, so a parked read need not block
   the write that will release it, if credits are managed for that case.
3. Hosts already have to tolerate long and variable CXL load latency (hundreds of ns, and up to
   microseconds on hybrid DRAM-over-NAND devices), and recent work shows a core can switch away
   from a stalled CXL load in under 20 ns (LiteSwitch, OSDI'26).

## Research questions

RQ1, protocol. How should an F/E access be expressed: an address-range attribute of a sync
region, or new MemRd/MemWr variants? Where does the F/E state live: device SRAM for a bounded
region, or the per-line metadata that CXL already carries (MetaField since 1.1, Extended
Metadata in 3.1, which travels only with write data and read data, not with read requests)?
How are parked requests kept from exhausting credits and deadlocking the request channel? How
is the parking bound related to the configured CXL.mem timeout? What happens to parked reads
when a host fails, or when one head of a multi-headed device fails?

RQ2, host. A parked load occupies a reorder-buffer entry and a fill buffer or MSHR. Speculative
or prefetched reads must never consume a FULL value. Should the sync region be mapped uncached,
and what does that cost? How does parking interact with UMONITOR/UMWAIT and with stall-switching
cores?

RQ3, device. How large must the parked-request table be, and how should it be organized? How
are waiters on one line woken (all at once, or in order)? Is a fetch-and-add unit in the device
needed alongside F/E for hot counters, and does combining of same-address operations matter at
pod scale (16 hosts or fewer per device)? The novelty check suggests combining is unlikely to
matter at this scale; treat it as an ablation.

RQ4, workloads. Where does cross-host synchronization latency, rather than bulk transfer,
bound end-to-end performance?

## Planned contributions

1. A characterization of cross-host synchronization cost in CXL pods (polling, back-invalidation
   ownership transfer, near-memory CAS) on calibrated simulation, using published measurements
   for the latency inputs.
2. A parked-load extension to CXL.mem: semantics, encoding, credit and deadlock rules, timeout
   behavior and failure semantics.
3. Device and host support for it, with cost estimates (SRAM, table entries, core resources).
4. An evaluation on latency-bound cross-host workloads, against the strongest baselines below.

## Workloads

The case must rest on fine-grained, latency-bound synchronization. A single handoff around a
bulk transfer is not enough: moving the KV cache of a 4K-token prompt for a 70B model takes
tens of ms, so saving a microsecond there is noise.

- Mixture-of-experts token dispatch at decode time across hosts: messages of a few hundred KB,
  about 15 us to move, where a 2 to 3 us synchronization is a visible fraction.
- Small-message collectives (below 64 KB) over a CXL pool, as in CCCL.
- Cross-host RPC and message passing (cMPI, RPCool).
- A shared memory allocator under contention (Cxlalloc reports its allocator dropping to about
  1% of the hardware-coherent throughput when it relies on near-memory CAS).
- Pipeline-parallel activation handoff between hosts, and layer-by-layer KV streaming from
  prefill to decode where the consumer starts on each layer as soon as it is written.
- Microbenchmarks: ping-pong, single and multi-producer queues, barriers.

## Baselines

1. Flush plus uncached polling, with backoff.
2. Near-memory CAS or fetch-and-add plus polling (Cxlalloc; AMD, "Memory Sharing with CXL",
   arXiv 2404.03245).
3. A small hardware-coherent (HDM-DB) region plus UMONITOR/UMWAIT, which may give event-driven
   wakeup through back-invalidation without poll traffic. Verify that UMWAIT wakes on a
   back-invalidation snoop.
4. Queued atomics with completion notification, as in the Intel patent below.
5. An ideal zero-cost synchronization bound.

## Metrics

Handoff latency (median and p99); throughput as hosts (1 to 16) and waiters grow; link and
device bandwidth spent on synchronization; host cycles and energy spent waiting; end-to-end
workload time; device cost.

## Methodology

There is no CXL device on the development machine (single-socket i9-14900K, 32 GB, RTX 4500
Ada), and remote-socket emulation is impossible on one socket. Plan:

1. A first-order analytical model of handoff latency and throughput for each workload, built
   before any simulator work. If no workload shows a meaningful end-to-end effect, stop or pivot.
2. A multi-host CXL simulator. Candidates: CXL-ClusterSim (gem5 plus SST, multi-host,
   arXiv 2605.27745), SimCXL/Cohet (HPCA'26, validated to about 3% against hardware, but check
   whether it models back-invalidation and multiple hosts), CXL-DMSim (TCAD 2025).
3. Calibration of idle and loaded latency against published device measurements
   ("Demystifying CXL Memory", MICRO'23; Melody, ASPLOS'25; CXL-DMSim), and sweeps over link
   latency (200 to 600 ns) so the conclusions do not depend on one assumed number.
4. A functional prototype of the programming model on shared memory between virtual machines
   (as in the ISEDA'25 paper below) to check correctness, not performance.

## Novelty position

Final verification (2026-10-09): a patent search (USPTO full text, Google Patents, EPO), a scan of
all 154 arXiv papers that mention CXL, and a check of the JSA'26 paper. Verdict: not anticipated
in CXL, but memory-side waiting has close precedents outside CXL (Lustig and Martonosi, HPCA'13;
Callback, ISCA'15). The paper must claim the multi-host CXL realization, not the mechanism.

No checked paper or patent parks CXL.mem loads on per-line full/empty state in a multi-host
device. Everything around that core already exists, and the paper must cite it up front.

| Work | What it does | What it does not do |
|---|---|---|
| Lustig and Martonosi, HPCA'13 (US20140240327A1, abandoned) | Full/empty bits in GPU DRAM; the GPU memory controller holds requests until the CPU's write over PCIe fills the bit | One CPU and one GPU; no independent hosts, no CXL protocol constraints, no timeout |
| Callback, Ros and Kaxiras, ISCA'15 (US10387312B2, US11237966B2) | A special callback read blocks in an on-chip directory until the next write; per-core full/empty bits; callback queue | Special instruction, on chip, no fabric |
| Intel US20220114098A1, granted as US12455827 | Device "atomic memory regions" for multi-host CXL behind a switch; contended atomics are queued and granted later through a completion write; timeout returns poison | No full/empty semantics on plain loads; notification by completion write, not by answering a parked read |
| Intel US20250021228A1, "Locking hardware for a shared memory" | Lock requests from independent CXL hosts wait in a device pending queue and are granted on release; mentions semaphores and doorbells | Explicit lock requests with keys and a lock table; no full/empty data, no data returned on write, no timeout with an empty response |
| KnuEdge US10203911B2; ICT-CAS US9411778B2; Ericsson US8385354B2 | Stalled and consuming reads in a CAM tuple space; a synchronization engine with produce bits; blocking queue reads with a timeout | On chip or network-processor queues; not address-indexed shared memory across hosts |
| Wu et al., JSA 174, May 2026 | Functional simulation of CXL shared memory with VMs; software spin locks and semaphores | No device mechanism; serves as a software baseline |
| Cxlalloc (ASPLOS'26) | Near-memory CAS on an FPGA device for a pod without coherence | No waiting primitive; clients poll |
| AMD, Memory Sharing with CXL (arXiv 2404.03245, HCDS'24) | Test-and-set and CAS registers per port on a dual-headed device | No F/E, no CXL.mem path |
| SynCron (HPCA'21) | Synchronization engines for near-data processing, with waiter lists | Single system, no CXL |
| LRSCwait / Colibri (DATE'24) | Memory-side waiter queues on chip, no polling | On chip, no fabric, no timeouts |
| Cohet (HPCA'26); "When CXL Meets Remote Atomic Operations" (ICDCSW'26) | Remote atomic operations offloaded to a CXL.cache NIC | Not Type-3 shared memory, no waiting reads |
| Active Memory Operations (Zhang, Fang, Carter; IPDPS'04, ICS'07) | Fetch-and-op in the memory controller, barrier counters | Single coherent system |
| HEP, Tera MTA, I-structures, QOLB | The original full/empty and queued-lock mechanisms | No multi-host fabric, no CXL protocol constraints |
| PNNL, MEMSYS'24 | Software locks on a CXL prototype; calls for hardware support | Motivation, not competition |

Read in full before writing the paper:

- Lustig and Martonosi, HPCA'13, and Callback, ISCA'15: the two precedents reviewers will cite.
  The related-work section must state exactly what multi-host CXL adds to each.
- Intel US12455827 and US20250021228A1: claim language on queued requests, pending queues and
  notification, to keep the design clear of them.
- Martins et al., "Characterizing Multi-Host CXL 2.0: Latency, Bandwidth, and Synchronization
  Costs" (DaMoN'26, DOI 10.1145/3789237.3809131): measured synchronization costs on real
  hardware, through the ACM library.
- "When CXL Meets Remote Atomic Operations, Cache Coherent NIC Demand Rearchitecturing"
  (ICDCSW'26, DOI 10.1109/icdcsw72724.2026.00005).
- Wu et al., JSA'26 (DOI 10.1016/j.sysarc.2026.103723): already checked from its abstract and
  introduction; software locks only.

Re-run the novelty search right before submission; this area moves within weeks.

## Expected reviewer objections

1. Significance. A parked read saves about half a polling round trip, roughly 300 ns per
   handoff, and bulk transfers dwarf it. Answer with latency-bound workloads, CPU time and
   energy not spent spinning, scaling with many waiters, and finer-grained cross-host
   pipelining that polling makes impractical.
2. "This is Lustig and Martonosi (HPCA'13) or Callback (ISCA'15) moved to CXL." Expect it from
   every reviewer. Answer with the parts that only exist in multi-host CXL: many independent hosts
   with separate failure and trust domains, credit rules for parked requests, a bound below the
   completion timeout that isolates the link when it fires, an explicit empty response that the
   protocol does not have today, speculation safety on plain loads, and fairness across hosts.
3. Host-side hazards: parked loads pinning core resources, and consuming reads under
   speculation. These need a concrete design, not a footnote.
4. Simulation validity: calibrate against published devices and show results across a range of
   link latencies.

## First steps

1. Read the must-read list and the closest works in full.
2. Write the first-order model and decide whether the significance holds.
3. Pick the simulator; confirm multi-host and back-invalidation support.
4. Implement the microbenchmarks and baselines 1 to 3.
