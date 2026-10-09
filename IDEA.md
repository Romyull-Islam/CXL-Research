# Parked loads: full/empty synchronization for multi-host CXL memory

Status (2026-10-09): passed the novelty gate (not anticipated in CXL), failed the significance
gate. The first-order model in model/ shows at most about 2% end-to-end gain for AI serving and
1.2 to 1.4x only for microsecond RPC with very short service times. Not pursued as a top-venue
paper in this form. The motivation section below was written before the model and is kept for
the record.

## Problem and gap

CXL 3.x lets many hosts (8 to 16 in a pod) load and store the same memory through a switch or a
multi-headed memory device. What it does not give them cheaply is a way to wait for each other.
Hardware coherence across hosts is either absent (CXL 2.0 sharing) or limited to a small region,
because the device's snoop filter cannot track terabytes; MEGALON (OSDI'26) and Tigon (OSDI'25)
both cite a practical coherent region of a few hundred MB. Everything else is synchronized in
software, and there are only three ways to do it today:

1. Poll: flush the cached copy, re-read the line uncached, repeat. Each poll is a link round
   trip of about 600 ns or more, and every waiting host keeps consuming link and DRAM bandwidth.
2. Move ownership through back-invalidation: more than 800 ns per handoff (arXiv 2502.11046), so
   a shared counter or queue tail saturates at roughly one operation per handoff whatever the
   number of hosts.
3. Run a compare-and-swap near memory (Cxlalloc, ASPLOS'26: about 2.3 us at the median on an
   FPGA device), and still poll for the result or for the next change.

The gap: CXL.mem has no way to say "answer this read when the data is ready". No published CXL
system, device design or patent that we found lets a read of a not-yet-written line wait inside
the memory device and complete when a producer writes it. The closest CXL designs queue atomic
operations or lock requests in the device and grant them later (Intel US12455827 and
US20250021228A1), or execute CAS near memory with no waiting at all (Cxlalloc; AMD,
arXiv 2404.03245).

Memory-side waiting itself is not new. Full/empty bits held requests in the GPU memory
controller until the CPU wrote over PCIe in Lustig and Martonosi (HPCA'13), and Callback (Ros and
Kaxiras, ISCA'15) blocked a special read in an on-chip directory until the next write. What has
not been done is the multi-host CXL version, where the hard parts are specific to CXL: many
independent hosts with separate failure and trust domains, plain MemRd requests under credit
flow control, completion timeouts that isolate the link when they fire, and no "not ready"
response in the protocol. That realization is the contribution.

## Goal

Design, specify and evaluate a parked-load primitive for CXL memory pods:

- the device keeps a full/empty state for each line of a synchronization region;
- a read of an empty line is parked in the device and answered when a write fills the line;
- parked reads are bounded by the CXL.mem completion timeout and end with an explicit "empty"
  response rather than an error;
- hosts get a programming model (queues, barriers, mailboxes, allocators) built on it.

The deliverables are the CXL.mem protocol extension (semantics, encoding, credit and deadlock
rules, timeout and failure behavior), the device and host hardware support with its cost, and a
simulation study showing where it beats the best available alternatives. Target venues are ISCA,
MICRO, HPCA and ASPLOS.

## Motivation

Why now. Software built on CXL pods is already paying this cost. TraCT (arXiv 2512.18194) hands
the KV cache from prefill to decode across hosts with polling; CCCL (arXiv 2602.22457) runs GPU
collectives over a CXL pool with doorbells; cMPI (arXiv 2510.05476) measures about 2.2 us for a
cross-host ping-pong with flushes; Cxlalloc reports its allocator falling to about 1% of the
hardware-coherent throughput when it must rely on near-memory CAS. PNNL's study of locks on CXL
memory (MEMSYS'24) concludes that hardware support for synchronization is needed.

Why the device. The CXL memory controller sits in the device, where the requests of every host
meet. It is the only place that sees both the waiting read and the write that should release it,
and it does not have to trust any host to do so.

Why an old idea fits. Full/empty bits were built into the HEP (1978) and the Tera MTA, and
I-structures (1989) used the same semantics for dataflow. They disappeared because coherent
caches made spinning on a cached flag nearly free. In a CXL pod that assumption fails again:
spinning means crossing the link. CXL also removes the obstacles those machines had to solve
themselves: requests and write data travel on separate channels, and hosts already tolerate long
and variable load latency on CXL memory, with recent cores able to switch away from a stalled
CXL load in under 20 ns (LiteSwitch, OSDI'26).

Why it matters for AI serving. Multi-host inference is full of fine-grained handoffs: mixture-of-
experts token dispatch at decode time (a few hundred KB per step, about 15 us to move, where a
2 to 3 us synchronization is a visible share), small collectives, pipeline-stage activation
handoff, and layer-by-layer KV streaming from prefill to decode. Bulk transfers are not the
target: for a single large transfer, a microsecond saved is noise.

Why simulation is acceptable. CPUs with multi-host CXL 3.x support are not expected before 2027,
so every evaluation of this space is in simulation or on FPGA prototypes. Architecture venues
accept that for hardware that does not exist yet (for example Cohet, HPCA'26; SkyByte, HPCA'25).

## Related work

CXL multi-host sharing systems (all synchronize in software):
TraCT (arXiv 2512.18194), CCCL (arXiv 2602.22457), cMPI (arXiv 2510.05476), RPCool
(arXiv 2408.11325), Tigon (OSDI'25), MEGALON (OSDI'26), Pasha (CIDR'25), CXL-SHM (SOSP'23),
PhasedStore (HPCA'26), Beluga (SIGMOD'26).

Near-memory atomics and synchronization for CXL:
- Intel US20220114098A1, granted as US12455827: device "atomic memory regions" for multiple hosts
  behind a CXL switch; contended atomics are queued and granted later through a completion write;
  timeout returns poison. Closest prior art. It does not give plain loads full/empty semantics.
- Cxlalloc (ASPLOS'26): near-memory CAS on an Intel Agilex FPGA for pods without coherence; no
  waiting primitive.
- AMD, "Memory Sharing with CXL" (arXiv 2404.03245, HCDS'24): test-and-set and CAS registers per
  port on a dual-headed device.
- Cohet (HPCA'26) and "When CXL Meets Remote Atomic Operations, Cache Coherent NIC Demand
  Rearchitecturing" (ICDCSW'26): remote atomics offloaded to a CXL.cache NIC.
- Intel US20250021228A1, "Locking hardware for a shared memory" (filed 2024): lock requests from
  independent hosts wait in a pending queue in the device and are granted on release; the text
  mentions semaphores and doorbells. Claims require explicit lock requests with keys and a lock
  table; no full/empty data, no data returned on write, no timeout with an empty response.
- Suetterlein et al., PNNL (MEMSYS'24): software locks on a CXL prototype.
- Wu et al., "CXL shared coherent memory simulation and cross-host synchronization mechanisms
  design for data sharing", Journal of Systems Architecture 174, May 2026
  (DOI 10.1016/j.sysarc.2026.103723), extending ISEDA'25. Checked: a functional simulation of
  shared memory between virtual machines, with software spin locks and semaphores; no device
  mechanism. A software baseline, not competing work.
- Martins, Baumstark, Becher and Sattler, "Characterizing Multi-Host CXL 2.0: Latency, Bandwidth,
  and Synchronization Costs" (DaMoN'26, DOI 10.1145/3789237.3809131): measured synchronization
  costs on real multi-host CXL 2.0. Not openly available yet; likely the best source of measured
  motivation numbers.

Waiting reads and memory-side synchronization outside CXL:
- Lustig and Martonosi, "Reducing GPU offload latency via fine-grained CPU-GPU synchronization"
  (HPCA'13; Princeton application US20140240327A1, abandoned): full/empty bits in GPU DRAM, with
  requests held in the GPU memory controller until the CPU's write over PCIe fills the bit. The
  closest precedent; one CPU and one GPU, no multiple hosts, no timeout.
- Callback (Ros and Kaxiras, ISCA'15; US10387312B2, US11237966B2): a callback read blocks in an
  on-chip directory until the next write, with per-core full/empty bits and a callback queue.
  Needs a special instruction; on chip only.
- KnuEdge US10203911B2 (CAM tuple spaces, stalled and consuming reads), ICT-CAS US9411778B2
  (synchronization engine with produce bits), Ericsson US8385354B2 (blocking queue reads with a
  timeout error), Vlassov and Moritz (2000, postponed memory operations).
- SynCron (HPCA'21, synchronization engines for near-data processing), LRSCwait/Colibri (DATE'24,
  memory-side waiter queues without polling), Active Memory Operations (Zhang, Fang and Carter,
  IPDPS'04 and ICS'07), IBM Blue Gene/Q L2 atomics and wakeup unit, QOLB (1989).

Full/empty bits and dataflow memory:
HEP (Smith, 1978), Tera MTA (1990), I-structures (Arvind, Nikhil and Pingali, 1989), NYU
Ultracomputer combining and fetch-and-add (1983).

CXL evaluation without hardware:
SimCXL / Cohet (HPCA'26), CXL-DMSim (TCAD 2025), CXL-ClusterSim (arXiv 2605.27745), Mess
(MICRO'24), calibrated against "Demystifying CXL Memory" (MICRO'23) and Melody (ASPLOS'25).

Before submission, re-run the novelty search; this area changes within weeks. The record of the
search done on 2026-10-08 and 2026-10-09, including the 16 ideas checked and the final patent and
arXiv sweep for this one, is in notes/idea-search-2026-10.md.
