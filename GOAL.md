# Parked loads: full/empty synchronization for multi-host CXL memory

Status (2026-10-09): passed the novelty gate in CXL (no paper or patent parks CXL.mem reads on
full/empty state), failed the significance gate for a top conference (at most about 2% end to end
for AI serving). New target: an IEEE Computer Architecture Letters paper, where a short proposal
with first-order evaluation fits. Full details in IDEA.md and PROPOSAL.md.

## Problem and gap

Hosts in a CXL pod share memory without hardware coherence across hosts (or with only a small
coherent region), so they synchronize by flushing and polling over the link. CXL.mem has no way to
say "answer this read when the data is written". Memory-side waiting exists outside CXL (Lustig and
Martonosi, HPCA'13; Callback, ISCA'15), but not for many independent hosts on CXL.

## Goal

Specify a parked-load extension: the device keeps a full/empty bit per line of a sync region and
holds reads of empty lines until a write fills them, bounded below the CXL.mem completion timeout
with an explicit empty response; evaluate it against flush polling, per-line valid polling,
coherent spinning and queued atomics.

## Motivation and evidence

The first-order model (model/) shows 1.5 to 1.7x lower handoff latency for 64 B to 4 KB messages
against the best software protocol, and 1.2 to 1.4x end to end for microsecond-scale cross-host
RPC with short service times.

## Related work

Lustig and Martonosi (HPCA'13); Callback (ISCA'15); Intel US12455827 and US20250021228A1; Cxlalloc
(ASPLOS'26); AMD "Memory Sharing with CXL" (arXiv 2404.03245); SynCron (HPCA'21); LRSCwait
(DATE'24); Wu et al. (JSA 2026, software locks); cMPI, RPCool, TraCT, CCCL (software sync over CXL).

## Remaining work for a CAL letter

1. Frame around microsecond-scale cross-host communication, not AI serving.
2. Add the CXL-specific design points: credits, timeout bound, speculation safety, host failure.
3. Replace the analytical latency inputs with a small multi-host simulation or calibrate against
   the DaMoN'26 multi-host CXL measurements (DOI 10.1145/3789237.3809131).
