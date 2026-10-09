# Significance model (gate 2)

`python3 run_model.py` prints the tables below. The model is first order: it adds up link round
trips, serialization and polling delays for one cross-host handoff. It is meant for a go/no-go
decision, not as evidence for a paper.

## Inputs

| Parameter | Value | Source |
|---|---|---|
| Idle load round trip | 250, 400, 600 ns | ASIC and FPGA expanders measured at 214 to 394 ns (Melody, ASPLOS'25; Demystifying CXL Memory, MICRO'23); Samsung CMM-B pooled box 596 ns |
| Per-poll overhead (clflushopt + fence) | 50 ns | assumption |
| Per-host link bandwidth | 32 GB/s | x8 Gen5 class expanders (Micron CZ120 up to 36 GB/s) |
| Outstanding line requests per consumer | 48 | assumption, roughly a modern core's L2 miss queue |
| Device bandwidth | 60 GB/s | Samsung CMM-B |

Correction (2026-10-09): cMPI (arXiv 2510.05476) reports 2.2 us as a flushed memory access (Table 1),
not a ping-pong; its MPI ping-pong is about 12 us. Use the simulation below for calibrated numbers.

## Result (2026-10-09)

- Per handoff, parked loads beat the best software protocol by 1.5 to 1.7x for 64 B to 4 KB
  messages at every latency class, and by 1.1 to 1.4x for 16 to 64 KB.
- Polling a flag burns only 0.1 to 0.2 GB/s per waiter; 70 to 300 waiters are needed to take a
  quarter of a 60 GB/s device. Polling whole 4 KB messages burns 3 to 6 GB/s per waiter.
- End to end (compute times are assumptions):

| Workload | Handoff share of step | Upper-bound speedup |
|---|---|---|
| Cross-host RPC, 256 B, service 0.5 us | 79% | 1.39x |
| Cross-host RPC, 256 B, service 2 us | 49% | 1.21x |
| Cross-host RPC, 256 B, service 10 us | 16% | 1.06x |
| MoE dispatch + combine, 64 KB, layer 100 us | 27% | 1.02x |
| MoE dispatch + combine, 64 KB, layer 300 us | 11% | 1.01x |
| Pipeline-stage handoff, 16 KB, stage 200 us | 3% | 1.01x |

## Verdict

Fails the significance gate for AI serving: at most about 2% end to end, because bulk data
movement and compute dwarf the synchronization saved. Borderline only for microsecond-scale RPC
with very short service times (1.2 to 1.4x). Not top-venue material as originally framed; see PROJECT.md.

## Discrete-event simulation (2026-10-09)

`parksim.py` (SimPy 4.1.2) models hosts with their own links to one device with shared ingress
and egress ports, DRAM banks, acknowledged writes and a parked-read table with a 25 us bound (half
the shortest CXL Root Port completion timeout, 50 us). `experiments.py` runs one-way handoffs,
ping-pong, RPC, a 16-host barrier; `run_contention.py` adds tenant hosts that saturate the device.
`test_parksim.py` checks the idle round trip, parking, the timeout path and table overflow.

Calibration: idle load round trip 250 ns (expander), 280 ns (multi-headed device), 600 ns (switch)
from Octopus (NSDI'26, Fig. 2) and CXL-CCL (ICS'26, 658 ns); extra cost per poll 50 ns (uncached
load) or 1.4 us (flush plus fence: cMPI flushed 2.2 us vs cached 0.79 us). Sanity check: a
zero-work 256 B RPC on the 280 ns device is 0.82 us with line polling and 2.0 us with flag polling;
measured values are 1.2 us (Octopus) and 1.47 us (HydraRPC, ATC'24, 300 ns device).

Results (model/results_latency.csv, results_contention.csv):
- One-way, 64 B to 1 KB: parked 1.5 to 1.7x faster than the best software protocol at 50 ns poll
  cost, 2.5 to 4x at 1.4 us; 16 KB: 1.2 to 1.9x.
- RPC: 1.3 to 3.9x with no service time, 1.1 to 1.8x at 2 us, at most 1.2x at 10 us.
- 16-host barrier: 1.4 to 3.9x.
- Switched device with 4 streaming tenants (45.4 GB/s alone) and 512 waiting threads: tenants keep
  41.6 GB/s with parked loads against 25.6 to 26.2 GB/s with polling; handoff 1.1 us against 2.7
  to 6.5 us.

Run: `python model/experiments.py`, `python model/run_contention.py`, `python paper/figures.py`.
