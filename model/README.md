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

Sanity check: the model's flag-polling ping-pong for small messages is 1.8 to 2.9 us across the
three latency classes; cMPI (arXiv 2510.05476) measures about 2.2 us.

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
with very short service times (1.2 to 1.4x). Not top-venue material as framed in IDEA.md.
