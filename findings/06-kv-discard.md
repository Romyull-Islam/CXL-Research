# KV-aware discard: a mechanism candidate (2026-10-09)

Question: request-scoped KV on CXL.mem reaches write amplification 5.2 because the device never
learns that KV is dead. What does telling it save?

Setup: analysis/kv_discard.py, results/kv_discard_first.csv (first version), results/kv_discard.csv (with latency and reads). Bailian thinking trace, decode KV written
per 16-token block and freed at request end; write-back LRU DRAM cache of a share of the peak live
KV in front of ftl.FTL (7% over-provisioning, 64 blocks per erase block).

Flash programs per host block written:

| device DRAM / peak live KV | today (FIFO reuse, no discard) | LIFO reuse only (host change) | per-request discard |
|---|---|---|---|
| 0    | 5.17 | 7.19 | 2.08 |
| 10%  | 5.12 | 4.50 | 1.07 |
| 25%  | 4.36 | 2.44 | 0.49 |
| 50%  | 3.00 | 0.90 | 0.17 |

Discard means the serving engine sends one command per finished request (request completion rate,
tens to hundreds per second per node, so the slow CXL.io mailbox path suffices); the device drops
the request's dirty DRAM blocks without writing them and trims them in the FTL. Most of the gain
comes from the DRAM drop, which Lee et al. (HotStorage'25, TRIM over CXL.io, analytical WAF model,
VM workloads) do not model; they leave the initiator layer and TRIM frequency open. LIFO slot reuse
helps only with device DRAM and hurts without it.

## With command latency, reads and the reuse-after-acknowledgement rule

results/kv_discard.csv: discards queue on one serial command channel and each takes `lat`; a slot
is reused only after its discard completes; each written block triggers `reads` random reads of the
request's earlier blocks. Flash programs per host block written:

| DRAM / live KV | reads per write | no discard | discard, 1 ms | discard, 100 ms | discard, 1 s (overloaded) |
|---|---|---|---|---|---|
| 10% | 0 | 5.12 | 1.05 | 1.05 | 2.60 |
| 10% | 1 | 5.20 | 1.28 | 1.28 | 3.01 |
| 10% | 4 | 5.23 | 1.55 | 1.55 | 3.55 |
| 25% | 0 | 4.40 | 0.48 | 0.48 | 1.53 |
| 25% | 1 | 4.84 | 0.63 | 0.63 | 1.83 |
| 25% | 4 | 5.24 | 0.86 | 0.86 | 2.31 |

Up to 100 ms per command the gain is unchanged (3.4 to 9x); slots waiting for an acknowledgement
peak at 3.5% of the live KV, which sets the extra logical space needed. The channel fails only when
request completions per second times the command latency exceed 1 (here 1.5/s x 1 s), and then
the host stalls. At node scale (up to about 100 completions/s) one command must take under 10 ms,
or one command must cover several requests. Absolute sizes: peak live KV on this trace is 714 GB
for Llama-3.1-70B and 154 GB for DeepSeek-V3, so 10 to 25% is 15 to 38 GB of device DRAM for
DeepSeek-V3 (within CMM-H's 48 GB) but 71 to 178 GB for Llama-3.1-70B.

The event stream had a bug: 194 requests with fewer than 16 output tokens were freed before their
only block was written. Fixed in lifetime_waf.request_scoped_stream; the published write
amplification moves from 2.081/5.171 to 2.078/5.169 (results/lifetime_waf_scoped_recheck.txt).

Gates: novelty partly checked (trimcxl, Micron US12393361 lifetime hints via ZNS); significance
shown on one trace; feasibility needs a device model with discard latency and the reuse-after-ack
rule trimcxl names. Not yet in the paper.
