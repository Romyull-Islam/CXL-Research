# Real-drive check (2026-10-12 to 2026-10-13)

Drive: the author's SanDisk Extreme Portable 1 TB (WD SN580E, ASMedia USB bridge, UNMAP supported),
on the workstation over USB, whole drive used, TRIM via UNMAP. Code: analysis/ssd_streams.py (operation
files), analysis/ssd_test.py (standard library only, needs root). Results: results/ssd/.

The drive does not report NAND writes (logs 0x09, 0xC0, 0xCA, 0xD0 fail through the bridge), so the
measure is steady host write bandwidth with the drive full; slowdown against sequential overwrites is
the proxy for write amplification.

| stream | written TB | steady MB/s | slowdown | FTL WAF at 7% |
|---|---|---|---|---|
| sequential | 1.00 | 393 | 1.00 | 1.00 |
| LRU prefix, admit all | 1.50 | 400 | 0.98 | 1.00 |
| request-scoped, TRIM | 1.50 | 172 | 2.28 | 2.08 |
| request-scoped, no TRIM | 1.50 | 66 | 5.95 | 5.17 |
| uniform random | 1.03 | 44 | 8.93 | 7.53 (closed form) |

Same order as the FTL; the drive runs 10 to 19% above it wherever garbage collection works. Missing
TRIM costs 2.6x (2.5x simulated). The random phase was stopped at 1.03 TB (steady well before) and the
KV streams were shortened from 3 to 1.5 capacities to save about 20 hours.

Discard (BLKDISCARD of freshly written ranges): idle medians 0.5 ms (1 MiB), 3.0 ms (64 MiB), 35 ms
(1 GiB), 82 ms (4 GiB). Behind 24 MiB writes: 22 to 77 ms for 16 MiB to 4 GiB, but 0.9 s for 1 MiB
(queued behind in-flight writes). The first run's under-load numbers (discard.csv, load=writing) are
idle in fact: its writer thread failed on alignment; discard_load.csv has the valid under-load data.
