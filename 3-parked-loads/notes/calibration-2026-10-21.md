# gem5 against a real machine (2026-10-20/21)

Native build of bench/sync_bench.c (-DNATIVE; the gem5 binary is unchanged byte for byte) on dgxa100,
two-socket AMD EPYC 7742: two hosts on cores 216 and 220 (socket 1), shared memory bound to NUMA node 1
(socket 0), 2,000 messages per run, 5 repetitions, median (results/native/, run_native.sh). A consumer
descheduled long enough for the producer to overrun it drops that sample (at most 2 of 2,000 per run).
Native flush round trip 616 ns; gem5 matches it at a 225 ns link delay (588 ns; results/cal_rtt.csv).
gem5 at 225 ns: results/cal_gem5.csv (flush, lines), cal_gem5_hw.csv (uc, parked, wbpark);
merged in results/cal_summary.csv (python calibrate_native.py report).

Findings
- Polls per message match: native flush 35 to 37, lines 34/32/27/11 (64 B to 4 KiB); gem5 34 to 36 and
  34/32/25/8. The device reads that cost other tenants bandwidth, the paper's main argument, are what a
  real CPU issues.
- Latency of the software baselines is overestimated by gem5 at the matched round trip: flush one-way
  +22 to +77%, lines one-way +26 to +89%, rpc flush +8 to +23%, rpc lines +11 to +69%.
- Against the native baselines, gem5's parked flag is still 1.19x (64 B) faster than native flush, but
  the parked line fill (wbpark, 1,163 ns at 64 B) is slower than native lines (764 ns). The paper's
  latency claims (parked flag 1.02 to 1.51x, parked fill 1.25 to 1.28x faster than software) rest on
  gem5's software latencies and cannot be stated as measured speedups.
- Caveat: the native shared memory is coherent cross-socket DRAM, not a non-coherent CXL device; the
  comparison mixes a simulator with hardware. It bounds how much the latency results can be trusted.

## Write-back adjustment (2026-10-21)

The coherent server lets the reader take a fresh line from the writer's cache (the reader's own
clflushopt pulls it), which a host sharing CXL memory cannot: there the line is visible only after the
writer's write-back. Measured write-back (store a line, clflushopt, mfence; test `wb`, built with -DCAL
so the paper's gem5 binary stays byte-identical): native 757 ns (median of 5,
results/native/native_wb.log), gem5 594 ns at 225 ns (results/cal_wb.csv). Adding one native write-back
per message direction (results/cal_summary.csv, *_adj columns):
- gem5 within -15 to +2% of the server for messages up to 1 KiB and for RPCs; 4 KiB is 30 to 50%
  slower in gem5 (it does not overlap clflushopt across lines).
- Parked line fill vs adjusted native line polling at 64 B: 1.31x faster (gem5 alone: 1.24x).
- Parked flag vs adjusted native flush polling: 1.46 to 1.61x (64 B to 1 KiB, RPCs).
The paper keeps its latency claims for one-line messages and reports this calibration (Section V).
