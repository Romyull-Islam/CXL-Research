# Parked loads in gem5

Full-system-style evaluation of parked loads with gem5 (v25.1.0.1, stable branch, commit
f5c5a6e390f5) in syscall-emulation mode. Several hosts, each its own gem5 `System` with its own
cores, caches and coherence domain, share one CXL Type-3 memory device that can park reads.

## Layout

- `src/` the device model, built into gem5 as an EXTRAS directory: `ParkedMemory` (parked_mem.hh,
  parked_mem.cc, ParkedMemory.py). Timing follows gem5's SimpleMemory; every access costs at least
  one 64 B line of device bandwidth. A sync region has one full/empty bit per line and two address
  windows: window 0 (a read waits while the line is empty; a write sets it full and answers the
  parked reads) and window 1 (consuming read; a write sets the line empty). A parked read is
  answered with the current contents after `park_timeout` (25 us), and a read that finds the
  parked table full is answered at once.
- `configs/multihost.py` the multi-host system: per host, out-of-order (or timing) x86 cores with
  32 KiB L1s, a 1 MiB L2 with a stride prefetcher, local memory, and a link to the device (a
  Bridge with the one-way flight delay and a crossbar with 32 GB/s per host); all links meet at a
  shared crossbar in front of the device port (64 GiB/s). The idle uncached round trip is
  124.5 ns + 2 x link delay: 63 ns, 78 ns and 238 ns give the 250 ns expander, 280 ns multi-headed
  device and 600 ns switch used throughout. The config maps the device into every process at
  fixed virtual addresses: payloads write-back cacheable, the two sync windows uncacheable (so
  loads to them are never issued speculatively).
- `bench/sync_bench.c` the programs that run on the simulated hosts (static x86-64): one-way
  handoff, RPC, central barrier, streaming tenant, waiter and poster. Wait protocols: `flush`
  (clflushopt + mfence + load of a write-back flag), `uc` (uncached load, parking off), `bN`
  (uncached polling with exponential backoff from 64 ns to a cap of N ns; experiments bo_*) and
  `parked`. Latencies use clock_gettime, which returns simulated time in SE mode, so a timestamp
  written by one host can be compared on another.
- `run_gem5.py` all experiments; results in `results/*.csv`.

## Build and run

    git clone -b stable https://github.com/gem5/gem5.git ../gem5     # not in git; ~10 GB built
    cd ../gem5 && git checkout f5c5a6e390f5
    scons build/X86/gem5.opt -j6 EXTRAS=$PWD/../gem5-model/src         # about an hour
    cd ../gem5-model/bench && gcc -O2 -static -mclflushopt -o sync_bench sync_bench.c -lm
    cd .. && WORKERS=5 python3 run_gem5.py oneway rpc barrier
    WORKERS=1 python3 run_gem5.py contention longwait table          # up to 7.5 GB per run

## Known limits

- gem5 charges about 13 ns for clflushopt + mfence on a line that is not cached; measured hardware
  pays about 1.4 us (cMPI). The `flush` protocol here is therefore the cheap end of polling, which
  understates the benefit of parking.
- x86 has no 64 B uncached load, so the programs signal with a flag line and read the payload
  separately through write-back memory; parking removes the wait for the flag, not the payload
  read.
- Hosts are independent gem5 Systems; nothing models a CXL switch's internal arbitration beyond
  the shared crossbar, and BI coherence is not modeled.
