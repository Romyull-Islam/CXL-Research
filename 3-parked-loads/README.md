# ParkedLoads

Full/empty synchronization for multi-host CXL memory.

Hosts in a CXL pod share memory, but hardware coherence across hosts is absent or limited to a
small region. Today a host that waits for data written by another host polls: it flushes its
cached copy and re-reads the line over the link until the value changes. This project studies a
different contract. The CXL memory device keeps a full/empty bit for each line of a
synchronization region, and a read of an empty line is parked inside the device until a
producer's write fills it. The idea comes from the HEP (1978) and Tera MTA machines and from
I-structures; CXL brings it back because the memory controller now lives in the device, where
every host's requests meet.

Status: IEEE CAL letter draft in paper/ (4 pages): five protocol-safety rules, model-checked in
TLA+, and a gem5 implementation run with x86 programs on up to 36 hosts. Main result: parked
waiters stop consuming a pooled device's bandwidth (1.4 instead of 39 to 750 device reads per
message) at equal or slightly lower handoff latency. See PROJECT.md for the author's remaining
items.

## Layout

    PROJECT.md                       everything: idea, gap, goal, motivation, related work, evidence, plan
    tla/                             TLA+ model of the five rules and the TLC runner (results.csv)
    gem5-model/                      gem5 device model (src/), multi-host config, x86 benchmark, results
    gem5-model/calibrate_native.py   gem5 against the same programs on a two-socket server (results/native/)
    rtl/                             table logic in Amaranth, reference test, Yosys/NanGate45 synthesis and timing
    cacti/                           SRAM area and access time (CACTI 7)
    model/                           first-order model and SimPy multi-host simulation (cross-check)
    paper/                           CAL letter (parked_loads.tex) and figures
    notes/idea-search-2026-10.md     the 16 ideas checked on 2026-10-08 and why this one was chosen
    chat/                            full conversation that produced the idea (not tracked by git)

## Run

    python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
    python3 -m pytest model/test_parksim.py     simulator checks
    python3 model/experiments.py                latency, RPC and barrier results
    python3 model/run_contention.py             waiters sharing a switched device with tenants
    python3 model/run_table.py                  parked-table size and overflow
    python3 tla/run_tlc.py                      model checking (needs Java; see tla/README.md)
    see gem5-model/README.md for building gem5 and running run_gem5.py
    native calibration: build bench/sync_bench.c with -DNATIVE on a two-socket machine and run
      gem5-model/results/native/run_native.sh; then python3 gem5-model/calibrate_native.py report
    see rtl/README.md for the table logic
    python3 paper/figures.py && (cd paper && tectonic parked_loads.tex)
