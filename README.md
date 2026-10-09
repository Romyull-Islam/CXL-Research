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

Status: IEEE CAL letter draft in paper/ (3 pages). Novel in CXL as of 2026-10-09; the gain (1.3 to
4x on microsecond-scale handoffs, RPCs and barriers) is too small for a top conference. See
PROJECT.md for the author's remaining items.

## Layout

    PROJECT.md                       everything: idea, gap, goal, motivation, related work, evidence, plan
    model/                           first-order model, SimPy multi-host simulation and results
    paper/                           CAL letter (parked_loads.tex) and figures
    notes/idea-search-2026-10.md     the 16 ideas checked on 2026-10-08 and why this one was chosen
    chat/                            full conversation that produced the idea (not tracked by git)

## Run

    python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
    python3 -m pytest model/test_parksim.py     simulator checks
    python3 model/experiments.py                latency, RPC and barrier results
    python3 model/run_contention.py             waiters sharing a switched device with tenants
    python3 model/run_table.py                  parked-table size and overflow
    python3 paper/figures.py && (cd paper && tectonic parked_loads.tex)
