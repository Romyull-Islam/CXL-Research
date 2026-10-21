# Parked-table controller logic

`parked_table.py` is the logic around one row of the 8-way parked table, in Amaranth: it parks a read
of an empty line or answers it at once (full line, ownership read under R5, or no free way), wakes a
line's waiters on a write (oldest first; a consuming window wakes only the oldest consuming read),
answers expired entries *not yet* (R3, R4) and drops a failed host's entries. The table and the
full/empty bits are SRAM; their cost is in `../cacti/`.

    uv venv ../.venv && VIRTUAL_ENV=../.venv uv pip install amaranth yowasp-yosys   # once
    mkdir -p lib && curl -L -o lib/NangateOpenCellLibrary_typical.lib \
      https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/master/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib
    python parked_table.py test     4,000 random operations against a Python reference: 0 mismatches
    python parked_table.py synth    Verilog (parked_row.v), Yosys + ABC on NanGate45, then sta.py

`sta.py` times the mapped netlist with the library's delay and transition tables (worse of rise and
fall per arc, fanout pin load, no wire load). Result (`timing.txt`, `stat.txt`), 45 nm typical corner:

| | |
|---|---|
| logic | 6,936 um2 (0.007 mm2), plus 6,019 um2 of input and output registers added only for timing |
| critical path | 1.20 ns over 24 cells (0.83 GHz) |

One row operation therefore fits in about one cycle at 800 MHz, next to the 0.32 ns SRAM access
(CACTI, 22 nm) and the 100 ns device access of the gem5 model. The NanGate library is not
redistributed here (lib/ is ignored).
