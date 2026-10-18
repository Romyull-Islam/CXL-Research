# Hardware cost with CACTI 7

CACTI 7 (github.com/HewlettPackard/cacti, commit 1ffd8df), 22 nm, itrs-hp cells and periphery,
built with `make`; run `./cacti -infile <name>.cfg`. Outputs are the .out files.

| Structure | Model | Area | Access | Read energy | Leakage |
|---|---|---|---|---|---|
| Full/empty bits for a 64 MiB region | 128 KiB RAM, 8 B reads | 0.389 x 0.297 mm = 0.116 mm2 | 0.44 ns | 8.6 pJ | 38.7 mW |
| Parked table, 4,096 entries of 16 B | 64 KiB cache, 16 B blocks, 8-way, keyed by line address | 0.184 x 0.355 mm = 0.065 mm2 | 0.32 ns | 15.6 pJ | 25.8 mW |

Both lookups are under 0.5 ns, against the device's 100 ns access in the gem5 model.
