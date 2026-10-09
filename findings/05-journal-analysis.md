# Journal analysis: consistent per-node scaling, measured write amplification, provisioning

This finding redoes finding 04 with three corrections and adds the pieces a journal version needs.
Scripts: analysis/nand_stream.py, ftl.py, endurance_sweep.py, lifetime_waf.py, pool_sharing.py,
mooncake_repro.py, provisioning.py; tests in analysis/test_ftl.py. Data: Mooncake (conversation,
tool-agent) and the four Qwen-Bailian traces (to-C chat A, to-B API B, thinking, coder).

## 1. Correction: one node is S copies of a trace, not one trace on a whole device

Finding 04 computed hit rates with the trace on a full 48 GB + 2 TB device, then multiplied the
write rate up to a node's prefill throughput. A node that serves S times a trace's traffic
interleaves S independent streams, which stretches reuse distances about S-fold, so the device
must be divided by S. analysis/nand_stream.scaled_node finds S as a fixed point (node prefill
tokens/s x duty = S x the trace's miss tokens/s at that scale).

S is 0.3 to 7 for Llama-3.1-70B, 0.5 to 14 for DeepSeek-V3 and 2.4 to 65 for Qwen2.5-7B. With
write amplification near 1 (28% over-provisioning), the corrected drive writes per day stay within
26% of finding 04, except Qwen2.5-7B on the API trace with oracle admission (47 instead of 26). The
larger change is write amplification at 7% over-provisioning (section 2): with oracle admission it
roughly doubles several cases, for example DeepSeek-V3 on Bailian chat (144 instead of 64)
and Llama-3.1-70B on the API trace (39 instead of 17).

## 2. Write amplification depends on how KV dies

A page-mapped FTL with greedy garbage collection (analysis/ftl.py) matches Desnoyers' closed forms
(SYSTOR'12): FIFO cleaning within 1% (8.61 vs 8.69 at 6.2% effective over-provisioning, 2.545 vs
2.544 at 27%), greedy within 5% of Eq. 11-12, and the published Table 1 values to three decimals.

| KV lifetime pattern (results/lifetime_waf.csv) | WAF, 7% OP | WAF, 28% OP |
|---|---|---|
| LRU prefix cache, admit all | 1.00 | 1.00 |
| LRU prefix cache, oracle admission | 1.01 | 1.00 |
| Request-scoped decode KV, freed with TRIM | 2.17 | 1.54 |
| Request-scoped decode KV, no TRIM (CXL.mem) | 6.92 | 2.13 |

An LRU cache evicts blocks roughly in write order, so erase blocks die whole; HiFC measured 1.02 on
a real drive and SPLASH assumes 1.15. KV freed when its request ends interleaves lifetimes inside
erase blocks; HBFlex (arXiv 2609.18675) reports about 30x for that pattern. CXL.mem has no TRIM, so a
freed block stays valid in the FTL until the host reuses its address, which triples write
amplification at 7% over-provisioning. Across the full sweep (section 4) prefix-cache WAF is at
most 1.08 at 28% over-provisioning, and up to about 3 at 7% under oracle admission, whose reads
reorder LRU eviction.

## 3. Shared pool versus per-host caches

results/pool_sharing.csv (4, 8, 16 hosts; total capacity 2 to 50% of a trace's unique blocks):
- against random routing a pool hits 1.05 to 4.5 times as often and writes 1.03 to 1.9 times fewer
  blocks (no duplicates across hosts);
- against cache-aware routing with load balancing (at most 25% imbalance) the pool's advantage is
  only 1.01 to 1.18 times in hits and 1.01 to 1.09 in writes;
- first-block affinity routing fails: many requests share a system prompt, so one host gets up to
  8x the mean load.

Mooncake FAST'25 (section 5.3.2) reports up to 136% more hits for its global pool than for local
caches with a cache-aware scheduler, on 10 prefill nodes with 3M tokens each. Reproducing that
setup (results/mooncake_repro.txt), the pool's advantage is +5 to +10% when the scheduler allows 5%
to 100% imbalance, +43% when it balances exactly, +240% with random routing and +165% when it
ignores load. Mooncake's figure lies inside this range, so the pool's value depends mostly on how
strictly the scheduler balances load; this replaces the too-small check in analysis/quickchecks.

## 4. Per-node wear and provisioning

Sweep: 6 traces x 3 models x BF16 / FP8 / INT4 KV x 3 admission policies x prefill duty 100% and
30%, on a 48 GB DRAM + 2 TB NAND device (results/endurance_sweep_2000GB.csv) and on the same NAND in
pseudo-SLC mode, 667 GB (endurance_sweep_667GB.csv). Write amplification comes from the FTL; for
the 25% of cases whose trace is too short to reach FTL steady state even after appending new-user
copies, it is taken as 1 (column waf_estimated).

Budgets (5-year ratings, datasheets collected 2026-10-09): TLC-1, 2 TB at 1 DWPD (Samsung PM9A3,
Micron 7450 PRO), 2 TB/day; TLC-3, 2 TB at 3 DWPD (Micron 7450 MAX, Kioxia CM7-V), 6 TB/day; pSLC,
667 GB at 35 DWPD (Micron XTR, TLC run as SLC), 23 TB/day; SLC, 2 TB at 50 DWPD (Solidigm
D7-P5810), 100 TB/day. TLC ratings for 128 KB sequential writes are 1.7 to 4.4x higher; using
1.7x (results/provisioning_seq.csv) moves only a few cases from pSLC to TLC.

Cheapest class that lasts five years (results/provisioning.csv), counts of trace x model x
precision cases:

| Workload | Admission | Duty 100% | Duty 30% |
|---|---|---|---|
| API (Bailian to-B) | all | none 8, SLC 1 | none 4, SLC 4, pSLC 1 |
| API | 2nd hit | SLC 4, pSLC 4, TLC-3 1 | TLC 6, pSLC 2, SLC 1 |
| Interactive (Bailian chat, coder, Mooncake) | all | none 32, SLC 4 | SLC 17, none 16, pSLC 3 |
| Interactive | 2nd hit | none 16, SLC 18, pSLC 2 | SLC 16, pSLC 16, TLC-3 2, none 2 |
| Thinking | 2nd hit | SLC 5, none 2, pSLC 2 | pSLC 5, SLC 3, TLC-3 1 |

Rules that follow:
1. Admit-all never fits TLC or pSLC: a hybrid KV device needs admission control.
2. KV precision is a lever of about one flash class: FP8 halves and INT4 quarters the NAND writes
   (0.49x and 0.24x over the sweep).
3. MLA models write 2.5 to 2.8x less than a GQA 70B model per node, and small models write the
   most, because their nodes prefill the most tokens.
4. TLC works only for API-like traffic (4 to 7% of computed KV is ever reused) at partial prefill
   duty, or for MLA with INT4 KV. Interactive traffic (22 to 47% reused; thinking 14 to 15%) needs
pseudo-SLC or SLC,
   and at full prefill duty nearly half of the cases exceed even 100 TB/day.
5. Bandwidth is a second wall: 11% of the cases need more than the 4.5 GB/s that a measured CMM-H
   sustains on its NAND path once the working set exceeds its DRAM cache, all of them with
   admit-all or Qwen2.5-7B.

## Reading

- The endurance wall of finding 04 stands after correct scaling and measured write
  amplification, and it is now stated as provisioning rules across flash type, KV precision,
  admission and prefill duty.
- Write amplification of an LRU KV cache is about 1, so the wall is the host write rate itself;
  request-scoped KV on CXL.mem (no TRIM) is the case that multiplies wear.
- Novelty (re-checked 2026-10-09): narrow but present. HBF Sucks? uses the same Bailian traces with
  unit write amplification, TLC only, no admission sweep, no device model and per-system budgets;
  HBFlex, Lachesis, 2609.07175 and 2609.39131 add admission or lifetime placement for HBF. No
  paper sizes hybrid DRAM-over-NAND CXL devices with these rules. Target: ACM TOS. Gate 3 still
  needs write amplification measured on a validated SSD emulator or real drive.
