# Project context for Claude

Parked loads: full/empty synchronization for multi-host CXL memory. Read PROJECT.md first (everything is
in it), then gem5-model/README.md and tla/README.md.

## Where it stands

- Novelty re-checked 2026-10-09: not anticipated in CXL. Cite and separate Micron US12645581B2,
  Intel US11093398B2, US12455827, US20250021228A1, HPCA'13 (names timeouts as future work),
  Callback ISCA'15, I-structures, IBM CLT patent, Microsoft conditional reads, SSB, QOLB.
  The letter's claim is five protocol-safety rules for parking in CXL, not full/empty bits.
  Motivation: every CXL shared-memory system polls (Tigon, cMPI, HydraRPC, CXL-CCL, TraCT; TraCT
  lock acquisition 20 to 100 us from 2 to 16 nodes in its thesis).
- Model checked (tla/, TLA+ and TLC): all five rules give no Root Port timeout, no stale read, no
  blocked writer and liveness; removing any one breaks one property (R2 is implied by R1).
- gem5 v25.1 (gem5-model/, device ParkedMemory as an EXTRAS dir, x86 SE programs on up to 36
  hosts): parked waiters make 1.4 device reads per message instead of 39 to 750 and leave tenants
  over 99% of their bandwidth (polling: 77 to 79% with 256 waiters). Latency gain is modest: a parked
  line fill beats the best software scheme by 1.25x only for one-line messages; longer messages,
  RPC and barriers favor line polling or are unchanged, because x86 has no 64 B uncached load and
  write-back stores read for ownership (MOVDIR64B, not in gem5, would remove that). The older SimPy
  numbers (1.3 to 4x) assumed 64 B uncached loads and expensive flushes; do not quote them.
- gem5 found two host rules: one sync line per 4 KiB page (a writer's store must not merge behind
  its own parked prefetch) and cacheable parked reads must be non-consuming.
- CAL letter draft: paper/parked_loads.tex (4 pages, compiles with tectonic).

## Next steps

The four author items under "Remaining work for a CAL letter" in PROJECT.md.

## Working rules (from the author)

- The author is the sole author. Never run git commit or git push; leave changes in the working
  tree and give the author the exact git commands. Never add a Co-Authored-By line.
- An idea is only presented as viable after three gates: (1) novelty, an adversarial search of
  papers, arXiv and patents; (2) significance, a first-order model showing a gain the target venue
  would care about; (3) feasibility with evaluation methods that venue accepts. Never promise
  acceptance; say which gate fails.
- Re-run the novelty search before any submission; this area changes within weeks.
- Write plainly: no decorative banner comments, no long narrative comments, no em dashes.

## Environment

Single-socket i9-14900K, 32 GB DRAM, RTX 4500 Ada 24 GB (usually occupied by the author's
llama-server; do not stop it). No CXL hardware. Python analysis env: create a venv and
pip install -r requirements.txt; downloaded data lives in data/ (fetch_data.sh, not in git).

## History

This project came out of a two-day idea search on 2026-10-08 and 2026-10-09 (16+ ideas checked,
most rejected by prior work or patents). The full conversation is in chat/ (not in git); read it
for the reasoning behind every decision. PROJECT.md is the single document for the project: idea, gap, goal, motivation, related work, evidence and plan.
