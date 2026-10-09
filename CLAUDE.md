# Project context for Claude

Parked loads: full/empty synchronization for multi-host CXL memory. Read PROJECT.md first (everything is
in it), then model/README.md.

## Where it stands

- Novelty re-checked 2026-10-09: not anticipated in CXL. Cite and separate Micron US12645581B2,
  Intel US11093398B2, US12455827, US20250021228A1, HPCA'13 (names timeouts as future work),
  Callback ISCA'15, I-structures, IBM CLT patent, Microsoft conditional reads, SSB, QOLB.
  The letter's claim is four protocol-safety rules for parking in CXL, not full/empty bits.
- Calibrated SimPy model (model/parksim.py): 1.3 to 4x lower handoff, RPC and barrier latency;
  polling waiters take up to 44% of a switched device's bandwidth, parked ones about 8%.
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
