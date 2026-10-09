# Project context for Claude

Sizing CXL memory tiers for test-time-scaling LLM workloads. Read PROJECT.md first (everything is in it), then
findings/01-tts-kv-skew.md.

## Where it stands

- Novelty scan: crowded for ISCA/MICRO/HPCA without a mechanism and real-system validation; partly
  open for a top journal (IEEE TC, TPDS, ACM TACO) or an IEEE CAL letter.
- Data so far (MathArena AIME 2025): across a mixed batch, the longest 10% of samples hold 53% of KV
  byte-seconds; within best-of-4 on one problem the skew is mild (36% vs 25%).
- Risk: tiering decoding KV to CXL needs sparse attention, which is close to OasisKV
  (arXiv 2608.08097) and "Not All Thoughts Need HBM" (2605.09490).

## Next step

Build the preemption sizing model: KV of paused or preempted requests is not read every step, so
CXL can hold it instead of recomputing. Use MathArena and the HuggingFace H4 best-of-N, beam and
DVTS completions (N up to 2048). Run gate 2 on it and report honestly if it fails.

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
