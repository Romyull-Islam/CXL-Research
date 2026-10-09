# Project context for Claude

Endurance limits of LLM KV caching on hybrid DRAM-over-NAND CXL devices. Read PROJECT.md first (everything is in it), then
findings/04-endurance-wall.md (it ends with the novelty verdict).

## Where it stands

- Core analysis done on Mooncake and Qwen-Bailian traces (analysis/hybrid_device_replay.py,
  analysis/per_node_endurance.py).
- Not novel for a top conference: "HBF Sucks?" (arXiv 2608.11668) reports the flash write excess on
  the same Bailian traces. Viable as a top-journal paper (IEEE TC, ACM TACO, ACM TOS) that adds the
  CXL device model, the oracle-admission bound, the interactive vs API split and provisioning rules.
- Counter-evidence to address: SPLASH (2609.23816) and HiFC (NeurIPS'25) get multi-year lifetimes
  with SLC or pseudo-SLC; Mooncake FAST'25 shows a shared pool beating prefix-routed local caches.

## Next steps

The five items under "Remaining work for a journal version" in PROJECT.md, starting with the sweep over
TLC, pseudo-SLC and SLC and over BF16, FP8 and INT4 KV, with budgets in TB/day.

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
