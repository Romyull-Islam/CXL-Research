# Project context for Claude

Endurance limits of LLM KV caching on hybrid DRAM-over-NAND CXL devices. Read PROJECT.md first (everything is in it), then
findings/04-endurance-wall.md (it ends with the novelty verdict).

## Where it stands

- Journal analysis done 2026-10-09 (findings/05): per-node scaling fixed, FTL with GC validated
  against Desnoyers' closed forms and MQSim, TLC/pSLC/SLC x BF16/FP8/INT4 provisioning, write amplification by
  KV death order, shared pool at Mooncake scale. Headline: admit-all never fits TLC; TLC fits only
  API-like traffic at partial duty or MLA with INT4; interactive traffic needs pSLC or SLC.
- Novelty narrow but present; target ACM TOS. New competitors to cite: HBFlex, Lachesis,
  2609.07175, 2609.39131, HBF Sucks?.
- Memory: endurance_sweep.py peaks near 1 GB per worker; use WORKERS=5 at most on this host.

## Next steps

Paper drafted (paper/endurance_tos.tex). Left: the author's items under "Remaining work for a journal
version" in PROJECT.md (affiliation, fresh novelty search, optional real-drive replay).

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
