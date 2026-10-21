# Project context for Claude

Sizing CXL memory tiers for test-time-scaling LLM workloads. Read PROJECT.md first (everything is in
it), then
findings/01-tts-kv-skew.md.

## Where it stands

- Novelty scan: crowded for ISCA/MICRO/HPCA; keeping preempted KV off-GPU already ships with host
  DRAM (vLLM 0.19+ CPU offload, vLLM-Ascend, SGLang).
- Data: MathArena (53% of KV byte-seconds in the longest 10% of a mixed batch), H4 large-N
  (findings/03), Qwen-Bailian thinking trace.
- Gate 2 on the preemption sizing model FAILED (2026-10-09, findings/02): a tier for preempted KV
  gains 2 to 10% throughput at saturation, 0 to 8% over tuned admission, no goodput at usable
  latency; capacity fits in host DRAM except offline best-of-N bursts. Reason: gain ~ R / rho with
  R < 0.9 and rho 4 to 30.

## Next step

Recommendation: not a CXL architecture paper; an IEEE letter (paper/preempted_kv.tex, 4 pages, "When
Does a KV Memory Tier Pay? A First-Order Model for LLM Serving"). Model: gain = R (c_r - c_l) / (c_d + R c_l). Tested on real vLLM (findings/04): eight models of four
architectures (Qwen2 1.5B/7B/14B, QwQ-32B on two GPUs, Llama 1B/8B, Granite MoE, Qwen3-30B-A3B MoE) on
an RTX 4500 Ada and A100s; gains 6 to 372%; with c_l from the link bandwidth the model errs by -1 to
+12% (dense) and +12 to +33% (MoE). The error is in the load term (the tier costs 1.7 to 5.6x bytes/bandwidth); with
c_l measured in the 4-token run, -8 to +4% (dense) and 0 to +18% (MoE; the +18% is within
run-to-run variation). Agent sessions (24 real SWE-agent traces, Llama 8B, A100): +11.4% measured,
+10.9% predicted. Applied to preempted KV: 2 to
10% simulated; real vLLM -4 to +2% (0.5B on the RTX 4500; on A100s 8B: -0.3 and +0.3%, QwQ-32B on two: +0.1 and +1.9%).
If revisited, the only open regime is offline best-of-N (RL rollouts, evaluation) on large-HBM
nodes with FP8 KV, where vLLM's LRU offload tier would need 2 to 4x HBM KV for a 5 to 8% gain.

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
for the reasoning behind every decision. PROJECT.md is the single document for the project: idea,
gap, goal, motivation, related work, evidence and plan.
