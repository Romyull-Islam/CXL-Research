# Sizing CXL memory tiers for test-time-scaling LLM workloads

Status (2026-10-09): early. Novelty scan says partly open for a top journal (IEEE TC, TPDS, ACM
TACO) or an IEEE CAL letter; crowded for ISCA/MICRO/HPCA without a mechanism and real-system
validation. First data analysis done (findings/01-tts-kv-skew.md).

## Problem and gap

Reasoning models and test-time scaling (best-of-N sampling, beam search, tree search with process
reward models, reasoning-effort settings) produce very long and very uneven decodes: median outputs
of 12K to 29K tokens with tails to 300K. KV-cache memory, not compute, becomes the limit. Recent
work shows the memory wall (Kinetics, NeurIPS'25; "Understanding Inference Scaling", ISCA'26
industry; "The Cost of Dynamic Reasoning", HPCA'26), offloads beam-search KV to host memory
(Locality-Aware Beam Scheduling, MLSys'26), manages memory for beam/DVTS search (FastTTS,
ASPLOS'26), and tiers long reasoning KV between HBM and DDR with sparse attention (OasisKV, arXiv
2608.08097; "Not All Thoughts Need HBM", 2605.09490).

What no paper does yet: use the real distribution of per-sample lengths across test-time
strategies to size and place CXL memory tiers: how much HBM versus CXL capacity a node needs, when
a pooled CXL tier beats per-GPU memory, and how often requests fall off a capacity cliff at a given
latency target.

## Goal

Build a trace-to-memory model that turns real per-sample token counts and search trees into KV
byte-seconds, sharing and peak concurrency per strategy, and derive CXL provisioning rules for GQA
and MLA models. Then find the one case where CXL helps without slowing decoding (see Motivation)
and quantify it.

## Motivation and first evidence

- Across a mixed batch, the longest 10% of samples hold 53% of KV byte-seconds and the longest 20%
  hold 75% (median over about 30 reasoning models on AIME 2025). Within best-of-4 on one problem the
  skew is mild (36% versus 25%).
- A decoding sample reads its whole KV every step, so moving tail KV to CXL slows it unless
  attention is sparse; that path is close to OasisKV.
- The open angle: KV of requests that are paused or preempted under memory pressure is not read
  every step. CXL capacity can hold it instead of recomputing, and the heavy tail decides how often
  preemption happens. This needs a sizing model.

## Related work

Kinetics (NeurIPS'25); Understanding Inference Scaling (ISCA'26 industry, arXiv 2605.19775); The
Cost of Dynamic Reasoning (HPCA'26, 2506.04301); Locality-Aware Beam Scheduling (MLSys'26);
FastTTS (ASPLOS'26, 2509.00195); OasisKV (2608.08097); Not All Thoughts Need HBM (2605.09490);
ArborKV (2605.22106); PASCAL (HPCA'26); R-KV, ThinKV (KV compression, saturated); generic CXL KV
work (TraCT, SAC, CXL-SpecKV, Beluga).

## Data

MathArena AIME 2025 outputs (per-sample tokens and correctness for about 30 models); HuggingFace H4
best-of-N, beam-search and DVTS completions (N up to 2048, Llama-3.2 1B/3B); Monkey Business
(10K samples per problem); Qwen-Bailian thinking trace. See fetch_data.sh.

## Next steps

1. Large-N analysis with the H4 completions.
2. Preemption sizing model: given HBM capacity and the length distribution, how often requests are
   preempted, and how much CXL capacity removes recomputation, for GQA and MLA models.
3. Gate 2 (significance) on that model before any further work; report honestly if it fails.
4. For a conference version: a mechanism plus validation on vLLM/SGLang with CPU offload or CXL
   emulation.
