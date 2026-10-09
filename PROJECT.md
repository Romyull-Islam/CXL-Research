# Sizing CXL memory tiers for test-time-scaling LLM workloads

Status (2026-10-09): early. Novelty scan: crowded for ISCA/MICRO/HPCA without a mechanism and
real-system validation; partly open for a top journal (IEEE TC, IEEE TPDS, ACM TACO) or an IEEE
Computer Architecture Letters paper.

## Idea

Reasoning and test-time-scaling workloads (best-of-N sampling, beam and tree search, long chains of
thought) have heavily skewed KV-cache memory: the longest 10 to 20% of samples hold most of the
memory over time. Use real per-sample length data to decide how much HBM versus CXL capacity a node
needs and which KV belongs on CXL, focusing on KV that is not read every step (paused or preempted
requests), where CXL does not slow decoding.

## Problem and gap

Reasoning models and test-time scaling produce very long and very uneven decodes: median outputs of
12K to 29K tokens with tails to 300K. KV-cache memory, not compute, becomes the limit. Recent work
shows this memory wall, offloads beam-search KV to host memory, manages memory for beam and DVTS
search, and tiers long reasoning KV between HBM and DDR with sparse attention (see Related work).

What no paper does yet: use the real distribution of per-sample lengths across test-time strategies
to size and place CXL memory tiers. How much HBM versus CXL capacity a node needs, when a pooled CXL
tier beats per-GPU memory, and how often requests fall off a capacity cliff at a given latency
target.

## Goal

1. A trace-to-memory model that turns real per-sample token counts and search trees into KV
   byte-seconds, sharing and peak concurrency per strategy, for GQA and MLA models.
2. CXL provisioning rules from that model: HBM to CXL capacity ratio, pooled versus per-GPU CXL,
   capacity-cliff frequency at a latency target.
3. Quantify the case where CXL helps without slowing decoding: holding the KV of paused or
   preempted requests instead of recomputing it.

## Motivation

- Test-time scaling is the dominant 2025 to 2026 inference trend, and its memory demand grows with
  reasoning effort and the number of samples.
- Measured skew (MathArena AIME 2025, about 30 reasoning models): across a mixed batch, the longest
  10% of samples hold 53% of KV byte-seconds and the longest 20% hold 75%; p99 output length is 4.5x
  the median. Sizing HBM for the tail wastes it most of the time; sizing it for the median causes
  preemption.
- A decoding sample reads its whole KV every step, so moving active tail KV to CXL slows it unless
  attention is sparse (that path is taken by OasisKV). KV of paused or preempted requests is not
  read every step, so CXL capacity can hold it instead of recomputing, and the heavy tail decides
  how often preemption happens. This is the open, CXL-friendly case.

## Related work

- Memory wall of test-time scaling: Kinetics (NeurIPS'25); "Understanding Inference Scaling for
  LLMs" (ISCA'26 industry, arXiv 2605.19775); "The Cost of Dynamic Reasoning" (HPCA'26, 2506.04301).
- Offloading and memory management for search: Locality-Aware Beam Scheduling (MLSys'26, host
  offload of beam KV); FastTTS (ASPLOS'26, 2509.00195, memory for beam/DVTS with a reward model);
  ArborKV (2605.22106, tree-of-thought KV eviction); PASCAL (HPCA'26, phase-aware reasoning
  scheduling).
- Tiering long reasoning KV: OasisKV (2608.08097, sparse-attention staging from host memory); "Not
  All Thoughts Need HBM" (2605.09490, four-tier placement for one long chain of thought).
- KV compression for reasoning (saturated): R-KV (NeurIPS'25), ThinKV (ICLR'26), GrowPage, KARA.
- Generic CXL KV work (no test-time focus): TraCT, SAC, CXL-SpecKV, Beluga, Photonic-CXL.

## Evidence so far

From MathArena AIME 2025 (7,915 samples): within best-of-4 on one problem, the longest sample holds
36% of byte-seconds (uniform would be 25%) and runs alone 12% of the time, so within-problem skew is
mild. Across a mixed batch the skew is strong (53% held by the longest 10%). Details:
findings/01-tts-kv-skew.md.

## Plan

1. Large-N analysis with the HuggingFace H4 best-of-N, beam-search and DVTS completions (N up to
   2048).
2. Preemption sizing model: given HBM capacity and the length distribution, how often requests are
   preempted, and how much CXL capacity removes recomputation, for GQA and MLA models.
3. Significance gate on that model; stop or pivot if the gain is small.
4. For a conference version: a placement mechanism plus validation on vLLM or SGLang with CPU
   offload or CXL emulation.

## Data and how to run

MathArena AIME 2025 outputs, HuggingFace H4 completions (Llama-3.2 1B/3B), Monkey Business, the
Qwen-Bailian thinking trace. `./fetch_data.sh` downloads into data/ (not in git); then
`pip install -r requirements.txt` and `python3 analysis/tts_skew.py`.
