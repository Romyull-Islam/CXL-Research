# Sizing CXL memory tiers for test-time-scaling LLM workloads

Status (2026-10-09): gate 2 failed for the preemption angle; recommendation is to stop this project
as a CXL paper (findings/02). Novelty scan: crowded for ISCA/MICRO/HPCA; the CXL-specific part (a
tier for preempted KV) is already shipped with host DRAM, and its measured value is small.

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
- Generic CXL KV work (no test-time focus): TraCT, SAC, CXL-SpecKV, Beluga, Photonic-CXL; Tang et
  al. (NeurIPS'24 ML for Systems, CXL-to-GPU bandwidth on par with host DRAM); Samsung CMM-D pool
  white paper (June 2026).
- Preempted or paused KV kept off-GPU (the angle tested in findings/02): vLLM 0.19+ CPU offloading
  restores preempted requests; vLLM-Ascend PreemptOffloadConnector; SGLang retraction backup;
  InferCept (ICML'24, discard/swap/preserve); CacheOPT (2503.13773); FastServe; EfficientAgent
  (2609.33762, sizes a host tier for paused agents' KV). Preemption models: "Service-Induced
  Congestion in Memory-Constrained LLM Serving" (2606.15555); measured preemption in reasoning RL
  rollouts: RollPacker (2509.21009).
- Simulators: Vidur (MLSys'24, measured kernel profiles up to 4K tokens, recompute only);
  LLMServingSim 2.0 (2602.23036, vLLM V1 scheduler port with CPU/CXL tiers).

## Evidence so far

- MathArena AIME 2025 (7,915 samples): within best-of-4 on one problem the longest sample holds 36%
  of byte-seconds (uniform 25%); across a mixed batch the longest 10% hold 53%
  (findings/01-tts-kv-skew.md).
- HuggingFace H4, N up to 2048: within a best-of-N group the longest 10% hold 23 to 27% of
  byte-seconds and the group spends 44 to 60% of its life with at most 10% of samples alive; those
  stragglers are decoding, not paused. Beam search final samples share prefixes 3.0 to 3.6 times,
  DVTS about 1.9 times (findings/03-h4-large-n.md).
- Preempted KV in a second tier, with a vLLM-V1-faithful simulator calibrated on measured H100
  kernels, cross-checked against LLMServingSim 2.0 and a real vLLM run, on the Qwen-Bailian thinking
  trace and MathArena best-of-N (findings/02-preempted-kv-tier.md):
  - throughput gain 2 to 10% at saturation (median 5.5%), 0 to 8% over tuned admission, 0 below
    saturation, no goodput gain at p99 TTFT of 10 or 60 s;
  - first-order model: gain is about R / rho, with R (recomputed per generated token) below 0.9
    and rho (decode over recompute cost per token) 4 to 30, because prefill is compute-bound and
    decode bandwidth-bound;
  - capacity: 0.1 to 0.25x HBM KV for production arrivals, 1 to 4x only for offline best-of-N
    bursts; inside a DGX node's 2 TB DRAM except offline best-of-64 on 8x H200/B200 with vLLM's
    LRU tier; pooling across 8 nodes saves 49 to 64% of a capacity that is not needed.

## Plan

1. Large-N analysis with the HuggingFace H4 best-of-N, beam-search and DVTS completions (N up to
   2048). Done: findings/03.
2. Preemption sizing model. Done: analysis/kvsim.py and the gate2_* sweeps, findings/02.
3. Significance gate on that model. Done: fails (gain 2 to 10%, capacity fits in host DRAM).
4. Conference version (placement mechanism plus vLLM/SGLang validation): not pursued, since gate 2
   failed. Instead, a short letter reports the negative result with its first-order model and
   validation (paper/preempted_kv.tex); it needs the A100 run (README) before submission. The
   only regime with a CXL capacity case is offline best-of-N on large-HBM nodes with an FP8 KV
   cache, worth 5 to 8% over tuned admission.

## Data and how to run

MathArena AIME 2025 outputs, HuggingFace H4 completions (Llama-3.2 1B/3B), the Qwen-Bailian thinking
trace, public model configs and Vidur's H100 kernel profiles. `./fetch_data.sh` downloads them into
data/ (not in git); then create a venv, `pip install -r requirements.txt`, and run the scripts in
analysis/ (see README.md). Results are in results/.
