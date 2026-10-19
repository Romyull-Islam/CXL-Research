# Keeping preempted KV in a CXL tier: gate 2 fails

Note (2026-10-10): paper/preempted_kv.tex carries the audited numbers; where this note differs, the paper
is correct (for example, 30 to 99% of requests preempted, 4 to 9 preemptions per preempted request on
average in best-of-N, and the online best-of-16 gain of up to 6.4% at 0.85 to 0.95 load).

Question: under KV-cache pressure vLLM preempts requests and later recomputes their KV. A second
tier (CXL memory, or plain host DRAM) could hold that KV instead, without slowing decode because
paused KV is not read every step. How much does that save for reasoning and test-time-scaling
workloads, how much capacity does it need, and does it need CXL rather than host DRAM?

Answer: the tier is worth 2 to 10% throughput at saturation (median 5.5%), 0 to 8% over a
recompute baseline whose admission limit is tuned, and nothing in goodput at latency targets
operators use. The capacity it needs fits in a DGX node's DRAM except for offline best-of-N batches
submitted all at once. Pooling saves half the tier capacity, but that capacity is not needed.
On a real vLLM 0.31 run the shipped CPU offloading tier was 3.9% slower than recompute. Gate 2
fails.

## Method

- Simulator: analysis/kvsim.py follows vLLM V1's FCFS scheduler (vllm commit 19c0bed): the last
  running request is preempted, its blocks freed and its KV recomputed on resume; freed blocks stay
  in the GPU prefix cache until reused; a waiting request is admitted only if its whole sequence
  fits. Tier policies: swap on preemption (vLLM V0 swap mode) and LRU write-through of every full
  block (the vLLM OffloadingConnector CPU tier). Loads run in the background as in the connector.
- Step time: analysis/models.py, a roofline whose efficiencies are fitted to Vidur's measured H100
  kernels (analysis/calibrate.py). Whole-layer error: median 4 to 10%, bias within 5%, on the fitted
  Llama-2-70B and on held-out CodeLlama-34B. Profiles stop at 4K tokens, so longer contexts are
  extrapolated physically (decode attention linear in KV bytes, prefill attention in query-key
  pairs). Prefill attention has two settings: FA3 (0.6 of peak, current vLLM on Hopper) and the
  measured FA2-era kernels (0.14 of peak), which make recompute dearer and favor the tier.
- Validation:
  1. analysis/test_kvsim.py: 11 tests (conservation, fast-forward equals step-by-step, a
     hand-worked preemption).
  2. LLMServingSim 2.0, an independent port of the vLLM V1 scheduler with its own GPU profiles
     (analysis/xcheck_llmservingsim.py, 64 requests with 10K to 30K-token outputs): preemptions 50
     vs 50, recomputed tokens 565,641 vs 567,365 (0.3%) with matching prefix-cache settings,
     per-request finish times correlate at 0.9999 with a 14% timing offset. Tier saving: 1.1% in
     kvsim, 2.7% in LLMServingSim.
  3. Real vLLM 0.31 on the local RTX 4500 Ada (analysis/vllm_validate.py,
     results/vllm_validate.json): Qwen2.5-0.5B with the KV cache fixed at 4,096 blocks, 300
     Bailian-shaped requests submitted at once. With recompute, vLLM preempted 297 times and
     recomputed 111,479 tokens; kvsim predicts 277 and 117,656 (-7% and +6%). vLLM's native CPU
     offloading tier (4 GiB) stored about a third of the KV; from the saved counters it cut recomputation by
     34% (73K against 111K tokens; the earlier '50K reloaded, 41%' has no saved source),
     recomputed the other 73K, and lowered throughput by 3.9% (5,090 vs 5,299 tokens/s). Removing
     the pressure altogether (15,000 blocks) raised throughput by only 5.9%. kvsim's LRU tier
     reloads nearly everything, so it is an upper bound on what the shipped tier achieves.
  4. Real vLLM with a reasoning model (analysis/a100_validate.py run on the local GPU,
     results/a100/run_DeepSeek-R1-Distill-Qwen-1.5B_tp1_bailian.json): DeepSeek-R1-Distill-Qwen-1.5B
     on the RTX 4500 Ada with about 2 GB of KV, 400 Bailian requests: 275 preemptions and 569K
     recomputed tokens (R = 0.43) against 270 and 574K from kvsim. The offloading tier removed 43% of
     the recomputation; its throughput is not reported because another process started on the GPU
     during that run. Measured rho (micro_*.json): 23, 77 and 128 at 2K, 8K and 16K tokens of
     context (batches of 17, 4 and 2 in the small KV budget).
- Workloads: the Qwen-Bailian thinking trace (10,812 production reasoning requests over 2 hours,
  real arrivals; output p50 1.7K, p99 35K tokens), and MathArena AIME 2025 per-sample lengths of
  the model with the same architecture (R1-Distill-Llama-70B, QwQ-32B, Qwen3-235B, DeepSeek-R1-0528)
  served as best-of-1, 16 and 64 groups.
- Deployments (vLLM defaults: block 16, max_num_seqs 1024, 8192-token budget, 0.9 memory use):
  Llama-70B BF16 on 8x H100, H200 and B200; QwQ-32B on 2x and 4x H100; Qwen3-235B FP8 on 8x
  H100; DeepSeek-R1 FP8 on 8x H200. MLA keeps the whole latent on every TP rank, so R1 holds only
  0.56M tokens (39 GB of unique KV).

## Results

Saturation, all requests at once (results/gate2_saturation.csv). Throughput gain of an unlimited
swap tier over recompute, FA3 calibration (FA2 in brackets):

| Deployment | HBM KV (GB) | Bailian | Best-of-1 | Best-of-16 | Best-of-64 | Requests preempted |
|---|---|---|---|---|---|---|
| 70B 8x H100 | 403 | 6.2% (7.2) | 5.5% (6.7) | 5.6% (7.3) | 6.1% (8.0) | 38 to 92% |
| 70B 8x H200 | 842 | 8.1% (9.3) | 9.3% (11.1) | 9.2% (11.8) | 10.0% (13.3) | 35 to 84% |
| 70B 8x B200 | 1,123 | 7.3% (8.1) | 9.4% (10.8) | 8.8% (11.0) | 9.8% (12.3) | 30 to 79% |
| QwQ-32B 2x H100 | 70 | 2.2% (2.9) | 4.6% (5.2) | 5.1% (6.0) | 4.0% (4.8) | 41 to 99% |
| QwQ-32B 4x H100 | 206 | 3.3% (4.2) | 7.0% (7.8) | 7.2% (8.2) | 5.7% (6.7) | 40 to 96% |
| Qwen3-235B FP8 8x H100 | 154 | 2.1% (3.2) | 4.6% (5.6) | 5.3% (6.4) | 4.3% (5.7) | 40 to 96% |
| R1 FP8 8x H200 | 39 | 1.7% (2.8) | 2.9% (3.9) | 2.8% (4.3) | 2.7% (4.6) | 42 to 98% |

Why it is small (analysis/first_order.py). With R recomputed tokens per generated token and rho
the GPU time of a decoded token over that of a recomputed token, the recompute share is
R / (rho + R). This matches every simulated case (correlation 0.997, mean error 0.3 points).
Prefill is compute-bound while decode with a full KV cache is bound by HBM bandwidth, so rho is 4
to 30. FCFS preemption puts the victim first in line, so R stays below 0.9 even when requests are
preempted up to 9 times each. The gain is therefore at most about R / rho: about 20% at the
extremes found here (R 0.89, rho 4.4), 2 to 10% in the cases simulated.

Admission tuning (results/gate2_maxseqs.csv, gate2_fp8_tuned.csv). Lowering max_num_seqs from
1024 to 96 to 192 recovers 0 to 6% by itself. Over the best-tuned recompute baseline the
unlimited tier adds 1.7 to 5.9% (BF16 KV) and 5 to 8% (FP8 KV, 70B best-of-N, the most favorable
case found).

Sensitivity (results/gate2_sensitivity.csv). An FP8 KV cache halves the KV read per decode step,
lowers rho and raises the gain over default recompute to 11 to 15% for 70B best-of-N (14 to 20%
with FA2-era prefill). A 3 ms step overhead or a 25 GB/s tier link changes the gain by under one
point.

Online load (results/gate2_online.csv, gate2_load.csv, gate2_goodput_summary.csv). Below
saturation the throughput gain is 0. The tier changes p99 time to first token by +4 to -49%, and
materially only where it is already 90 to 17,000 s. Goodput, the highest load meeting p99 TTFT <= 10 s or <= 60 s, is
unchanged in 19 of 20 cases (70B 8x H200 Bailian: 0.75 to 0.80 of saturation). At loads that meet
the SLO, under 1% of Bailian requests are preempted.

Capacity (results/gate2_capacity.csv). Share of the unlimited-tier gain kept:

| Workload | Swap on preemption | LRU write-through (vLLM) |
|---|---|---|
| Bailian, all submitted at once (offline) | 93 to 96% at 0.1x HBM KV, 98 to 100% at 0.25x | 97 to 103% at 0.25x |
| Best-of-16/64, all at once | 65 to 93% at 1x, 87 to 106% at 2x | 54 to 94% at 2x, 87 to 110% at 4x |

In absolute terms the swap tier peaks at 19 to 214 GB for Bailian and 0.1 to 2.2 TB for offline
best-of-N; online it peaks at 0 to 62 GB (Bailian) and 38 to 797 GB (best-of-16). A DGX H100,
H200 or B200 has 2 TB of DRAM (up to 4 TB on B200). Only offline best-of-64 on 8x H200/B200 with
vLLM's LRU tier needs more, to buy a gain of at most 10% (5 to 8% over tuned admission).

Pooling (results/gate2_pool_summary.csv). Eight instances behind a round-robin balancer: one
shared tier needs 36 to 51% of the sum of per-instance peaks (a 49 to 64% saving). The pooled peak
is at most 1.26 TB across eight nodes that together have 16 TB of DRAM.

## Gates

- Gate 1, novelty: the mechanism ships. vLLM 0.19+ restores preempted requests from its CPU
  offloading tier; vLLM-Ascend's PreemptOffloadConnector and SGLang's retraction backup do the
  same; swap versus recompute is covered by InferCept (ICML'24), CacheOPT (2503.13773) and
  FastServe. EfficientAgent (2609.33762) sizes a host tier for paused requests from a working-set
  model. Left open: sizing from test-time-scaling length distributions and pooled CXL.
- Gate 2, significance: fails. Gain 2 to 10% at saturation, 0 to 8% over tuned admission, no
  goodput gain at usable latency, capacity inside host DRAM.
- Gate 3, feasibility: met (calibrated simulator, independent simulator cross-check, real vLLM
  run), but it does not rescue gate 2.

## What is left

- The only regime with a capacity case for CXL is offline batches (RL rollouts, evaluation with
  best-of-N) on large-HBM nodes with an FP8 KV cache: 11 to 15% over default recompute, 5 to 8%
  over tuned admission, with a tier of 1x HBM KV (swap) to 4x (vLLM's LRU write-through, up to
  4.5 TB on 8x B200). A store-on-preemption policy needs about half the capacity of LRU
  write-through for the same gain; that is a policy note, not a CXL result.
- Grace-based racks have less host memory than HBM (GB300 NVL72: 17 TB LPDDR5X for 20 TB HBM), but
  whether Grace or Vera can attach CXL memory was not verified.

Reproduce: python analysis/gate2_saturation.py, then gate2_capacity, gate2_maxseqs, gate2_online,
gate2_goodput, gate2_pool, gate2_sensitivity, gate2_fp8_tuned and first_order.py (16 cores, about
an hour in total; each sweep keeps 12 workers at about 250 MB each).

## Real A100 (2026-10-18)

DeepSeek-R1-Distill-Llama-8B on one A100-SXM4-80GB, vLLM 0.31, 400 Bailian requests at once, GPU KV
28,049 blocks (449K tokens), native CPU tier 32 GB (results/a100/run_DeepSeek-R1-Distill-Llama-8B_tp1_bailian.json).
Recompute: 808 and 806 tok/s, 225 preemptions, R = 0.34. CPU tier: 806 and 809 tok/s, R = 0.12 and
0.10 (66 to 71% of recomputed tokens removed). Gain -0.3% and +0.3%; the bound R/rho with rho 5.7 to
28 (microbenchmark at 2K to 16K context) is 1.2 to 6%. A first rep 1 (-8.7%) overlapped another
user's jobs on GPU 0 and was rerun on GPU 1 (analysis/rerun_rep.sh).
