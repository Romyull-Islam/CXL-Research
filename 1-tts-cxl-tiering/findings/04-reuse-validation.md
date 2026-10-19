# Testing the first-order model where recomputation is large (real vLLM, 2026-10-10)

Script: analysis/reuse_validate.py; results: results/reuse/{runs.json,summary.csv,run.log}.

Model: a KV tier changes the GPU time per generated token from c_d + R c_r to c_d + R c_l, so
throughput rises by g = R (c_r - c_l) / (c_d + R c_l); g is about R/rho (rho = c_d/c_r) for small R
and approaches c_r/c_l - 1 for large R.

Setup: vLLM 0.31 with its native CPU offloading tier (4 GB), DeepSeek-R1-Distill-Qwen-1.5B on the
local RTX 4500 Ada (shared with the author's idle llama-server), GPU KV cache fixed at 2,048 blocks
(32K tokens), max_model_len 8,192. Document QA: 16 documents of 4,096 tokens, 8 questions each
(distinct 32-token suffixes), submitted at once in round-robin order so a document's KV leaves the
GPU before its next question; answers of 4, 64 and 512 tokens. Configurations: ideal (large GPU
cache, only checks the token accounting), recompute and cpu_offload at the same small cache.
Inputs not fitted to the runs: c_r = 32.9 us/token from the prefill microbenchmark on this GPU
(results/a100/micro_*.json, interpolated to 4,096 tokens); c_l = 1.2 us/token (28,672 B of KV per
token over a measured 24.0 GB/s pinned host-to-GPU copy). c_d comes from the offload run.

| answer tokens | R | rho | measured gain | predicted gain |
|---|---|---|---|---|
| 4 | 896 | 201 | 342% | 370% |
| 64 | 56 | 58 | 82% | 91% |
| 512 | 7 | 49 | 13% | 14% |

Three repetitions each, spread of run time at most 1.1%. The model over-predicts by 6 to 11%.

A first analysis used the ideal run's time as T0; that is wrong, because with a small GPU cache
fewer requests run at once and decode is slower per token (the 64-token case spends about 12 s in
small-batch decode). The comparison must be at equal cache size, as above.

## Four models, three architectures, two benchmarks (2026-10-11)

Same setup on DeepSeek-R1-Distill-Qwen 1.5B and 7B (Qwen2), Llama-3.2-1B and Granite-3.1-1B-A400M
(MoE, 8 of 32 experts), on synthetic 4,096-token documents and on 16 LongBench documents (2,048 to
6,144 tokens, five tasks, real questions). results/reuse/summary.csv; runner
analysis/run_reuse_models.sh. Gemma-3-1B did not start: its sliding-window layers need more than the
fixed 2,048-block GPU cache.

- Measured gains 6 to 372% at R = 7 to 1,027; spread at most 2.2% (17% for 4-token answers).
- With c_r from the standalone prefill microbenchmark the model over-predicts in all 24 cases:
  +4 to +12% on the dense models, +17 to +33% on the MoE model.
- (Superseded 2026-10-18, see the A100 section: the error is in c_l, not c_r.) First explanation:
  recompute in these runs shares steps with decoding. A mixed microbenchmark
  (16 extra 4,096-token prefills added to 7 decoding requests; reuse_validate.py mixbench) gives c_r
  3 to 5% lower on the dense models and 12 to 14% lower on the MoE model; with it the error is
  -2 to +8% in 23 of 24 cases and +15% in one (MoE, synthetic, 64-token answers).

## A100, three more models, and where the error lies (2026-10-18)

A100-SXM4-80GB on dgxa100 (GPU 1 for the reuse runs), same setup and cache size (2,048 blocks):
DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-14B and Qwen3-30B-A3B (MoE, 8 of 128
experts). Runner analysis/run_a100.sh; results/reuse_a100/ (report: REUSE_DIR=results/reuse_a100
python analysis/reuse_validate.py report). Pinned host-to-GPU bandwidth 21.1 to 26.2 GB/s.

- Measured gains 10 to 321%; spread at most 2.6% (14% for 4-token answers). With c_l from the
  bandwidth the model over-predicts in all 18 cases: +1 to +10% dense, +12 to +31% MoE.
- One LongBench rep of Qwen3 (512-token answers) ran 1.3 to 1.8 times slower in all three
  configurations while another job shared the node; it was rerun (analysis/rerun_rep.sh) and matched
  rep 0 within 0.3%. The disturbed values are in runs.json.bak on the server only.
- The mixed microbenchmark raises c_r for Qwen3 MoE by 6 to 8% (52.1 vs 48.4 to 49.0 us), making the
  error worse (+21 to +42%). So the mixed c_r does not explain the gap.
- Decomposition at 4-token answers, where decoding is negligible and the recompute, offload and ideal
  runs compute the same tokens: (T_recompute - T_ideal) / X is within -7 to +2% of the
  microbenchmark c_r on all seven models, while (T_offload - T_ideal) / X is 1.7 to 5.6 times the
  bandwidth c_l (A100 1.7 to 2.2, RTX 4500 1.8 to 5.6). The error is in the load term.
- With c_l measured that way once per model and benchmark (the 4-token run) and applied to the 64-
  and 512-token runs, the error is -5 to +4% on the five dense models and 0 to +18% on the two MoE
  models (summary.csv column error_insitu).
- Microbenchmark at full cache (prefill vs decode): rho 6 to 33 for 8B and 14B, 13 to 42 for Qwen3;
  prefill reaches 82 to 83% of A100 peak BF16 linear-layer FLOP/s at 2K tokens, 61 to 62% at 16K.

## QwQ-32B over two A100s (2026-10-19)

Tensor parallel over GPUs 0 and 1 (analysis/run_a100.sh 0,1 Qwen/QwQ-32B --preemption). Host-to-GPU
bandwidth measured with both GPUs copying at once: 25.9 GB/s, the same as one GPU (the two share a
host link), so c_l = 262,144 B / 25.9 GB/s = 10.1 us. Microbenchmark: c_r 150 to 183 us, rho 4.0 (2K),
11.7 (8K), 18.5 (16K).
- Gains 22 to 332%. With c_l from the bandwidth the error is -0.7 to +3.2% (one of 48 cases
  under-predicts); with c_l from the 4-token run, -8 to -4%: the recompute run costs 4% more per token
  than the microbenchmark (cross-GPU communication), and loading takes 1.9 to 2.1 times bytes/bandwidth.
- Across all eight models: c_l in place gives -8 to +4% (dense) and 0 to +18% (MoE).
