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
- The error is in c_r: recompute in these runs shares steps with decoding. A mixed microbenchmark
  (16 extra 4,096-token prefills added to 7 decoding requests; reuse_validate.py mixbench) gives c_r
  3 to 5% lower on the dense models and 12 to 14% lower on the MoE model; with it the error is
  -2 to +8% in 23 of 24 cases and +15% in one (MoE, synthetic, 64-token answers).
