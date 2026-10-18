# Test-time-scaling KV memory skew (MathArena AIME 2025)

Data: MathArena aime_2025_outputs (7,915 samples, about 30 reasoning models, 30 problems x 4
samples each; per-sample input and output tokens). Script: analysis/tts_skew.py plus the batch
check below. KV byte-seconds of a sample taken as in*out + out^2/2 (linear KV growth, equal decode
rate).

## Within one problem (best-of-4 parallel sampling)

Median over models: the longest sample holds 36% of the group's KV byte-seconds (uniform 25%), is
1.23x the median sample's length, and runs alone for 12% of the group's time. Peak KV is about 1.9x
the mean. Weak skew: samples of the same problem have similar lengths.

## Across a mixed batch (all samples of one model)

Median over models: the longest 10% of samples hold 53% of KV byte-seconds and the longest 20% hold
75%; p99 output length is 4.5x the median (up to 17.8x). Median output lengths are 12K to 29K tokens
with maxima to 300K.

## Reading

- The skew that matters is across requests (heavy-tailed reasoning lengths), not within best-of-N.
- Age identifies the tail for free: samples still running late are the long ones.
- But a decoding sample reads its whole KV every step, so moving tail KV to CXL slows those samples
  unless attention is sparse. OasisKV (arXiv 2608.08097) and "Not All Thoughts Need HBM"
  (2605.09490) already tier long reasoning KV this way, so a mechanism here sits close to them.
- Still to test: larger N with the HuggingFace H4 best-of-N, beam and DVTS completions (N up to
  2048), and whether a tail-aware HBM/CXL capacity split beats sizing HBM for the median.
