# KV memory of large-N test-time scaling (HuggingFace H4, N up to 2048)

Data: HuggingFace H4 best-of-N, beam-search and DVTS completions of Llama-3.2 1B and 3B on the 500
MATH-500 problems (search-and-learn settings: T=0.8, 2048 tokens per sample or per step, beam
width 4, 40 iterations). Token counts: analysis/h4_tokens.py (Llama-3.2 tokenizer, checked by hand
and against vLLM's stored counts; weighted best-of-N accuracy reproduces the stored evals within
3 points). Memory metrics: analysis/h4_memory.py, results/h4_memory.csv.

Best-of-N model: the N samples of a problem start together, decode one token per step and share
the prompt KV. Medians over problems:

| Model | N | Median output | Longest 10% share of byte-seconds | Peak / mean KV | Lifetime with <= 10% alive |
|---|---|---|---|---|---|
| 1B | 256 | 420 | 0.27 | 2.7 | 0.48 |
| 1B | 1024 | 419 | 0.27 | 3.1 | 0.56 |
| 1B | 2048 | 418 | 0.27 | 3.4 | 0.60 |
| 3B | 256 | 380 | 0.23 | 2.7 | 0.44 |
| 3B | 2048 | 381 | 0.23 | 3.3 | 0.56 |

Across all best-of-N samples of a run (a mixed batch), the longest 10% hold 48 to 49% of KV
byte-seconds, close to the 53% measured on MathArena reasoning models (finding 01).

Beam search and DVTS store only their final samples. Their prefix tries hold 3.0 to 3.6 times
fewer tokens than the samples summed (beam search, N >= 64) and about 1.9 times fewer (DVTS); the
live search tree, with pruned branches, shares at least this much.

Reading:
- Larger N mostly lengthens the straggler tail: a best-of-2048 group spends 60% of its life with
  at most 10% of its samples still running. Those stragglers are decoding, so their KV is read
  every step and gains nothing from a slow tier; the lever is stopping them early (known:
  early-stopping self-consistency, confidence-based trace pruning), not where their KV lives.
- Within-problem skew is moderate (27% for the top 10%) and the same at N=256 and N=2048.
- Caveats: 1B/3B outputs are short and capped at 2048 tokens (4% of 1B samples hit the cap), so
  tails are truncated; pruned beam and DVTS branches are not in the data.
