# Prompt-block reuse in the Mooncake traces

Data: Mooncake FAST'25 traces (Kimi), one hour each, 512-token prompt blocks identified by hash.
Scripts: analysis/mooncake_basic.py, analysis/mooncake_reuse.py.

| Trace | Requests | Block refs | Reuse share with an infinite cache | Blocks used once |
|---|---|---|---|---|
| conversation | 12,031 | 288,500 | 0.37 | 0.76 |
| toolagent | 23,608 | 409,616 | 0.55 | 0.79 |
| synthetic | 3,993 | 121,877 | 0.64 | 0.58 |

Time between consecutive uses of a block:

| Trace | under 1 s | 1 to 10 s | 10 to 60 s | 1 to 10 min | over 10 min |
|---|---|---|---|---|---|
| conversation | 0.10 | 0.01 | 0.16 | 0.66 | 0.06 |
| toolagent | 0.52 | 0.09 | 0.07 | 0.30 | 0.01 |
| synthetic | 0.02 | 0.15 | 0.34 | 0.49 | 0.01 |

LRU capacity needed to capture 80% of the achievable hits (one hour of this load):

| Trace | Qwen2.5-7B | DeepSeek-V3 (MLA) | Llama-3.1-70B |
|---|---|---|---|
| conversation | 615 GB | 753 GB | 3.5 TB |
| toolagent | 247 GB | 303 GB | 1.4 TB |
| synthetic | 422 GB | 517 GB | 2.4 TB |

Reading:
- Most reuse happens one to ten minutes later and needs hundreds of GB to several TB, which is
  beyond HBM and host DRAM and is the regime CXL pools and hybrid devices target. This matches
  what Mooncake and later KV-tiering papers report, so it is context, not a new finding.
- Three quarters of blocks are never reused, yet LRU-style prefix caches admit every block. On a
  flash-backed tier those blocks cost writes and endurance for nothing. (Correction: arXiv
  2609.07175, cited here earlier as write-aware admission for LLM KV, is about generative
  recommendation models, not LLM KV caches.)
- Tool-agent traffic is bimodal: half of all reuses are concurrent requests sharing a prefix
  within the same second, the rest come minutes later.
