# Public data for memory-system research (checked 2026-10-09)

Locally cached Hugging Face chat datasets are mostly README stubs (WildChat, LMSYS-Chat-1M,
Chatbot Arena are gated); ShareGPT, zai-org/LongBench, GSM8K and AIME are really present.

## LLM serving traces

| Name | Where | Size | Key fields | Notes |
|---|---|---|---|---|
| Mooncake FAST'25 | github.com/kvcache-ai/Mooncake, FAST25-release/traces | 3 to 4 MB each | timestamp (ms), lengths, 512-token block hashes | downloaded to data/mooncake |
| Qwen-Bailian | github.com/alibaba-edu/qwen-bailian-usagetraces-anon (LFS) | 28 to 132 MB per trace | chat_id, parent_chat_id, turn, timestamp, lengths, type, 16-token block hashes | already analyzed by KVCache-in-the-Wild (ATC'25) and KVSET (arXiv 2609.27746) |
| ServeGen | github.com/alibaba/ServeGen | 915 MB | per-client distributions; conversations with per-turn timestamps | NSDI'26, SOSP'26 |
| Azure LLM 2023, 2024 | github.com/Azure/AzurePublicDataset | 1 MB; 1.8 GB | timestamp, context and generated tokens | |
| Azure multimodal 2025 | AzurePublicDataset | 8.5 MB gz | adds image counts | |
| BurstGPT v2.0 | github.com/HPMLL/BurstGPT | 430 MB | session ID, timestamps, tokens | |
| Chutes, one year | huggingface.co/datasets/harvardMadsys/chutes_metadata | 94 GB | 6.1B requests, 9,174 models, cached tokens for the last ten weeks | arXiv 2608.13573 |
| FineServe | github.com/hihiztc1/FineServe | up to 727 MB per day | 55 models, timestamps | license unclear |

## Agentic traces

| Name | Where | Size | Key fields |
|---|---|---|---|
| GitHub Copilot coding agent (June 2026) | AzurePublicDataset release ghcp-coding-agent-2026 | 3.2 GB | sessions, turns, LLM calls with cached tokens, tool batches with durations; no prefix hashes |
| FreeInference agentic (May to Sep 2026) | huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace | 0.44 GB compressed, 8.4 GB parquet | 16-token block ids, append vs rewrite mutations, tool calls with latency, nested subagents |
| TraceLab coding trace | github.com/uw-syfi/TraceLab | 100 to 160 MB | rounds, cache read/creation tokens, tool latencies |

| LMCache agentic traces (Mar 2026) | huggingface.co/datasets/sammshen/lmcache-agentic-traces | 2.4 GB | session id, cumulative input, gap before each request (tool or think seconds), output length; OpenHands, Inspect, WildClaw |
| SWE-bench experiments | github.com/SWE-bench/experiments, s3://swe-bench-submissions/ | 47 + 40 GB | per-call cache read/creation tokens and timestamps (mini-SWE-agent v2) |
| Terminal-Bench 2.0 | huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard | about 122 GB | per-step timestamps and cached tokens, API request times |

## Reasoning and test-time compute

| Name | Where | Size | Key fields |
|---|---|---|---|
| MathArena outputs | huggingface.co/datasets/MathArena/aime_2025_outputs (and about 40 more) | 143 MB for AIME25 | input and output tokens per sample, 67 configs x 4 samples, reasoning-effort sweeps, tails to 301K tokens (CC-BY-NC-SA) |
| H4 best-of-N, beam, DVTS completions | huggingface.co/HuggingFaceH4 (Llama-3.2 1B/3B) | 2.8 GB | completion tokens per sample, PRM scores, N from 4 to 2048 |
| Monkey Business | huggingface.co/datasets/ScalingIntelligence/monkey_business | 18.5 GB | 10,000 samples per problem (needs tokenizing) |

## Fleet traces

| Name | Where | Size | Key fields |
|---|---|---|---|
| Alibaba xMaaS 2026 (ATC'26) | alibaba/clusterdata, cluster-trace-v2026-maas | 10.5 + 4.8 GB + 0.1 MB | per-instance GPU and host memory, SM use; hourly kv_cache_util for two apps |
| Alibaba GPU v2026 (OSDI'26) | alibaba/clusterdata | 352 GB pod table (too large here) | pod requests and utilization |
| Azure VM, packing, GreenSKU | AzurePublicDataset | up to 167 GB | VM memory buckets |

## CXL device measurements

| Name | Where | Content |
|---|---|---|
| Melody (ASPLOS'25) | github.com/MoatLab/Melody, data/ | slowdowns of 263 workloads (including llama7b, dlrm, gpt-2) on three real CXL devices and remote NUMA |
| ByteDance ASIC CXL (EuroSys'24) | github.com/bytedance/eurosys24-artifacts | about 2.1K Intel MLC loaded-latency logs, CXL and DDR5, read/write mixes |

## Model internals

| Name | Where | Content |
|---|---|---|
| MoE expert selection traces | huggingface.co/datasets/core12345/MoE_expert_selection_trace | 165 GB+, DeepSeek-R1, Kimi-K2, Llama-4-Maverick, Qwen3-235B routing (gated, auto-approve) |
| FlashMoE routing history | huggingface.co/datasets/FlashMoE/FlashMoE-routing-history | 3.8 GB, OLMoE, Qwen3-30B-A3B |
| FLAME-MoE traces | huggingface.co/datasets/CMU-FLAME/FLAME-MoE-Traces | 296 GB, pretraining routing |
| DuoAttention, Retrieval_Head, MInference | GitHub repos | per-head attention patterns and scores |
| vLLM / SGLang KV events | engine feature | BlockStored and BlockRemoved events to record block lifetimes yourself |
