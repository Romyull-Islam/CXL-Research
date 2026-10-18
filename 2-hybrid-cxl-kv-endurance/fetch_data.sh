#!/usr/bin/env bash
# Download the public traces and device measurements into data/ (not tracked by git).
set -euo pipefail
mkdir -p data/mooncake data/bailian data/bytedance data/maas
for f in conversation_trace synthetic_trace toolagent_trace; do
  curl -fsL -o data/mooncake/$f.jsonl \
    https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/$f.jsonl
done
for f in qwen_traceA_blksz_16 qwen_traceB_blksz_16 qwen_thinking_blksz_16 qwen_coder_blksz_16; do
  curl -fsL -o data/bailian/$f.jsonl \
    https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/main/$f.jsonl
done
git clone -q --depth 1 https://github.com/bytedance/eurosys24-artifacts.git data/bytedance/repo
curl -fsL -o data/maas/app_metrics.csv.zip \
  https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/cluster-trace-v2026-maas/data/app_metrics.csv.zip
(cd data/maas && unzip -o -q app_metrics.csv.zip)
