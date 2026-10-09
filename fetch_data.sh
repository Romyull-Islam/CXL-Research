#!/usr/bin/env bash
# Download the public test-time-scaling data into data/ (not tracked by git).
set -euo pipefail
mkdir -p data/tts
curl -fL -o data/tts/matharena_aime25.parquet \
  https://huggingface.co/datasets/MathArena/aime_2025_outputs/resolve/main/data/train-00000-of-00001.parquet
# HuggingFace H4 best-of-N, beam-search and DVTS completions on MATH-500 (Llama-3.2 1B/3B), about 2.7 GB.
python -I -c '
from huggingface_hub import snapshot_download
for m in ("1B", "3B"):
    for s in ("best-of-N", "beam-search", "DVTS"):
        r = f"Llama-3.2-{m}-Instruct-{s}-completions"
        snapshot_download("HuggingFaceH4/" + r, repo_type="dataset", local_dir="data/h4/" + r, token=False)
'
# Qwen-Bailian thinking trace (Apache 2.0): real arrival times and token counts of a reasoning service.
mkdir -p data/bailian data/configs data/vidur
curl -fsL -o data/bailian/qwen_thinking_blksz_16.jsonl \
  https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/main/qwen_thinking_blksz_16.jsonl
# Model shapes for the simulator (public config.json files).
for r in deepseek-ai/DeepSeek-R1-Distill-Llama-70B deepseek-ai/DeepSeek-R1-Distill-Llama-8B Qwen/QwQ-32B \
         Qwen/Qwen3-32B Qwen/Qwen3-235B-A22B Qwen/Qwen3-30B-A3B deepseek-ai/DeepSeek-R1 Qwen/Qwen2.5-0.5B-Instruct; do
  curl -fsL -o "data/configs/${r//\//_}.json" "https://huggingface.co/$r/resolve/main/config.json"
done
# Vidur's measured H100 kernel profiles, used by analysis/calibrate.py (commit abae7f6).
git clone -q --filter=blob:none --sparse https://github.com/microsoft/vidur.git data/vidur/repo
git -C data/vidur/repo checkout -q abae7f63aa857300f5cdc6f5e0d27860cd24721b
git -C data/vidur/repo sparse-checkout set data/profiling
cp -r data/vidur/repo/data/profiling data/vidur/
