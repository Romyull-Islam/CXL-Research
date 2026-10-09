#!/usr/bin/env bash
# Download the public test-time-scaling data into data/ (not tracked by git).
set -euo pipefail
mkdir -p data/tts
curl -fL -o data/tts/matharena_aime25.parquet \
  https://huggingface.co/datasets/MathArena/aime_2025_outputs/resolve/main/data/train-00000-of-00001.parquet
echo "HuggingFace H4 best-of-N, beam-search and DVTS completions (Llama-3.2 1B/3B) are listed at"
echo "https://huggingface.co/HuggingFaceH4; download the configs you need with the datasets library."
