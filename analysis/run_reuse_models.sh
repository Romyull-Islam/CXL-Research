#!/bin/bash
# Real-vLLM test of the first-order model on several architectures and two benchmarks, one run at a
# time. Usage: PY=/path/to/vllm-venv/bin/python bash analysis/run_reuse_models.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}
LOG=results/reuse
# model, GPU memory fraction for the reuse runs
MODELS="unsloth/Llama-3.2-1B-Instruct:0.6
ibm-granite/granite-3.1-1b-a400m-instruct:0.6
unsloth/gemma-3-1b-it:0.6
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:0.9"
for entry in $MODELS; do
    m=${entry%%:*}; util=${entry##*:}; s=${m##*/}
    [ -f results/a100/micro_${s}_tp1.json ] || \
        $PY analysis/a100_validate.py microbench --model "$m" --gpus 0 > $LOG/micro_$s.log 2>&1 || \
        { echo "microbench failed: $m"; continue; }
    for b in docqa longbench; do
        [ -f $LOG/$s/$b/runs.json ] && continue
        $PY analysis/reuse_validate.py run --model "$m" --bench $b --outs 4 64 512 --reps 2 \
            --mem-util $util > $LOG/run_${s}_$b.log 2>&1 || echo "run failed: $m $b"
    done
done
echo done
