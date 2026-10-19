#!/bin/bash
# A100 runs for the letter; finished steps are skipped on a restart. <gpus> is one index or a comma
# list ("0,1" runs each model tensor parallel over both GPUs). With --preemption, the first model's
# Bailian preemption run follows its microbenchmark, before the reuse runs.
# Usage (in the repository copy on the server, venv in ./venv):
#   setsid nohup bash analysis/run_a100.sh <gpus> <model> [<model> ...] [--preemption] > log 2>&1 &
set -u
cd "$(dirname "$0")/.."
. venv/bin/activate
export HF_HOME=$PWD/hf HF_HUB_DISABLE_XET=1 LONGBENCH_DIR=$PWD/longbench
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
G=$1; shift
TP=$(echo "$G" | tr ',' '\n' | wc -l)
PRE=0
MODELS=()
for a in "$@"; do [ "$a" = "--preemption" ] && PRE=1 || MODELS+=("$a"); done
UTIL=0.9
for m in "${MODELS[@]}"; do
    s=${m##*/}
    echo "== $(date +%T) $m on GPU $G (tp $TP)"
    python -c "from huggingface_hub import snapshot_download as d; d('$m', allow_patterns=['*.json','*.safetensors','*.model','*.txt','tokenizer*'])" \
        || { echo "download failed: $m"; continue; }
    [ -f results/a100/micro_${s}_tp${TP}.json ] || \
        python analysis/a100_validate.py microbench --model "$m" --gpus "$G" --tp $TP --mem-util $UTIL \
        || { echo "microbench failed: $m"; continue; }
    if [ $PRE = 1 ] && [ "$m" = "${MODELS[0]}" ]; then
        echo "== $(date +%T) preemption run, $m, Bailian trace"
        [ -f results/a100/run_${s}_tp${TP}_bailian.json ] || \
            python analysis/a100_validate.py run --model "$m" --gpus "$G" --tp $TP --trace bailian --mem-util $UTIL \
            || echo "preemption run failed"
    fi
    for b in docqa longbench; do
        [ -f results/reuse/$s/$b/runs.json ] && continue
        GPU=$G python analysis/reuse_validate.py run --model "$m" --bench $b --outs 4 64 512 --reps 2 \
            --mem-util $UTIL --tp $TP || echo "reuse failed: $m $b"
    done
    [ -f results/reuse/$s/mixbench.json ] || \
        CUDA_VISIBLE_DEVICES=$G python analysis/reuse_validate.py mixbench --model "$m" --mem-util $UTIL --tp $TP \
        || echo "mixbench failed: $m"
done
echo "== $(date +%T) done"
