#!/bin/bash
# Rerun one repetition that another user's job disturbed: drop its entries, run one rep, relabel it.
# Usage: setsid nohup bash analysis/rerun_rep.sh <gpu> > log 2>&1 &
set -u
cd "$(dirname "$0")/.."
. venv/bin/activate
export HF_HOME=$PWD/hf HF_HUB_DISABLE_XET=1 LONGBENCH_DIR=$PWD/longbench
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
G=$1

drop() {  # file, python filter for entries to drop
    cp "$1" "$1.bak"
    python - "$1" "$2" <<'EOF'
import json, sys
p, cond = sys.argv[1], sys.argv[2]
r = json.load(open(p))
keep = [x for x in r if not eval(cond, {}, {"x": x})]
json.dump(keep, open(p, "w"), indent=1)
print(f"{p}: dropped {len(r) - len(keep)}, kept {len(keep)}")
EOF
}
relabel() {  # file, number of entries kept before the rerun
    python - "$1" "$2" <<'EOF'
import json, sys
p, n = sys.argv[1], int(sys.argv[2])
r = json.load(open(p))
for x in r[n:]:
    x["rep"] = 1
json.dump(r, open(p, "w"), indent=1)
print(f"{p}: relabeled {len(r) - n} entries as rep 1")
EOF
}
count() { python -c "import json;print(len(json.load(open('$1'))))"; }

f=results/reuse/Qwen3-30B-A3B/longbench/runs.json
echo "== $(date +%T) Qwen3-30B-A3B longbench out 512 rep 1"
drop $f 'x["out"] == 512 and x["rep"] == 1'
n=$(count $f)
GPU=$G python analysis/reuse_validate.py run --model Qwen/Qwen3-30B-A3B --bench longbench --outs 512 --reps 1 \
    --mem-util 0.9 && relabel $f $n || echo "reuse rerun failed"

f=results/a100/run_DeepSeek-R1-Distill-Llama-8B_tp1_bailian.json
echo "== $(date +%T) preemption rep 1, Llama-8B"
drop $f 'x["rep"] == 1'
n=$(count $f)
python analysis/a100_validate.py run --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --gpus $G --trace bailian \
    --reps 1 --mem-util 0.9 && relabel $f $n || echo "preemption rerun failed"
echo "== $(date +%T) done"
