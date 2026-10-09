# Sizing CXL memory tiers for test-time-scaling LLM workloads

Everything about the project (idea, gap, goal, motivation, related work, evidence, plan): [PROJECT.md](PROJECT.md). Results so far: [findings/](findings/). Status: gate 2 failed for keeping preempted KV in CXL ([findings/02](findings/02-preempted-kv-tier.md)).

    ./fetch_data.sh                          download the public data into data/
    python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
    python3 analysis/tts_skew.py             MathArena KV byte-second skew (findings/01)
    python3 analysis/h4_tokens.py            token counts of the H4 completions (about 11 min)
    python3 analysis/h4_memory.py            large-N KV memory per strategy (findings/03)
    python3 analysis/calibrate.py            fit and check the step-time model on Vidur's H100 profiles
    python3 -m pytest analysis/test_kvsim.py simulator tests
    python3 analysis/gate2_saturation.py     then gate2_capacity, gate2_maxseqs, gate2_online,
                                             gate2_load, gate2_goodput, gate2_pool, gate2_sensitivity,
                                             gate2_fp8_tuned (findings/02)
    python3 analysis/first_order.py          first-order model against the saturation runs
    python3 analysis/xcheck_llmservingsim.py cross-check against LLMServingSim 2.0
    python3 analysis/vllm_validate.py run    real vLLM run on a local GPU (needs a vLLM venv),
    python3 analysis/vllm_validate.py compare  then compare with the simulator

Code: analysis/kvsim.py (vLLM V1 scheduler and KV tiers), models.py (model shapes and step time),
workloads.py (traces), runner.py (parallel sweeps). Sweeps use 12 workers, about 250 MB each.

## Final test on an A100 server (pending)

    here:    python3 analysis/a100_validate.py traces           (already in results/a100/)
    server:  rsync the repository without data/ and .venv/, then in a fresh venv:
             uv pip install vllm==0.31.0 pandas --torch-backend=auto
             python3 analysis/a100_validate.py check             pick idle GPUs
             python3 analysis/a100_validate.py microbench --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --gpus 0
             python3 analysis/a100_validate.py run --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --gpus 0 --trace bailian
             python3 analysis/a100_validate.py run ... --trace bo16_llama
             optional, 4x A100 80 GB: --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --tp 4 --gpus 0,1,2,3
    here:    copy results/a100/*.json back, then python3 analysis/a100_validate.py compare

The 8B runs take about 10 minutes per configuration for bailian and 30 for bo16_llama (two
repetitions of recompute and CPU offload each). Every run records other users' GPU processes;
contended runs are dropped by compare.
