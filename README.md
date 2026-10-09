# Endurance limits of LLM KV caching on hybrid DRAM-over-NAND CXL memory

Everything about the project (idea, gap, goal, motivation, related work, evidence, plan): [PROJECT.md](PROJECT.md). Results so far: [findings/](findings/).

    ./fetch_data.sh                     Mooncake and Qwen-Bailian traces, ByteDance CXL MLC logs, Alibaba MaaS app metrics
    pip install -r requirements.txt

    python3 analysis/mooncake_basic.py data/mooncake/*.jsonl          trace shape and reuse share
    python3 analysis/mooncake_reuse.py data/mooncake/*.jsonl          reuse time and LRU capacity curves
    python3 analysis/hybrid_device_replay.py data/mooncake/conversation_trace.jsonl
                                                                      device DRAM over NAND replay, three admission policies
    python3 analysis/per_node_endurance.py 512 data/mooncake/*_trace.jsonl
    python3 analysis/per_node_endurance.py 16 data/bailian/qwen_trace{A,B}_blksz_16.jsonl
                                                                      NAND writes per node, load versus recompute
    python3 analysis/bytedance_mlc.py                                 CXL vs DDR5 bandwidth by read/write mix

analysis/quickchecks/ holds rough first checks (session reconstruction, shared pool versus
per-host caches); they are too small to support claims and conflict with Mooncake FAST'25 on pool
sharing.
