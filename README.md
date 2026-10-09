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

Journal analysis (findings/05):

    python3 -m pytest analysis/test_ftl.py                            FTL against Desnoyers' closed forms
    WORKERS=5 NAND_GB=2000 python3 analysis/endurance_sweep.py        per-node wear, 6 traces x 3 models x
    WORKERS=5 NAND_GB=667 python3 analysis/endurance_sweep.py         BF16/FP8/INT4 x 3 admissions x 2 duties
    python3 analysis/provisioning.py                                  cheapest flash class per workload
    python3 analysis/lifetime_waf.py                                  write amplification by KV death order
    python3 analysis/pool_sharing.py                                  shared pool vs per-host caches
    python3 analysis/mooncake_repro.py                                Mooncake FAST'25 pool setup

    python3 analysis/dram_sensitivity.py                              device DRAM cache 16 to 256 GB
    python3 analysis/load_vs_recompute.py                             break-even device bandwidth
    python3 analysis/export_streams.py                                KV write streams for MQSim
    analysis/mqsim/README.md                                          MQSim (FAST'18) cross-check of ftl.py

Each sweep takes about 40 minutes and peaks near 1 GB per worker.

Paper (ACM TOS draft): `python3 paper/make_assets.py && (cd paper && tectonic endurance_tos.tex)`.

analysis/quickchecks/ holds rough first checks (session reconstruction, shared pool versus
per-host caches); they are too small to support claims and conflict with Mooncake FAST'25 on pool
sharing.
