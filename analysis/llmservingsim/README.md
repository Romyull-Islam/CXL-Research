# LLMServingSim 2.0 cross-check of analysis/kvsim.py

LLMServingSim 2.0 (KAIST; ISPASS 2026, arXiv 2602.23036; repository casys-kaist/LLMServingSim,
commit b09c7aa) is an independent port of the vLLM V1 scheduler with its own GPU profiles. It is
run on the same 64-request trace (prompts 150 to 800 tokens, outputs 10K to 30K tokens) as
analysis/xcheck_llmservingsim.py, whose LLMServingSim outputs are in results/xcheck/.

Setup (CPU only, no Docker):

    git clone https://github.com/casys-kaist/LLMServingSim.git && cd LLMServingSim
    git checkout b09c7aa && git submodule update --init --recursive --depth 1
    python3.10 -m venv ../venv-lss && . ../venv-lss/bin/activate
    pip install pyyaml pyinstrument rich "pandas<2" "numpy<2" matplotlib protobuf
    pip install ./astra-sim/extern/graph_frontend/chakra
    bash astra-sim/build/astra_analytical/build.sh

Runs (Llama-3.1-8B on the shipped RTX PRO 6000 profile, block 16):

    # recompute (no token ids, so no prefix tier)
    python -m serving --cluster-config configs/cluster/single_node_single_instance.json \
      --block-size 16 --dataset ../long_decode_64_noids.jsonl --output ../out/long_recompute.csv
    # write-through CPU tier
    python -m serving --cluster-config configs/cluster/single_node_single_instance.json \
      --block-size 16 --enable-prefix-caching --prefix-storage CPU \
      --dataset ../long_decode_64_ids.jsonl --output ../out/long_cpu.csv
    # CXL tier (256 GB, 60 GB/s, 250 ns), config in this folder
    python -m serving --cluster-config ../cxl_kv_evict.json --block-size 16 \
      --enable-prefix-caching --prefix-storage CXL \
      --dataset ../long_decode_64_ids.jsonl --output ../out/long_cxl.csv

Trace files: long_decode_64_noids.jsonl and long_decode_64_ids.jsonl (13 MB, not in git; rebuild it
with `python make_ids_trace.py`) ({"input_toks",
"output_toks", "arrival_time_ns"}, the second with unique synthetic token ids so the tier is used);
the same requests as results/xcheck/long_decode_64.csv.
