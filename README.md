# Sizing CXL memory tiers for test-time-scaling LLM workloads

Goal, gap, motivation and related work: [GOAL.md](GOAL.md). Results so far: [findings/](findings/).

    ./fetch_data.sh                     download MathArena AIME 2025 outputs into data/
    pip install -r requirements.txt
    python3 analysis/tts_skew.py        KV byte-second skew within and across samples
