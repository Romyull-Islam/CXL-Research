# Sizing CXL memory tiers for test-time-scaling LLM workloads

Everything about the project (idea, gap, goal, motivation, related work, evidence, plan): [PROJECT.md](PROJECT.md). Results so far: [findings/](findings/).

    ./fetch_data.sh                     download MathArena AIME 2025 outputs into data/
    pip install -r requirements.txt
    python3 analysis/tts_skew.py        KV byte-second skew within and across samples
