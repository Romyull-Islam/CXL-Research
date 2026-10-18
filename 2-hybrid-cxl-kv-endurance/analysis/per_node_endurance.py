"""Usage: python3 per_node_endurance.py BLOCK_TOKENS trace.jsonl ...

Per-node NAND write rate if a hybrid CXL device holds the KV of one 8-GPU node.

persisted = NAND blocks written / blocks computed (prefill misses), taken from the trace replay.
Node prefill rate from a FLOP roofline: 8 x H100, dense BF16 at 50% utilization, linear term only
(attention FLOPs ignored, which overstates throughput slightly and therefore the write rate).
"""
import json
import sys

sys.path.insert(0, "analysis")
from hybrid_device_replay import BLOCK, H100_FLOPS, MODELS, future_reuse, replay  # noqa: E402

GPUS_PER_NODE = 8
DEVICE_TB = 2.0
TLC_DWPD = (1, 3)
PREFILL_DUTY = (1.0, 0.3)       # dedicated prefill node; collocated node spending 30% on prefill

BLOCK_TOKENS = int(sys.argv[1])

for path in sys.argv[2:]:
    reqs = [json.loads(line) for line in open(path)]
    oracle = future_reuse(reqs)
    print(f"== {path.split('/')[-1]}")
    print(f"   {'model':14} {'admit':>8} {'persisted':>9} {'node tok/s':>10} {'NAND GB/s':>9} "
          f"{'DWPD (2 TB), duty 100% / 30%':>30}")
    for model, (bpt, params, _, _, _) in MODELS.items():
        tok_s = GPUS_PER_NODE * H100_FLOPS / (2 * params)
        for admit in ("all", "2nd hit", "oracle"):
            s = replay(reqs, model, 48, admit, oracle, BLOCK_TOKENS)
            persisted = s["nand_w"] / s["miss"]
            gbps = tok_s * persisted * bpt / 1e9
            dwpd = [gbps * d * 86400 / (DEVICE_TB * 1e3) for d in PREFILL_DUTY]
            print(f"   {model:14} {admit:>8} {persisted:>9.2f} {tok_s:>10,.0f} {gbps:>9.2f} "
                  f"{dwpd[0]:>14.0f} / {dwpd[1]:<14.0f}  load wins {s['load_wins'][1.0] / max(s['hit_reqs'], 1):.0%} / "
                  f"{s['load_wins'][4.5] / max(s['hit_reqs'], 1):.0%}  DRAM share {s['hit_dram'] / max(s['hit_dram'] + s['hit_nand'], 1):.0%}")
