"""Per-node NAND wear of a hybrid CXL KV cache, with consistent scaling and measured write
amplification.

For each trace, model, KV precision, admission policy and prefill duty:
1. find the traffic multiple S that fills one 8-GPU node's prefill compute (nand_stream.scaled_node),
   with the device (48 GB DRAM over 2 TB NAND) divided by S;
2. replay at that scale and log the NAND write stream;
3. run it through the FTL (ftl.py, greedy GC, 64 MB erase blocks) at 7% and 28% over-provisioning;
4. NAND writes per day = host write rate x WAF, in TB/day and drive writes per day of the 2 TB device.

Node prefill rate: 8 x H100, dense BF16 at 50% utilization, linear FLOPs only (the attention term
is left out, which overstates the rate and so the writes, by under 10% at these prompt lengths).

Run: python analysis/endurance_sweep.py  (writes results/endurance_sweep.csv; 8 worker processes)
"""
import itertools
import os
import sys

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ftl import FTL  # noqa: E402
from nand_stream import load, oracle_flags, replay, scaled_node  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
H100 = 989e12 * 0.5
GPUS = 8
DRAM_GB = 48
NAND_GB = float(os.environ.get("NAND_GB", 2000))   # 2000: TLC device; 667: the same NAND in pSLC mode
ERASE_BYTES = 64 << 20
OPS = (0.07, 0.28)
TRACES = {
    "mooncake_conversation": ("mooncake/conversation_trace.jsonl", 512),
    "mooncake_toolagent": ("mooncake/toolagent_trace.jsonl", 512),
    "bailian_A_chat": ("bailian/qwen_traceA_blksz_16.jsonl", 16),
    "bailian_B_api": ("bailian/qwen_traceB_blksz_16.jsonl", 16),
    "bailian_thinking": ("bailian/qwen_thinking_blksz_16.jsonl", 16),
    "bailian_coder": ("bailian/qwen_coder_blksz_16.jsonl", 16),
}
# name: KV elements per token (all layers), active parameters
MODELS = {
    "Llama-3.1-70B": (2 * 80 * 8 * 128, 70.6e9),
    "DeepSeek-V3": ((512 + 64) * 61, 37.6e9),
    "Qwen2.5-7B": (2 * 28 * 4 * 128, 7.6e9),
}
PRECISION = {"BF16": 2.0, "FP8": 1.0, "INT4": 0.5}
ADMIT = ("all", "2nd hit", "oracle")
DUTY = (1.0, 0.3)

_cache = {}


def _trace(name):
    if name not in _cache:
        tr = load(ROOT / "data" / TRACES[name][0])
        _cache.clear()
        _cache[name] = (tr, oracle_flags(tr["seqs"]))
    return _cache[name]


def one(cfg):
    name, model, prec, admit, duty = cfg
    tr, orc = _trace(name)
    bt = TRACES[name][1]
    elems, params = MODELS[model]
    kv_tok = elems * PRECISION[prec]
    st = scaled_node(tr, bt, kv_tok, GPUS * H100 / (2 * params), duty, DRAM_GB, NAND_GB, admit, orc, log=True)
    row = dict(trace=name, model=model, precision=prec, admit=admit, duty=duty, S=st["S"],
               block_MB=bt * kv_tok / 2**20, refs=st["refs"], hit_share=(st["hit_dram"] + st["hit_nand"]) / st["refs"],
               dram_share_of_hits=st["hit_dram"] / max(st["hit_dram"] + st["hit_nand"], 1),
               persisted=st["nand_w"] / max(st["miss"], 1), nand_w_per_r=st["nand_w"] / max(st["nand_r"], 1),
               host_GBs=st["nand_write_GBs"], nand_blocks=st["nand_blocks"], nand_writes=st["nand_w"])
    units_per_eb = ERASE_BYTES / (bt * kv_tok)
    writes, row["waf_extended"] = st["writes"], 1
    if units_per_eb >= 1 and len(writes) < 4 * st["nand_blocks"]:
        # the trace is too short to overwrite the NAND several times: append copies of it with
        # new block ids (new users, no reuse across copies) until it does, for the WAF only
        cap = max(int(25e6 // max(st["refs"], 1)), 1)        # bound memory: about 25M references
        k = int(min(np.ceil(4 * st["nand_blocks"] / max(len(writes), 1)), 12, cap))
        off = tr["n_blocks"]
        ext = dict(seqs=[s + c * off for c in range(k) for s in tr["seqs"]], duration=tr["duration"] * k)
        orc_k = np.tile(orc, k) if orc is not None else None
        writes = replay(ext, st["dram_blocks"], st["nand_blocks"], admit, orc_k, log=True)["writes"]
        row["waf_extended"] = k
    for op in OPS:
        key = f"waf_op{int(op * 100)}"
        if units_per_eb < 1:
            row[key] = 1.0      # KV blocks span whole erase blocks: no GC copies
        elif len(writes) < 3 * st["nand_blocks"]:
            row[key] = float("nan")
        else:
            row[key] = FTL(st["nand_blocks"], op, units_per_eb).run(writes, warmup=2 * st["nand_blocks"])["waf"]
    for op in OPS:
        waf = row[f"waf_op{int(op * 100)}"]
        row[f"TBday_op{int(op * 100)}"] = row["host_GBs"] * 86400 / 1e3 * (waf if waf == waf else 1.0)
        row[f"DWPD_op{int(op * 100)}"] = row[f"TBday_op{int(op * 100)}"] / (NAND_GB / 1e3)
    print(name, model, prec, admit, duty, f"S={row['S']:.2f} host {row['host_GBs']:.2f} GB/s "
          f"WAF {row['waf_op7']:.2f}/{row['waf_op28']:.2f}", flush=True)
    return row


def main():
    cfgs = list(itertools.product(TRACES, MODELS, PRECISION, ADMIT, DUTY))
    # group by trace so each worker keeps one trace in memory
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 8))) as ex:
        rows = list(ex.map(one, cfgs, chunksize=6))
    out = ROOT / "results" / f"endurance_sweep_{int(NAND_GB)}GB.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
