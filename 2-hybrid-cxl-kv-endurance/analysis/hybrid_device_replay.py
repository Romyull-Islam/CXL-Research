"""Replay Mooncake prompt-block streams through a hybrid CXL device (DRAM cache over NAND).

Questions: how much reuse the device DRAM serves, how many NAND bytes are written per byte read,
and whether loading reused KV from the device beats recomputing it on the GPU.

Usage: python3 hybrid_device_replay.py [trace.jsonl ...]
"""
import json
import sys
from collections import OrderedDict

BLOCK = 512
H100_FLOPS = 989e12 * 0.5            # dense BF16 at 50% utilization

# name: KV bytes/token, active params, layers, attention width, GPUs used for prefill
MODELS = {
    "Qwen2.5-7B": (2 * 28 * 4 * 128 * 2, 7.6e9, 28, 3584, 1),
    "DeepSeek-V3": ((512 + 64) * 61 * 2, 37e9, 61, 20480, 8),
    "Llama-3.1-70B": (2 * 80 * 8 * 128 * 2, 70e9, 80, 8192, 8),
}
DEVICE_DRAM_GB = (16, 48)
NAND_GB = 2000
DRAM_HIT_GBPS = 26.0                  # GPU <-> CXL pool, measured by Beluga
NAND_MISS_GBPS = (1.0, 4.5)           # ITME prototype miss path; CMM-H measured peak


def recompute_s(n_tokens, model):
    _, params, layers, width, gpus = MODELS[model]
    flops = 2 * params * n_tokens + 2 * layers * width * n_tokens ** 2
    return flops / (gpus * H100_FLOPS)


class Tier:
    def __init__(self, capacity):
        self.cap, self.lru = capacity, OrderedDict()

    def __contains__(self, h):
        return h in self.lru

    def touch(self, h):
        self.lru.move_to_end(h)

    def insert(self, h):
        """Insert h; return the evicted block id or None."""
        self.lru[h] = True
        if len(self.lru) > self.cap:
            return self.lru.popitem(last=False)[0]
        return None


def future_reuse(reqs):
    """For every reference position, whether that block is referenced again later."""
    flat = [h for r in reqs for h in r["hash_ids"]]
    later, out = set(), [False] * len(flat)
    for i in range(len(flat) - 1, -1, -1):
        out[i] = flat[i] in later
        later.add(flat[i])
    return out


def replay(reqs, model, dram_gb, admit, reused_later=None, block_tokens=BLOCK):
    bpt = MODELS[model][0]
    block_bytes = block_tokens * bpt
    dram, nand = Tier(int(dram_gb * 1e9 // block_bytes)), Tier(int(NAND_GB * 1e9 // block_bytes))
    refs, last_ref_reused, pos = {}, {}, 0

    def admit_ok(h):
        if admit == "all":
            return True
        if admit == "2nd hit":
            return refs.get(h, 0) >= 2
        return last_ref_reused.get(h, False)
    st = dict(hit_dram=0, hit_nand=0, miss=0, nand_w=0, nand_r=0, load_wins={g: 0 for g in NAND_MISS_GBPS},
              hit_reqs=0, t_load={g: 0.0 for g in NAND_MISS_GBPS}, t_recompute=0.0)
    for r in reqs:
        hd = hn = 0
        prefix_hit = True
        for h in r["hash_ids"]:
            refs[h] = refs.get(h, 0) + 1
            if reused_later is not None:
                last_ref_reused[h] = reused_later[pos]
            pos += 1
            in_dram, in_nand = h in dram, h in nand
            if prefix_hit and (in_dram or in_nand):
                if in_dram:
                    hd += 1
                    dram.touch(h)
                else:
                    hn += 1
                    nand.touch(h)
                    st["nand_r"] += 1
                    ev = dram.insert(h)
                    if ev is not None and ev not in nand and admit_ok(ev):
                        st["nand_w"] += 1
                        nand.insert(ev)
                continue
            prefix_hit = False
            st["miss"] += 1
            if not in_dram:
                ev = dram.insert(h)
                if ev is not None and ev not in nand and admit_ok(ev):
                    st["nand_w"] += 1
                    nand.insert(ev)
        st["hit_dram"] += hd
        st["hit_nand"] += hn
        if hd + hn:
            st["hit_reqs"] += 1
            t_rc = recompute_s((hd + hn) * block_tokens, model)
            st["t_recompute"] += t_rc
            for g in NAND_MISS_GBPS:
                t_ld = (hd * block_bytes) / (DRAM_HIT_GBPS * 1e9) + (hn * block_bytes) / (g * 1e9)
                st["t_load"][g] += min(t_ld, t_rc)
                st["load_wins"][g] += t_ld < t_rc
    return st


def main(paths):
    for path in paths:
        reqs = [json.loads(line) for line in open(path)]
        hours = (reqs[-1]["timestamp"] - reqs[0]["timestamp"]) / 3.6e6
        print(f"== {path.split('/')[-1]} ({len(reqs)} requests, {hours:.2f} h)")
        oracle = future_reuse(reqs)
        print(f"   {'model':14} {'dev DRAM':>8} {'admit':>8} {'hits served':>11} {'from DRAM':>9} {'NAND w/r':>8} "
              f"{'NAND TB/day':>11} {'DWPD 2TB':>8} {'load beats recompute, NAND 1 / 4.5 GB/s':>40}")
        for model in MODELS:
            bpt = MODELS[model][0]
            total_refs = sum(len(r["hash_ids"]) for r in reqs)
            for dram_gb in DEVICE_DRAM_GB:
                for admit in ("all", "2nd hit", "oracle"):
                    s = replay(reqs, model, dram_gb, admit, oracle)
                    hits = s["hit_dram"] + s["hit_nand"]
                    tb_day = s["nand_w"] * BLOCK * bpt / 1e12 / hours * 24
                    wins = "  ".join(f"{s['load_wins'][g] / max(s['hit_reqs'], 1):.0%}" for g in NAND_MISS_GBPS)
                    print(f"   {model:14} {dram_gb:>6} GB {admit:>8} {hits / total_refs:>11.0%} "
                          f"{s['hit_dram'] / max(hits, 1):>9.0%} {s['nand_w'] / max(s['nand_r'], 1):>8.2f} "
                          f"{tb_day:>11.1f} {tb_day / 2:>8.0f} {wins:>40}")


if __name__ == "__main__":
    main(sys.argv[1:])
