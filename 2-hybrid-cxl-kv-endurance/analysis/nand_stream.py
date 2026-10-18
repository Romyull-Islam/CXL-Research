"""Replay a prefix-block trace through a hybrid CXL device and log what reaches the NAND.

Device: an LRU DRAM cache over an LRU NAND tier (inclusive: a block read from NAND is promoted to
DRAM and stays in NAND). A block evicted from DRAM is written to NAND if the admission policy
accepts it and NAND does not already hold it. When NAND is full, its LRU block is evicted and the
new block takes the freed slot, so the host overwrites that slot (CXL.mem has no TRIM, so the
device learns that the old contents are dead only when the slot is overwritten).

Requests use their blocks as a prefix: blocks found in DRAM or NAND are hits until the first
miss; from there on every block is recomputed and inserted into DRAM.

Admission: "all", "2nd hit" (block referenced at least twice so far) or "oracle" (the block's
latest reference is followed by another reference later in the trace).

Scaling: a trace is one stream of users. A node that serves S times that traffic interleaves S
independent streams, which multiplies reuse distances by about S; we model it by dividing the
device capacities by S. scaled_node() finds S such that the node's prefill compute matches the
trace's miss tokens.
"""
import json
from array import array
from collections import OrderedDict

import numpy as np


def load(path):
    reqs = [json.loads(line) for line in open(path)]
    ts = [r["timestamp"] for r in reqs]
    unit = 1e-3 if max(ts) > 1e5 else 1.0          # Mooncake: ms, Bailian: s
    ids, seqs = {}, []
    for r in reqs:
        seqs.append(np.fromiter((ids.setdefault(h, len(ids)) for h in r["hash_ids"]), dtype=np.int64))
    return dict(seqs=seqs, duration=(max(ts) - min(ts)) * unit, n_blocks=len(ids))


def oracle_flags(seqs):
    """For every reference position, whether that block is referenced again later."""
    flat = np.concatenate(seqs)
    out = np.zeros(len(flat), dtype=bool)
    seen = set()
    for i in range(len(flat) - 1, -1, -1):
        b = int(flat[i])
        out[i] = b in seen
        seen.add(b)
    return out


def replay(tr, dram_blocks, nand_blocks, admit, oracle=None, log=False):
    """Returns counters and, with log=True, the NAND write stream as slot numbers (an overwrite
    when the slot was used before) plus the slot freed by each NAND eviction."""
    dram, nand = OrderedDict(), OrderedDict()      # block -> None ; block -> slot
    refs = {}
    last_reused = {}
    free_slots, next_slot = [], 0
    writes = array("q") if log else None
    evicts = array("q") if log else None
    st = dict(refs=0, hit_dram=0, hit_nand=0, miss=0, nand_w=0, nand_r=0, nand_evict=0, hit_reqs=0)
    pos = 0

    count_refs = admit == "2nd hit"

    def admit_ok(b):
        if admit == "all":
            return True
        if admit == "2nd hit":
            return refs.get(b, 0) >= 2
        # oracle flags are only needed while a block sits in DRAM
        return last_reused.pop(b, False)

    def dram_insert(b):
        nonlocal next_slot
        dram[b] = None
        if len(dram) > dram_blocks:
            ev, _ = dram.popitem(last=False)
            if ev not in nand and admit_ok(ev):
                if len(nand) >= nand_blocks:
                    old, slot = nand.popitem(last=False)
                    st["nand_evict"] += 1
                    if log:
                        evicts.append(slot)
                else:
                    slot = next_slot
                    next_slot += 1
                nand[ev] = slot
                st["nand_w"] += 1
                if log:
                    writes.append(slot)

    for seq in tr["seqs"]:
        hit = True
        nh = 0
        for b in seq.tolist():
            if count_refs:
                refs[b] = refs.get(b, 0) + 1
            if oracle is not None and admit == "oracle":
                last_reused[b] = bool(oracle[pos])
            pos += 1
            st["refs"] += 1
            if hit:
                if b in dram:
                    dram.move_to_end(b)
                    st["hit_dram"] += 1
                    nh += 1
                    continue
                if b in nand:
                    nand.move_to_end(b)
                    st["hit_nand"] += 1
                    st["nand_r"] += 1
                    nh += 1
                    dram_insert(b)
                    continue
                hit = False
            st["miss"] += 1
            if b in dram:
                dram.move_to_end(b)
            else:
                dram_insert(b)
        st["hit_reqs"] += nh > 0
    st["slots_used"] = next_slot
    if log:
        st["writes"] = np.frombuffer(writes, dtype=np.int64).copy()
        st["evicts"] = np.frombuffer(evicts, dtype=np.int64).copy()
    return st


def scaled_node(tr, block_tokens, kv_bytes_tok, node_tok_s, duty, dram_gb, nand_gb, admit, oracle,
                iters=6, log=False):
    """Fixed point for S: node prefill tokens/s x duty = S x trace miss tokens/s, with device
    capacities divided by S. Returns the replay at that scale with S and per-node rates."""
    block_bytes = block_tokens * kv_bytes_tok

    def caps(s):
        return max(int(dram_gb * 1e9 / block_bytes / s), 1), max(int(nand_gb * 1e9 / block_bytes / s), 1)
    s = 1.0
    for _ in range(iters):
        st = replay(tr, *caps(s), admit, oracle)
        s_new = node_tok_s * duty / (st["miss"] * block_tokens / tr["duration"])
        done = abs(s_new / s - 1) < 0.01
        s = s_new
        if done:
            break
    st = replay(tr, *caps(s), admit, oracle, log=log)
    st.update(S=s, dram_blocks=caps(s)[0], nand_blocks=caps(s)[1],
              nand_write_GBs=s * st["nand_w"] * block_bytes / tr["duration"] / 1e9,
              miss_tok_s=s * st["miss"] * block_tokens / tr["duration"])
    return st
