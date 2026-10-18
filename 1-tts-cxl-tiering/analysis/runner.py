"""Run kvsim configurations in parallel and collect summaries.

A configuration is a dict: model, gpu, tp, weight_bytes, kv_bytes, workload (name and args),
policy, tier_gb, tier_bw, plus Sim keyword arguments. Results go to results/<name>.csv.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import workloads
from calibrate import calibrated_eff
from kvsim import Sim, summarize
from models import GPUS, MODELS, Instance, load_model

RESULTS = Path(__file__).resolve().parent.parent / "results"
SIM_KEYS = ("max_num_seqs", "max_batched_tokens", "gpu_prefix_cache", "num_blocks")


def build_instance(c):
    m = load_model(MODELS[c["model"]], kv_dtype_bytes=c.get("kv_bytes", 2))
    return Instance(m, GPUS[c["gpu"]], c["tp"], weight_bytes=c.get("weight_bytes", 2.0),
                    eff=calibrated_eff(c["gpu"], kernel=c.get("kernel", "fa3"),
                                       step_overhead=c.get("overhead", 0.5e-3)))


def build_requests(c, inst):
    w = dict(c["workload"])
    kind = w.pop("kind")
    max_len = c.get("max_len", 131072)
    return getattr(workloads, kind)(max_len=max_len, **w)


def run_one(c):
    t0 = time.time()
    inst = build_instance(c)
    reqs = build_requests(c, inst)
    kw = {k: c[k] for k in SIM_KEYS if k in c}
    if "hbm_kv_gb" in c:
        kw["num_blocks"] = int(c["hbm_kv_gb"] * 1e9 // inst.block_bytes)
    sim = Sim(inst, reqs, policy=c.get("policy", "recompute"), tier_bytes=c.get("tier_gb", 0) * 1e9,
              tier_bw=c.get("tier_bw", 50e9), **kw)
    sim.run()
    s = summarize(sim)
    s["wall_s"] = time.time() - t0
    s["block_gb"] = inst.block_bytes / 1e9
    flat = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in c.items()}
    out = {**flat, **s}
    if c.get("keep_timeline"):
        out["timeline"] = json.dumps(sim.timeline)
    return out


def run_all(configs, name, workers=None):
    # 12 workers keep total RSS near 3 GB; the 32 GB host also runs the author's llama-server
    workers = workers or min(len(configs), 12)
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(run_one, configs, chunksize=1))
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.drop(columns=[c for c in ("timeline",) if c in df], errors="ignore").to_csv(RESULTS / f"{name}.csv", index=False)
    if "timeline" in df:
        df[["timeline"]].assign(idx=range(len(df))).to_json(RESULTS / f"{name}_timelines.json", orient="records")
    return df
