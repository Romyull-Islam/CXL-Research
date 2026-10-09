"""Cross-check kvsim against LLMServingSim 2.0 (KAIST; arXiv 2602.23036, repo commit b09c7aa).

Same trace (64 requests, prompts 150 to 800 tokens, outputs 10K to 30K tokens, Poisson arrivals),
same model shapes (Llama-3.1-8B), same GPU (RTX PRO 6000 Blackwell Server, LLMServingSim's
shipped profile), same KV capacity (36,578 blocks of 16 tokens), max_num_seqs 128,
max_num_batched_tokens 2048, prefix caching on. LLMServingSim's outputs (results/xcheck/long_*.csv
and the summary numbers below) come from runs of its vLLM-v1 scheduler with recompute and with
its write-through CPU tier (512 GB at a 16 GB/s link).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate import calibrated_eff
from kvsim import Req, Sim, summarize
from models import GPUS, MODELS, Instance, load_model

X = Path(__file__).resolve().parent.parent / "results" / "xcheck"
LSS = {"recompute": dict(preemptions=50, recomputed=567365, makespan=1782.939, tpot_mean=58.30),
       "cpu tier": dict(preemptions=50, recomputed=7056, makespan=1735.561, tpot_mean=56.70)}


def run(policy, gpc=True):
    tr = pd.read_csv(X / "long_decode_64.csv")
    reqs = [Req(i, r.arrived_at, int(r.num_prefill_tokens), int(r.num_decode_tokens)) for i, r in enumerate(tr.itertuples())]
    inst = Instance(load_model(MODELS["llama8b"]), GPUS["PRO6000"], 1, eff=calibrated_eff("PRO6000"))
    sim = Sim(inst, reqs, policy=policy, tier_bytes=512e9, tier_bw=16e9, max_num_seqs=128,
              max_batched_tokens=2048, num_blocks=36578, gpu_prefix_cache=gpc)
    sim.run()
    s = summarize(sim)
    tpot = np.mean([(r.finish - r.first_tok) / max(r.out - 1, 1) for r in reqs]) * 1e3
    return s, tpot, reqs


def main():
    rows = []
    ends = {}
    for name, pol in (("recompute", "recompute"), ("cpu tier", "lru")):
        s, tpot, reqs = run(pol)
        ends[name] = np.array([r.finish for r in reqs])
        l = LSS[name]
        rows.append(dict(policy=name, preempt_kvsim=s["preemptions"], preempt_lss=l["preemptions"],
                         recomputed_kvsim=s["recomputed_tokens"], recomputed_lss=l["recomputed"],
                         makespan_kvsim=round(s["makespan"], 1), makespan_lss=l["makespan"],
                         tpot_ms_kvsim=round(tpot, 2), tpot_ms_lss=l["tpot_mean"]))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    g_k = 1 - df.makespan_kvsim[1] / df.makespan_kvsim[0]
    g_l = 1 - df.makespan_lss[1] / df.makespan_lss[0]
    print(f"\nmakespan saved by the tier: kvsim {g_k * 100:.2f}%, LLMServingSim {g_l * 100:.2f}%")
    lss = pd.read_csv(X / "long_recompute.csv", usecols=["request id", "end_time"]).sort_values("request id")
    e_l = lss.end_time.to_numpy() / 1e9
    e_k = ends["recompute"][lss["request id"].to_numpy()]
    print(f"per-request finish time, recompute: correlation {np.corrcoef(e_k, e_l)[0, 1]:.4f}, "
          f"median |error| {np.median(np.abs(e_k / e_l - 1)) * 100:.1f}%")
    s, _, _ = run("recompute", gpc=False)
    print(f"recompute without GPU prefix-cache reuse of preempted blocks (LLMServingSim reports 0 "
          f"prefix hits): preemptions {s['preemptions']}, recomputed tokens {s['recomputed_tokens']} "
          f"vs {LSS['recompute']['recomputed']}")
    return df


if __name__ == "__main__":
    main()
