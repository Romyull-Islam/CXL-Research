"""Checks for kvsim.py and models.py. Run: python -m pytest analysis/test_kvsim.py -q"""
import random

import pytest

from kvsim import Req, Sim, summarize
from models import Instance, GPUS, MODELS, load_model


def inst(model="llama8b", tp=1, gpu="H100"):
    return Instance(load_model(MODELS[model]), GPUS[gpu], tp)


def make(n, seed=0, plo=50, phi=4000, olo=300, ohi=20000, rate=None):
    rng = random.Random(seed)
    t = 0.0
    out = []
    for k in range(n):
        if rate:
            t += rng.expovariate(rate)
        out.append(Req(k, t, rng.randint(plo, phi), int(rng.paretovariate(1.2) * olo) % ohi + 1))
    return out


def run(reqs, ff=True, **kw):
    s = Sim(kw.pop("inst", inst()), reqs, **kw)
    if not ff:
        s._fast_forward = lambda i: False
    s.run()
    return s


def test_param_counts():
    # published totals: Llama-3.1-70B 70.6B, Llama-3.1-8B 8.03B, Qwen3-32B 32.8B,
    # Qwen3-235B-A22B 235B/22B, Qwen3-30B-A3B 30.5B/3.3B, DeepSeek-V3/R1 671B/37B
    exp = {"llama70b": (70.6, 70.6), "llama8b": (8.03, 8.03), "qwen3-32b": (32.8, 32.8),
           "qwen3-235b": (235, 22), "qwen3-30b": (30.5, 3.3), "dsr1": (671, 37)}
    for k, (tot, act) in exp.items():
        m = load_model(MODELS[k])
        assert m.params_total / 1e9 == pytest.approx(tot, rel=0.02)
        assert m.params_active / 1e9 == pytest.approx(act, rel=0.05)
    assert load_model(MODELS["llama70b"]).kv_bytes_per_token == 327680
    assert load_model(MODELS["dsr1"]).kv_bytes_per_token == 61 * 576 * 2


@pytest.mark.parametrize("policy", ["recompute", "swap", "lru"])
def test_conservation(policy):
    reqs = make(300, seed=1, rate=5.0)
    s = run(reqs, policy=policy, tier_bytes=50e9, num_blocks=3000)
    assert all(r.finish is not None and r.gen == r.out for r in reqs)
    assert s.fq.n == s.nblocks
    assert s.st["preemptions"] > 0
    if policy == "swap":
        assert s.swap_used == 0


@pytest.mark.parametrize("policy", ["recompute", "swap"])
def test_fast_forward_exact(policy):
    a = make(200, seed=2, rate=3.0)
    b = make(200, seed=2, rate=3.0)
    s1 = run(a, ff=True, policy=policy, tier_bytes=20e9, num_blocks=2500)
    s2 = run(b, ff=False, policy=policy, tier_bytes=20e9, num_blocks=2500)
    assert s1.st["preemptions"] == s2.st["preemptions"]
    assert s1.st["recomputed_tokens"] == s2.st["recomputed_tokens"]
    for x, y in zip(a, b):
        assert x.finish == pytest.approx(y.finish, rel=1e-9)
        assert x.n_pre == y.n_pre


def test_no_pressure_no_preemption():
    reqs = make(100, seed=3, rate=0.5)
    s = run(reqs)
    assert s.st["preemptions"] == 0 and s.st["recomputed_tokens"] == 0


def test_swap_unlimited_has_no_recompute():
    reqs = make(300, seed=4, rate=5.0)
    s = run(reqs, policy="swap", tier_bytes=1e15, num_blocks=3000)
    assert s.st["preemptions"] > 0
    assert s.st["tier_full_fallbacks"] == 0
    # a swapped request restores all of its KV, so nothing is computed twice
    assert s.st["recomputed_tokens"] == 0


def test_hand_case():
    # 10 blocks of 16 tokens. A (prompt 16, out 100) arrives first, B (prompt 16, out 100) second.
    # Both admitted; they grow one block every 16 steps; at 80 tokens each they need block 6 each
    # (5 + 5 = 10 held), so B (last in running) is preempted and A continues alone.
    a, b = Req(0, 0.0, 16, 100), Req(1, 0.0, 16, 100)
    s = run([a, b], num_blocks=10, max_batched_tokens=64, gpu_prefix_cache=False)
    assert a.n_pre == 0 and b.n_pre >= 1
    assert a.finish < b.finish
    # B had 80 tokens of KV (16 prompt + 64 generated) when preempted, recomputes them on resume
    assert b.recomputed >= 80 - 1
    assert s.fq.n == 10


def test_gpu_prefix_cache_reduces_recompute():
    r1 = make(300, seed=5, rate=5.0)
    r2 = make(300, seed=5, rate=5.0)
    on = run(r1, num_blocks=3000, gpu_prefix_cache=True).st["recomputed_tokens"]
    off = run(r2, num_blocks=3000, gpu_prefix_cache=False).st["recomputed_tokens"]
    assert on <= off


def test_step_time_monotone():
    i = inst("llama70b", tp=8)
    assert i.step_time(64, 64 * 1000, []) < i.step_time(64, 64 * 10000, [])
    assert i.step_time(1, 100, []) < i.step_time(0, 0, [(8192, 4096, True)])
    assert i.step_time(0, 0, [(8192, 0, True)]) < i.step_time(0, 0, [(8192, 30000, True)])
    s = summarize(run(make(50, seed=6, rate=1.0)))
    assert s["n_done"] == 50
