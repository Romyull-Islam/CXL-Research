"""Checks for ftl.py and nand_stream.py. Run: python -m pytest analysis/test_ftl.py -q"""
import numpy as np
import pytest

from ftl import FTL, fifo_waf, greedy_waf


def test_sequential_overwrite_has_no_gc_copies():
    L = 4000
    f = FTL(L, 0.07, 64)
    r = f.run(np.tile(np.arange(L), 8), warmup=2 * L)
    assert r["waf"] == pytest.approx(1.0)


@pytest.mark.parametrize("op", [0.07, 0.28])
def test_fifo_matches_closed_form(op):
    L = 20000
    rng = np.random.default_rng(0)
    f = FTL(L, op, 64, policy="fifo")
    r = f.run(rng.integers(0, L, 12 * L), warmup=4 * L)
    assert r["waf"] == pytest.approx(fifo_waf(f.op_effective), rel=0.05)


@pytest.mark.parametrize("op", [0.07, 0.28])
def test_greedy_beats_fifo(op):
    L = 20000
    rng = np.random.default_rng(1)
    w = rng.integers(0, L, 12 * L)
    f = FTL(L, op, 64, policy="greedy")
    g = f.run(w, warmup=4 * L)["waf"]
    assert 1.0 < g < fifo_waf(f.op_effective)


def test_replay_conserves_slots():
    from nand_stream import replay
    rng = np.random.default_rng(2)
    tr = dict(seqs=[rng.integers(0, 3000, rng.integers(1, 30)) for _ in range(3000)], duration=100.0)
    st = replay(tr, 50, 400, "all", log=True)
    assert st["nand_w"] == len(st["writes"])
    assert st["slots_used"] <= 400
    assert st["writes"].max() < 400
    assert st["nand_evict"] == len(st["evicts"])


def test_closed_forms_match_published_values():
    # Desnoyers SYSTOR'12 Table 1 (LRU, spare factor Sf = 1 - 1/alpha) and the uncorrected greedy
    # value 13.393 at Sf = 0.03, Np = 64 quoted in its text
    for sf, a in ((0.03, 16.837), (0.07, 7.318), (0.11, 4.725), (0.23, 2.371)):
        assert fifo_waf(1 / (1 - sf) - 1) == pytest.approx(a, abs=2e-3)
    assert greedy_waf(1 / (1 - 0.03) - 1, 64) == pytest.approx(13.393, abs=2e-3)


@pytest.mark.parametrize("op", [0.07, 0.28])
def test_greedy_matches_closed_form(op):
    L = 40000
    f = FTL(L, op, 64, policy="greedy")
    r = f.run(np.random.default_rng(4).integers(0, L, 15 * L), warmup=5 * L)
    assert r["waf"] == pytest.approx(greedy_waf(f.op_effective, 64), rel=0.05)
