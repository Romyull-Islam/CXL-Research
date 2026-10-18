"""Checks for parksim.py and experiments.py. Run: python -m pytest model/test_parksim.py -q"""
import pytest
import simpy

from experiments import calibrated, one_way
from parksim import Device, Params


def _time(gen_fn):
    env = simpy.Environment()
    p = calibrated(400.0)
    dev = Device(env, p)
    out = {}

    def run():
        t0 = env.now
        out["v"] = yield env.process(gen_fn(env, dev, p))
        out["t"] = env.now - t0
    env.process(run())
    env.run()
    return out, p


def test_idle_read_matches_calibration():
    out, p = _time(lambda env, dev, p: dev.read(0, 1))
    assert out["t"] == pytest.approx(400.0, rel=1e-9)


def test_parked_read_waits_for_write():
    def gen(env, dev, p):
        r = env.process(dev.read(1, 7, park=True))
        yield env.timeout(5000)
        env.process(dev.write(0, 7, 3))
        return (yield r)
    out, p = _time(gen)
    assert out["v"] == (3, True)
    # the write lands about half a round trip after 5 us, the response takes another half
    assert 5000 + 0.8 * p.idle_rtt() < out["t"] < 5000 + 1.3 * p.idle_rtt()


def test_parked_read_times_out_empty():
    def gen(env, dev, p):
        return (yield env.process(dev.read(1, 9, park=True)))
    out, p = _time(gen)
    assert out["v"] == (0, False)
    assert out["t"] == pytest.approx(p.park_timeout + p.idle_rtt(), rel=1e-6)


def test_table_full_answers_empty_at_once():
    env = simpy.Environment()
    p = calibrated(400.0)
    p.table = 1
    dev = Device(env, p)
    env.process(dev.read(1, 11, park=True))
    t = {}

    def second():
        yield env.timeout(10)
        t0 = env.now
        v = yield env.process(dev.read(1, 12, park=True))
        t["v"], t["dt"] = v, env.now - t0
    env.process(second())
    env.run(until=20000)
    assert t["v"] == (0, False) and t["dt"] < 2 * p.idle_rtt()
    assert dev.stats["table_full"] == 1


def test_parked_beats_flag_polling_for_small_messages():
    p = calibrated(600.0)
    assert one_way(p, 64, "parked") < one_way(p, 64, "poll_flag")
