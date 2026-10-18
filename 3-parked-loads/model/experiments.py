"""Handoff protocols and experiments on the parksim device model.

Protocols for one message of n lines from a producer host to a consumer host that is already
waiting:
  poll_flag      producer writes the payload, fences (waits for completions), writes a flag;
                 the consumer re-reads the flag (uncached, plus clflushopt and fence), then
                 reads the payload (TraCT, cMPI style)
  poll_lines     every line carries a sequence number; the consumer re-reads the lines that are
                 not yet valid until all are (the fastest software scheme)
  coherent_spin  the flag is in a hardware-coherent region; the consumer spins on its cached copy,
                 the flag write back-invalidates it, the consumer reloads the flag and the payload
  parked         the consumer's reads of the payload lines wait in the device and are answered
                 as the producer's writes fill the lines

Experiments: ping-pong latency; many waiters sharing the device with a streaming tenant; RPC with
a service time; a barrier across hosts. Run: python model/experiments.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import simpy

from parksim import Device, Params, read_lines, write_lines

PROTOCOLS = ("poll_flag", "poll_lines", "coherent_spin", "parked")
_addr = itertools.count(1 << 20)
OUT = Path(__file__).resolve().parent
_jit = np.random.default_rng(42)


def jitter(p):
    """Random start offset so the producer meets the polling loop at a random phase."""
    return float(_jit.uniform(0, 2 * p.idle_rtt() + p.poll_overhead))


def fresh(n):
    return [next(_addr) for _ in range(n)]


def producer(env, dev, h, msg, version, protocol):
    payload, flag = msg
    procs = write_lines(env, dev, h, payload, version)
    if protocol in ("poll_flag", "coherent_spin"):
        yield env.all_of(procs)                      # fence: all payload completions back
        yield env.process(dev.write(h, flag, version))
    else:
        yield env.all_of(procs)


def consumer(env, dev, p, h, msg, version, protocol):
    payload, flag = msg
    if protocol == "poll_flag":
        while True:
            v, _ = yield env.process(dev.read(h, flag))
            if v == version:
                break
            yield env.timeout(p.poll_overhead)
        yield env.process(read_lines(env, dev, h, payload, p))
    elif protocol == "poll_lines":
        todo = list(payload)
        while todo:
            got = yield env.process(read_lines(env, dev, h, todo, p))
            todo = [a for a, (v, _) in zip(todo, got) if v != version]
            if todo:
                yield env.timeout(p.poll_overhead)
    elif protocol == "coherent_spin":
        if dev.line(flag).version != version:
            yield dev.watch(h, flag)
        yield env.process(dev.read(h, flag))
        yield env.process(read_lines(env, dev, h, payload, p))
    else:
        todo = list(payload)
        while todo:
            got = yield env.process(read_lines(env, dev, h, todo, p, park=True))
            todo = [a for a, (_, full) in zip(todo, got) if not full]


def one_way(p, size, protocol, trials=60, gap=5000.0):
    """Mean one-way handoff latency: the consumer waits first, the producer starts at t0."""
    env = simpy.Environment()
    dev = Device(env, p)
    lat = []

    def run():
        for t in range(trials):
            msg = (fresh(max(size // 64, 1)), fresh(1)[0])
            c = env.process(consumer(env, dev, p, 1, msg, t + 1, protocol))
            yield env.timeout(gap + jitter(p))
            t0 = env.now
            env.process(producer(env, dev, 0, msg, t + 1, protocol))
            yield c
            lat.append(env.now - t0)
    env.process(run())
    env.run()
    return float(np.mean(lat[2:]))


def ping_pong(p, size, protocol, rounds=60):
    """Round trip of a message there and back; each side waits before the other sends."""
    env = simpy.Environment()
    dev = Device(env, p)
    rtts = []

    def run():
        for r in range(rounds):
            a = (fresh(max(size // 64, 1)), fresh(1)[0])
            b = (fresh(max(size // 64, 1)), fresh(1)[0])
            wait_b = env.process(consumer(env, dev, p, 1, a, r + 1, protocol))
            wait_a = env.process(consumer(env, dev, p, 0, b, r + 1, protocol))
            yield env.timeout(3000 + jitter(p))
            t0 = env.now
            env.process(producer(env, dev, 0, a, r + 1, protocol))
            yield wait_b
            env.process(producer(env, dev, 1, b, r + 1, protocol))
            yield wait_a
            rtts.append(env.now - t0)
    env.process(run())
    env.run()
    return float(np.mean(rtts[2:]))


def waiters_vs_tenant(p, waiters, protocol, horizon=200_000.0, mean_wait=20_000.0, seed=0,
                      tenant_hosts=4, streams=4):
    """`waiters` consumers (on hosts tenant_hosts..H-1) each wait for a 64 B message that arrives
    after an exponential delay, over and over; hosts 0..tenant_hosts-1 each run `streams` cores
    streaming reads of distinct lines with p.mlp in flight, enough to saturate the device port.
    Returns the tenants' bandwidth and the handoff latency."""
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    dev = Device(env, p)
    lat = []
    tenant_bytes = [0]

    def tenant(h):
        while env.now < horizon:
            yield env.process(read_lines(env, dev, h, fresh(p.mlp), p))
            tenant_bytes[0] += p.mlp * 64

    w_hosts = p.hosts - tenant_hosts

    def pair(i):
        h = tenant_hosts + i % w_hosts
        prod_h = tenant_hosts + (i + 1) % w_hosts
        v = 0
        while env.now < horizon:
            v += 1
            msg = (fresh(1), fresh(1)[0])
            c = env.process(consumer(env, dev, p, h, msg, v, protocol))
            yield env.timeout(rng.exponential(mean_wait))
            t0 = env.now
            env.process(producer(env, dev, prod_h, msg, v, protocol))
            yield c
            lat.append(env.now - t0)
    for h in range(tenant_hosts):
        for _ in range(streams):
            env.process(tenant(h))
    for i in range(waiters):
        env.process(pair(i))
    env.run(until=horizon)
    return dict(tenant_GBs=tenant_bytes[0] / horizon, handoff_ns=float(np.mean(lat)) if lat else float("nan"),
                handoff_p99_ns=float(np.percentile(lat, 99)) if lat else float("nan"),
                device_reads=dev.stats["reads"], parked_max=dev.stats["parked_max"],
                table_full=dev.stats["table_full"], timeouts=dev.stats["timeouts"], messages=len(lat))


def rpc(p, protocol, service_ns, size=256, calls=60):
    """Client sends a request, server (already waiting) serves it and replies; returns mean RPC time."""
    env = simpy.Environment()
    dev = Device(env, p)
    times = []

    def run():
        for c in range(calls):
            req = (fresh(max(size // 64, 1)), fresh(1)[0])
            rsp = (fresh(max(size // 64, 1)), fresh(1)[0])
            server = env.process(consumer(env, dev, p, 1, req, c + 1, protocol))
            client = env.process(consumer(env, dev, p, 0, rsp, c + 1, protocol))
            yield env.timeout(3000 + jitter(p))
            t0 = env.now
            env.process(producer(env, dev, 0, req, c + 1, protocol))
            yield server
            yield env.timeout(service_ns)
            env.process(producer(env, dev, 1, rsp, c + 1, protocol))
            yield client
            times.append(env.now - t0)
    env.process(run())
    env.run()
    return float(np.mean(times[2:]))


def barrier(p, protocol, rounds=10, seed=0):
    """Hosts 1..H-1 arrive at random times and signal host 0, which then releases everyone.
    Returns the mean time from the last arrival until every host has seen the release."""
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    dev = Device(env, p)
    out = []
    n = p.hosts - 1

    def run():
        for r in range(rounds):
            arrive = [(fresh(1), fresh(1)[0]) for _ in range(n)]
            release = (fresh(1), fresh(1)[0])
            coord = [env.process(consumer(env, dev, p, 0, arrive[i], r + 1, protocol)) for i in range(n)]
            waits = [env.process(consumer(env, dev, p, 1 + i, release, r + 1, protocol)) for i in range(n)]
            yield env.timeout(3000)
            delays = rng.uniform(0, 2000, n)
            for i in range(n):
                env.process(_later(env, delays[i], producer(env, dev, 1 + i, arrive[i], r + 1, protocol)))
            t_last = env.now + delays.max()
            yield env.all_of(coord)
            env.process(producer(env, dev, 0, release, r + 1, protocol))
            yield env.all_of(waits)
            out.append(env.now - t_last)
    env.process(run())
    env.run()
    return float(np.mean(out[1:]))


def _later(env, d, gen):
    yield env.timeout(d)
    yield env.process(gen)


def calibrated(rtt, hosts=2, **kw):
    """Params whose idle load round trip equals rtt (link flight time absorbs the difference)."""
    p = Params(hosts=hosts, **kw)
    p.link_ns = 0.0
    p.link_ns = (rtt - p.idle_rtt()) / 2
    return p


def main():
    rows = []
    # device classes: expander 250 ns, multi-ported device 280 ns (Octopus 260-300), switch 600 ns
    # (Octopus 490-600, CCCL 658); poll overhead 50 ns (uncached load) or 1.4 us (clflush + fence,
    # cMPI: flushed access 2.2 us vs 0.79 us cached)
    for rtt, ovh in itertools.product((250.0, 280.0, 600.0), (50.0, 1400.0)):
        p = calibrated(rtt, poll_overhead=ovh)
        for size in (64, 1024, 4096, 16384):
            r = {pr: one_way(p, size, pr) for pr in PROTOCOLS}
            best = min(r[k] for k in PROTOCOLS if k != "parked")
            rows.append(dict(exp="one_way", rtt=rtt, poll_ovh=ovh, size=size, **r, speedup=best / r["parked"]))
        pp = {pr: ping_pong(p, 64, pr) for pr in PROTOCOLS}
        best = min(pp[k] for k in PROTOCOLS if k != "parked")
        rows.append(dict(exp="ping_pong", rtt=rtt, poll_ovh=ovh, size=64, **pp, speedup=best / pp["parked"]))
        for svc in (0.0, 500.0, 2000.0, 10000.0):
            r = {pr: rpc(p, pr, svc) for pr in PROTOCOLS}
            best = min(r[k] for k in PROTOCOLS if k != "parked")
            rows.append(dict(exp=f"rpc_{int(svc)}ns", rtt=rtt, poll_ovh=ovh, size=256, **r, speedup=best / r["parked"]))
        pb = calibrated(rtt, hosts=16, poll_overhead=ovh)
        r = {pr: barrier(pb, pr) for pr in PROTOCOLS}
        best = min(r[k] for k in PROTOCOLS if k != "parked")
        rows.append(dict(exp="barrier_16", rtt=rtt, poll_ovh=ovh, size=64, **r, speedup=best / r["parked"]))
        print(rtt, ovh, flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
    df.to_csv(OUT / "results_latency.csv", index=False)
    rows = []
    for w, pr in itertools.product((0, 16, 64, 128, 256, 512), ("poll_flag", "poll_lines", "parked")):
        p = calibrated(600.0, hosts=20)
        rows.append(dict(waiters=w, protocol=pr, **waiters_vs_tenant(p, w, pr)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "results_contention.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
