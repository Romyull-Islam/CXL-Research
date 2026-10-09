"""Discrete-event model of hosts sharing one CXL memory device, with and without parked loads.

Built on SimPy. Every host has a link to the device with a one-way latency and a bandwidth in each
direction; the device has one shared ingress and one shared egress port and a set of DRAM banks.
Messages are serialized on the sender's link, cross the link, and are serialized again on the
device port, so many polling hosts contend for the device the way they do on real hardware.

Per 64 B line the device keeps the data version and, for synchronization lines, a full/empty
state. A parked load to an EMPTY line waits in the device's parked table and is answered when a
write fills the line, or answered EMPTY when it reaches the timeout or the table is full. Writes
are acknowledged (an NDR completion) so a fence waits for all of a core's writes.

Units: ns and bytes; bandwidths in bytes/ns (= GB/s).
"""
from dataclasses import dataclass, field

import simpy

LINE = 64
HDR = 16


@dataclass
class Params:
    hosts: int = 2
    link_ns: float = 100.0          # one-way flight time host <-> device
    host_bw: float = 32.0           # per host link, each direction (x8 Gen5 class)
    dev_bw: float = 64.0            # device port, each direction
    mem_ns: float = 45.0            # DRAM access at the device
    banks: int = 32
    poll_overhead: float = 50.0     # clflushopt + fence per poll iteration
    mlp: int = 48                   # line requests one core keeps in flight
    table: int = 4096               # parked-read entries in the device
    park_timeout: float = 25_000.0  # half of the shortest CXL.mem Root Port timeout (Range A, 50 us)
    bi_ns: float = 0.0              # extra device time to issue a back-invalidate

    def idle_rtt(self):
        return (2 * self.link_ns + self.mem_ns + HDR / self.host_bw + HDR / self.dev_bw
                + (LINE + HDR) / self.dev_bw + (LINE + HDR) / self.host_bw)


@dataclass
class Line:
    version: int = 0
    full: bool = False
    parked: list = field(default_factory=list)


class Device:
    def __init__(self, env, p):
        self.env, self.p = env, p
        self.ingress = simpy.Resource(env, 1)
        self.egress = simpy.Resource(env, 1)
        self.banks = simpy.Resource(env, p.banks)
        self.up = [simpy.Resource(env, 1) for _ in range(p.hosts)]
        self.down = [simpy.Resource(env, 1) for _ in range(p.hosts)]
        self.lines = {}
        self.parked_now = 0
        self.stats = dict(reads=0, writes=0, parked=0, timeouts=0, table_full=0, egress_bytes=0, ingress_bytes=0)
        self.bi_waiters = {}        # line -> list of (host, event) holding a coherent copy

    def line(self, a):
        if a not in self.lines:
            self.lines[a] = Line()
        return self.lines[a]

    def _ser(self, res, nbytes, bw):
        with res.request() as r:
            yield r
            yield self.env.timeout(nbytes / bw)

    def _to_dev(self, h, nbytes):
        yield from self._ser(self.up[h], nbytes, self.p.host_bw)
        yield self.env.timeout(self.p.link_ns)
        yield from self._ser(self.ingress, nbytes, self.p.dev_bw)
        self.stats["ingress_bytes"] += nbytes

    def _to_host(self, h, nbytes):
        yield from self._ser(self.egress, nbytes, self.p.dev_bw)
        self.stats["egress_bytes"] += nbytes
        yield self.env.timeout(self.p.link_ns)
        yield from self._ser(self.down[h], nbytes, self.p.host_bw)

    def _mem(self):
        with self.banks.request() as r:
            yield r
            yield self.env.timeout(self.p.mem_ns)

    def read(self, h, a, park=False, consume=False):
        """Returns (version, full). A parked read of an EMPTY line returns when it fills."""
        yield from self._to_dev(h, HDR)
        yield from self._mem()
        self.stats["reads"] += 1
        ln = self.line(a)
        if park and not ln.full:
            if self.parked_now >= self.p.table:
                self.stats["table_full"] += 1
            else:
                self.stats["parked"] += 1
                self.parked_now += 1
                ev = self.env.event()
                ln.parked.append(ev)
                got = yield ev | self.env.timeout(self.p.park_timeout)
                self.parked_now -= 1
                if ev not in got:
                    ln.parked.remove(ev)
                    self.stats["timeouts"] += 1
        version, full = ln.version, ln.full
        if consume and full:
            ln.full = False
        yield from self._to_host(h, LINE + HDR)
        return version, full

    def write(self, h, a, version, fill=True):
        """A store reaching the device; returns when the completion is back at the host."""
        yield from self._to_dev(h, LINE + HDR)
        yield from self._mem()
        self.stats["writes"] += 1
        ln = self.line(a)
        ln.version = version
        if fill:
            ln.full = True
            for ev in ln.parked:
                ev.succeed()
            ln.parked = []
        for hh, ev in self.bi_waiters.pop(a, []):
            self.env.process(self._bi(hh, ev))
        yield from self._to_host(h, HDR)

    def _bi(self, h, ev):
        """Back-invalidate snoop to a host holding the line in a coherent region."""
        yield self.env.timeout(self.p.bi_ns)
        yield from self._to_host(h, HDR)
        ev.succeed()

    def watch(self, h, a):
        """A host caches line a coherently; the event fires when a write invalidates it."""
        ev = self.env.event()
        self.bi_waiters.setdefault(a, []).append((h, ev))
        return ev


def read_lines(env, dev, h, addrs, p, park=False):
    """Read many lines with at most p.mlp in flight; returns the list of (version, full)."""
    out = [None] * len(addrs)
    pending = list(enumerate(addrs))
    running = {}
    while pending or running:
        while pending and len(running) < p.mlp:
            i, a = pending.pop(0)
            running[i] = env.process(dev.read(h, a, park=park))
        done = yield env.any_of(list(running.values()))
        for i, proc in list(running.items()):
            if proc in done:
                out[i] = done[proc]
                del running[i]
    return out


def write_lines(env, dev, h, addrs, version, fill=True):
    """Posted stores from one core; returns the processes so a fence can wait on them."""
    return [env.process(dev.write(h, a, version, fill)) for a in addrs]
