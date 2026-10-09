"""Page-mapped flash translation layer with garbage collection, to measure write amplification.

The unit is one host write unit (here one KV block). An erase block holds B units. Physical space
is the logical space times (1 + op), rounded up to whole erase blocks. Host writes append to the
host write block; overwriting a logical unit invalidates its old copy. When free erase blocks fall
to `reserve`, garbage collection picks a victim among full erase blocks (fewest valid units for
"greedy", oldest for "fifo"), copies its valid units and erases it. As in MQSim and most SSD
firmware, relocated units go to a separate GC write block (gc_stream=True), which keeps data that
has already survived one collection away from fresh host writes; gc_stream=False mixes them in one
block. Write amplification = (host + GC writes) / host writes, counted after a warm-up.

Checked in test_ftl.py against sequential overwrite (WAF 1) and the closed forms for FIFO cleaning
(Desnoyers, SYSTOR'12, Eq. 4) and greedy cleaning (Eq. 11-12) under uniform random writes, and
against MQSim (FAST'18) on the KV write streams (results/mqsim_crosscheck.csv).
"""
from collections import deque

import numpy as np


class FTL:
    def __init__(self, logical_units, op, units_per_eblock, policy="greedy", reserve=2, gc_stream=True):
        B = max(int(units_per_eblock), 1)
        self.B, self.L, self.policy, self.reserve = B, int(logical_units), policy, reserve
        self.gc_stream = gc_stream
        self.n_eb = max(int(np.ceil(self.L * (1 + op) / B)), self.L // B + reserve + 3)
        self.l2p = np.full(self.L, -1, dtype=np.int64)
        self.p2l = np.full(self.n_eb * B, -1, dtype=np.int64)
        self.valid = np.zeros(self.n_eb, dtype=np.int64)
        self.full = np.zeros(self.n_eb, dtype=bool)
        self.seq = np.zeros(self.n_eb, dtype=np.int64)      # when the block was closed (fifo)
        first = 2 if gc_stream else 1
        self.free = deque(range(first, self.n_eb))
        # open blocks: [block, write pointer] for host writes and, if separate, for GC writes
        self.open = {"host": [0, 0], "gc": [1, 0] if gc_stream else None}
        self.closed = 0
        self.host = self.gc = self.erases = 0

    @property
    def op_effective(self):
        """Spare space that can hold data: reserved free blocks and open blocks cannot."""
        n_open = 2 if self.gc_stream else 1
        return (self.n_eb - self.reserve - n_open) * self.B / self.L - 1

    def _append(self, lu, stream="host"):
        o = self.open[stream if self.gc_stream else "host"]
        if o[1] == self.B:
            self.full[o[0]] = True
            self.closed += 1
            self.seq[o[0]] = self.closed
            o[0], o[1] = self.free.popleft(), 0
        p = o[0] * self.B + o[1]
        self.p2l[p] = lu
        self.l2p[lu] = p
        self.valid[o[0]] += 1
        o[1] += 1

    def _invalidate(self, lu):
        p = self.l2p[lu]
        if p >= 0:
            self.p2l[p] = -1
            self.valid[p // self.B] -= 1
            self.l2p[lu] = -1

    def _collect(self):
        while len(self.free) <= self.reserve:
            if self.policy == "greedy":
                v = int(np.where(self.full, self.valid, np.iinfo(np.int64).max).argmin())
            else:
                v = int(np.where(self.full, self.seq, np.iinfo(np.int64).max).argmin())
            base = v * self.B
            for p in range(base, base + self.B):
                lu = self.p2l[p]
                if lu >= 0:
                    self.p2l[p] = -1
                    self.valid[v] -= 1
                    self._append(lu, "gc")
                    self.gc += 1
            self.full[v] = False
            self.valid[v] = 0
            self.free.append(v)
            self.erases += 1

    def write(self, lu):
        self._invalidate(lu)
        self._collect()
        self._append(lu)
        self.host += 1

    def trim(self, lu):
        self._invalidate(lu)

    def run(self, writes, warmup=None):
        """Feed a stream of logical writes; returns WAF over the writes after `warmup`."""
        warmup = 2 * self.n_eb * self.B if warmup is None else warmup
        h0 = g0 = 0
        for i, lu in enumerate(writes.tolist() if hasattr(writes, "tolist") else writes):
            if i == warmup:
                h0, g0 = self.host, self.gc
            self.write(lu)
        h, g = self.host - h0, self.gc - g0
        return dict(waf=(h + g) / h if h else float("nan"), host=h, gc=g, measured=len(writes) > warmup)


def fifo_waf(op):
    """Closed form for FIFO (LRU) cleaning under uniform random writes: Desnoyers, SYSTOR'12,
    Eq. 4, A = alpha / (alpha + W(-alpha e^-alpha)) with alpha = 1 + op."""
    from scipy.special import lambertw
    a = 1 + op
    return a / (a + lambertw(-a * np.exp(-a)).real)


def greedy_waf(op, units_per_eblock):
    """Desnoyers, SYSTOR'12, Eq. 11-12: greedy cleaning under uniform random writes, Np pages per
    block, alpha = 1 + op (no free-list correction)."""
    from scipy.special import lambertw
    a, n = 1 + op, units_per_eblock
    c = 1 + 1 / (2 * n)
    x0 = 0.5 - (n / a) * lambertw(-c * a * np.exp(-c * a)).real
    return n / (n - (x0 - 1))
