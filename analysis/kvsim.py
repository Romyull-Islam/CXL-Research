"""Step-level simulator of one vLLM V1 instance under KV-cache pressure.

Scheduling follows vllm/v1/core/sched/scheduler.py (FCFS policy, commit 19c0bed, 2026-10-09):
- running requests are served in admission order; each gets its remaining prefill (chunked by the
  token budget) or one decode token;
- when a running request cannot get a block, the last running request is preempted: its blocks
  are freed, num_computed_tokens goes to 0 and it is put at the front of the waiting queue;
- waiting requests are admitted FCFS only in steps without preemption, only if the full input
  sequence fits (scheduler_reserve_full_isl=True) and while running < max_num_seqs; a resumed
  request's input is its prompt plus everything it generated.

Freed blocks keep their contents until reallocated (GPU prefix cache). vLLM frees a request's
blocks tail first and allocates from the head of the free queue, so a preempted request keeps a
cached prefix that shrinks as other requests allocate.

Preempted KV can also be kept in a second tier (host DRAM or CXL) with capacity tier_bytes and
GPU-to-tier bandwidth tier_bw per TP rank. Policies:
- "recompute": no tier (vLLM V1 default).
- "swap": copy a victim's blocks to the tier when it is preempted (blocking, as vLLM V0 swap
  mode) and load them back when it resumes; falls back to recompute if the tier is full.
- "lru": write-through of every newly computed full block with LRU eviction, as the vLLM
  OffloadingConnector CPU tier does; a resuming request loads the contiguous run of its blocks
  that follows its GPU-cached prefix.
Loads are asynchronous (the request waits, the batch does not), as in the V1 offloading
connector, and share one FIFO link.
"""
from collections import deque

import numpy as np

WAITING, RUNNING, LOADING, DONE = 0, 1, 2, 3


class Req:
    __slots__ = ("rid", "arrival", "prompt", "out", "gen", "comp", "hw", "blocks", "state",
                 "first_tok", "finish", "n_pre", "recomputed", "loaded", "seg", "group",
                 "tier_lo", "tier_hi", "stamps", "swapped", "load_done")

    def __init__(self, rid, arrival, prompt, out, group=-1):
        self.rid, self.arrival, self.prompt, self.out, self.group = rid, arrival, prompt, max(out, 1), group
        self.gen = 0            # output tokens sampled so far
        self.comp = 0           # tokens whose KV is in HBM
        self.hw = 0             # most tokens ever computed (recompute below this is waste)
        self.blocks = 0         # HBM blocks held
        self.state = WAITING
        self.first_tok = self.finish = None
        self.n_pre = 0
        self.recomputed = 0
        self.loaded = 0
        self.seg = None         # free-queue segment holding this request's cached prefix
        self.tier_lo = 0        # lru tier: blocks [tier_lo, tier_hi) are present
        self.tier_hi = 0
        self.stamps = None
        self.swapped = 0        # swap tier: tokens held
        self.load_done = 0.0

    @property
    def ntok(self):
        return self.prompt + self.gen


class FreeQueue:
    """Free block pool: segments appended at the tail, allocation consumes from the head."""

    def __init__(self, n):
        self.q = deque([[None, n]])
        self.n = n

    def free(self, owner, k):
        if k <= 0:
            return None
        seg = [owner, k]
        self.q.append(seg)
        self.n += k
        return seg

    def alloc(self, k):
        if k > self.n:
            raise RuntimeError(f"alloc {k} > free {self.n}")
        self.n -= k
        while k:
            seg = self.q[0]
            if seg[1] <= k:
                k -= seg[1]
                seg[1] = 0
                self.q.popleft()
                if seg[0] is not None and seg[0].seg is seg:
                    seg[0].seg = None
            else:
                seg[1] -= k
                k = 0

    def take(self, seg, k):
        """Reuse k cached blocks of a segment (prefix-cache hit)."""
        seg[1] -= k
        self.n -= k


class LRUTier:
    """Exact LRU over (request, block index) with lazy deletion.

    Within one request, blocks are stored in index order and a load touches a contiguous run in
    index order, so eviction inside a request always removes its lowest present block.
    """

    def __init__(self, cap):
        self.cap, self.used, self.clock = cap, 0, 0
        self.q = deque()

    def _touch(self, r, lo, hi):
        st = r.stamps
        for b in range(lo, hi):
            self.clock += 1
            st[b] = self.clock
            self.q.append((self.clock, r, b))

    def touch(self, r, lo, hi):
        lo, hi = max(lo, r.tier_lo), min(hi, r.tier_hi)
        if hi > lo:
            self._touch(r, lo, hi)

    def store(self, r, lo, hi):
        """Blocks [lo, hi) were just computed; insert those not present."""
        if r.stamps is None:
            r.stamps = {}
        if r.tier_hi == r.tier_lo or lo > r.tier_hi or lo < r.tier_lo:
            # a gap, or recompute below the present run (approximation: the run is re-stored)
            self.drop(r)
            r.tier_lo = r.tier_hi = lo
        start = max(lo, r.tier_hi)
        if hi <= start:
            return 0
        self._touch(r, start, hi)
        r.tier_hi = hi
        self.used += hi - start
        while self.used > self.cap:
            self._evict_one()
        return hi - start

    def drop(self, r):
        if r.stamps:
            self.used -= len(r.stamps)
            r.stamps.clear()
        r.tier_lo = r.tier_hi

    def _evict_one(self):
        while True:
            stamp, r, b = self.q.popleft()
            if r.stamps is not None and r.stamps.get(b) == stamp:
                break
        if b != r.tier_lo:
            raise AssertionError("out-of-order eviction inside a request")
        del r.stamps[b]
        self.used -= 1
        r.tier_lo += 1


class Sim:
    def __init__(self, inst, reqs, policy="recompute", tier_bytes=0.0, tier_bw=50e9,
                 max_num_seqs=1024, max_batched_tokens=8192, gpu_prefix_cache=True,
                 num_blocks=None, record_every=10.0):
        self.inst = inst
        self.bs = inst.block_size
        self.nblocks = num_blocks or inst.num_blocks
        self.reqs = sorted(reqs, key=lambda r: r.arrival)
        self.policy = policy
        self.tier_blocks = int(tier_bytes // inst.block_bytes) if policy != "recompute" else 0
        self.tier_bw = tier_bw
        self.max_seqs, self.budget = max_num_seqs, max_batched_tokens
        self.gpc = gpu_prefix_cache
        self.fq = FreeQueue(self.nblocks)
        self.lru = LRUTier(self.tier_blocks) if policy == "lru" else None
        self.swap_used = 0
        self.running, self.waiting, self.loading = [], deque(), []
        self.t = 0.0
        self.link_free = 0.0
        self.st = dict(steps=0, preemptions=0, recomputed_tokens=0, loaded_tokens=0, stored_blocks=0,
                       swap_out_bytes=0.0, load_bytes=0.0, recompute_time=0.0, tier_full_fallbacks=0,
                       busy_time=0.0, swap_stall=0.0, tier_peak_blocks=0, kv_peak_blocks=0,
                       gpu_cache_hit_tokens=0)
        self.record_every = record_every
        self.timeline = []
        self._next_rec = 0.0

    def nb(self, tokens):
        return -(-tokens // self.bs)

    def tier_used(self):
        if self.policy == "swap":
            return self.swap_used
        return self.lru.used if self.lru else 0

    def _free_req(self, r, cache):
        k, r.blocks = r.blocks, 0
        keep = cache and self.gpc
        seg = self.fq.free(r if keep else None, k)
        r.seg = seg if keep else None

    def _preempt(self, v):
        self.st["preemptions"] += 1
        v.n_pre += 1
        stall = 0.0
        if self.policy == "swap":
            need = v.blocks
            if self.swap_used + need <= self.tier_blocks:
                self.swap_used += need
                v.swapped = v.comp
                self.st["swap_out_bytes"] += need * self.inst.block_bytes
                stall = self.inst.kv_transfer_time(need * self.bs, self.tier_bw)
            else:
                self.st["tier_full_fallbacks"] += 1
        self._free_req(v, cache=not v.swapped)
        v.comp = 0
        v.state = WAITING
        self.waiting.appendleft(v)
        return stall

    def _admit_hits(self, r):
        """Tokens reusable on (re)admission: GPU-cached prefix, then the tier."""
        maxhit = r.ntok - 1
        hit = 0
        if r.seg is not None and r.seg[1] > 0:
            hit = min(r.seg[1], maxhit // self.bs) * self.bs
        load = 0
        if self.policy == "swap" and r.swapped:
            load = max(min(r.swapped, maxhit) - hit, 0)
        elif self.policy == "lru" and r.tier_hi > r.tier_lo:
            g = hit // self.bs
            if r.tier_lo <= g < r.tier_hi:
                load = max(min(r.tier_hi, maxhit // self.bs) * self.bs - hit, 0)
        return hit, load

    def run(self, t_end=None):
        reqs, n = self.reqs, len(self.reqs)
        i = 0
        while True:
            while i < n and reqs[i].arrival <= self.t:
                self.waiting.append(reqs[i])
                i += 1
            if self.loading:
                for r in [r for r in self.loading if r.load_done <= self.t]:
                    self.loading.remove(r)
                    r.state = RUNNING
                    self.running.append(r)
            if not self.running and not self.waiting and not self.loading:
                if i >= n:
                    break
                self.t = reqs[i].arrival
                continue
            if t_end is not None and self.t >= t_end:
                break
            if self._fast_forward(i):
                continue
            if self._step() == 0.0:
                nxt = [r.load_done for r in self.loading]
                if i < n:
                    nxt.append(reqs[i].arrival)
                if not nxt:
                    raise RuntimeError("deadlock: nothing schedulable and no pending event")
                self.t = max(self.t, min(nxt))
        return self

    def _record(self):
        if self.t >= self._next_rec:
            self.timeline.append((self.t, len(self.running), len(self.waiting), len(self.loading),
                                  self.nblocks - self.fq.n, self.tier_used()))
            self._next_rec = self.t + self.record_every

    def _peaks(self):
        used = self.nblocks - self.fq.n
        if used > self.st["kv_peak_blocks"]:
            self.st["kv_peak_blocks"] = used
        tu = self.tier_used()
        if tu > self.st["tier_peak_blocks"]:
            self.st["tier_peak_blocks"] = tu

    def _step(self):
        bs, budget, running = self.bs, self.budget, self.running
        sched = []
        preempted = False
        stall = 0.0
        idx = 0
        while idx < len(running) and budget > 0:
            r = running[idx]
            new = min(r.ntok - r.comp, budget)
            need = self.nb(r.comp + new) - r.blocks
            ok = True
            while need > self.fq.n:
                v = running.pop()
                stall += self._preempt(v)
                preempted = True
                if v is r:
                    ok = False
                    break
            if not ok:
                break
            self.fq.alloc(need)
            r.blocks += need
            sched.append((r, new))
            budget -= new
            idx += 1
        if not preempted:
            while budget > 0 and self.waiting and len(running) < self.max_seqs:
                r = self.waiting[0]
                hit, load = self._admit_hits(r) if r.comp == 0 else (0, 0)
                # cached blocks sit in the free pool, so the whole sequence must fit in it
                if self.nb(r.ntok) > self.fq.n:
                    break
                self.waiting.popleft()
                if hit:
                    self.fq.take(r.seg, hit // bs)
                    r.blocks += hit // bs
                    r.comp = hit
                    self.st["gpu_cache_hit_tokens"] += hit
                if r.seg is not None:
                    r.seg[0] = None
                    r.seg = None
                if self.policy == "swap" and r.swapped:
                    self.swap_used -= self.nb(r.swapped)
                    r.swapped = 0
                if load:
                    kb = self.nb(hit + load) - r.blocks
                    self.fq.alloc(kb)
                    r.blocks += kb
                    if self.lru is not None:
                        self.lru.touch(r, hit // bs, (hit + load) // bs)
                    nbytes = load * self.inst.model.kv_bytes_per_token
                    start = max(self.t, self.link_free)
                    r.load_done = start + self.inst.kv_transfer_time(load, self.tier_bw)
                    self.link_free = r.load_done
                    r.comp = hit + load
                    r.loaded += load
                    self.st["loaded_tokens"] += load
                    self.st["load_bytes"] += nbytes
                    r.state = LOADING
                    self.loading.append(r)
                    continue
                new = min(r.ntok - r.comp, budget)
                need = self.nb(r.comp + new) - r.blocks
                self.fq.alloc(need)
                r.blocks += need
                r.state = RUNNING
                running.append(r)
                sched.append((r, new))
                budget -= new
        if not sched:
            return 0.0
        n_dec, dsum, chunks, base, rtok = 0, 0, [], [], 0
        for r, k in sched:
            if k == 1 and r.comp == r.ntok - 1:
                n_dec += 1
                dsum += r.comp + 1
                continue
            samples = r.comp + k == r.ntok
            chunks.append((k, r.comp, samples))
            rec = min(r.comp + k, r.hw) - r.comp
            if rec > 0:
                # counterfactual without the preemption: only the tokens above hw are computed
                rtok += rec
                r.recomputed += rec
                if k > rec:
                    base.append((k - rec, r.comp + rec, samples))
            else:
                base.append((k, r.comp, samples))
        dt = self.inst.step_time(n_dec, dsum, chunks)
        if rtok:
            self.st["recomputed_tokens"] += rtok
            self.st["recompute_time"] += dt - self.inst.step_time(n_dec, dsum, base)
        dt += stall
        self.st["swap_stall"] += stall
        self.t += dt
        self.st["steps"] += 1
        self.st["busy_time"] += dt
        for r, k in sched:
            lo = r.comp // bs
            r.comp += k
            if r.comp > r.hw:
                r.hw = r.comp
            if self.lru is not None and r.comp // bs > lo:
                self.st["stored_blocks"] += self.lru.store(r, lo, r.comp // bs)
            if r.comp == r.ntok:
                r.gen += 1
                if r.first_tok is None:
                    r.first_tok = self.t
                if r.gen >= r.out:
                    r.finish = self.t
                    r.state = DONE
                    running.remove(r)
                    self._free_req(r, cache=False)
        self._peaks()
        self._record()
        return dt

    def _fast_forward(self, i):
        """Advance k decode-only steps at once when none of them can preempt, admit, finish a
        request, or cross an arrival, a load completion or a timeline sample. Equivalent to k
        calls of _step (checked in test_kvsim.py)."""
        running = self.running
        if not running or self.lru is not None:
            return False
        for r in running:
            if r.ntok - r.comp != 1:
                return False
        nr = len(running)
        if nr > self.budget:
            return False
        if self.waiting and nr < self.max_seqs and nr < self.budget:
            if self.nb(self.waiting[0].ntok) <= self.fq.n:
                return False
        bs = self.bs
        comp = np.fromiter((r.comp for r in running), dtype=np.int64, count=nr)
        left = np.fromiter((r.out - r.gen for r in running), dtype=np.int64, count=nr)
        held = np.fromiter((r.blocks for r in running), dtype=np.int64, count=nr)
        k = int(left.min()) - 1          # the finishing step is left to _step
        if k < 2:
            return False
        free = self.fq.n

        def extra(kk):
            return int((-(-(comp + kk) // bs) - held).sum())
        if extra(k) > free:
            lo, hi = 0, k
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if extra(mid) <= free:
                    lo = mid
                else:
                    hi = mid - 1
            k = lo
        if k < 2:
            return False
        ctx = comp.sum() + nr + nr * np.arange(k, dtype=np.float64)
        cum = np.cumsum(self.inst.decode_step_times(nr, ctx))
        events = [self._next_rec] + [r.load_done for r in self.loading]
        if i < len(self.reqs):
            events.append(self.reqs[i].arrival)
        j = int(np.searchsorted(cum, min(events) - self.t, side="left"))
        k = min(k, j + 1)
        if k < 2:
            return False
        dt = float(cum[k - 1])
        self.t += dt
        self.st["steps"] += k
        self.st["busy_time"] += dt
        need = -(-(comp + k) // bs) - held
        self.fq.alloc(int(need.sum()))
        for r, nd in zip(running, need.tolist()):
            r.blocks += nd
            r.comp += k
            r.gen += k
            if r.comp > r.hw:
                r.hw = r.comp
        self._peaks()
        self._record()
        return True


def summarize(sim):
    done = [r for r in sim.reqs if r.finish is not None]
    st = dict(sim.st)
    out_tok = sum(r.out for r in done)
    span = (max(r.finish for r in done) - min(r.arrival for r in done)) if done else 0.0
    st.update(
        n_done=len(done), n_total=len(sim.reqs), makespan=sim.t, out_tokens=out_tok,
        throughput=out_tok / span if span else 0.0,
        recompute_frac=st["recompute_time"] / st["busy_time"] if st["busy_time"] else 0.0,
        recomputed_per_out=st["recomputed_tokens"] / out_tok if out_tok else 0.0,
        frac_preempted=float(np.mean([r.n_pre > 0 for r in done])) if done else 0.0,
        tier_peak_gb=st["tier_peak_blocks"] * sim.inst.block_bytes / 1e9,
        hbm_kv_gb=sim.nblocks * sim.inst.block_bytes / 1e9,
    )
    if done:
        ttft = np.array([r.first_tok - r.arrival for r in done])
        e2e = np.array([r.finish - r.arrival for r in done])
        tpot = np.array([(r.finish - r.first_tok) / max(r.out - 1, 1) for r in done])
        st.update(ttft_p50=float(np.median(ttft)), ttft_p99=float(np.percentile(ttft, 99)),
                  e2e_p50=float(np.median(e2e)), e2e_p99=float(np.percentile(e2e, 99)),
                  tpot_p50=float(np.median(tpot)), tpot_p99=float(np.percentile(tpot, 99)))
    return st
