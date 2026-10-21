"""Parked-table controller logic (one set of an 8-way table), in Amaranth.

The table itself is SRAM (its cost is in cacti/); this is the logic around one row: given an
operation and the row read from SRAM, it computes the immediate response, the waiters to wake and the
row to write back. Operations:
  READ   a read of line `tag` from `port` with CXL tag `ctag`; `full` is the line's full/empty bit.
         Answered at once with data if the line is full or the read is an ownership read (R5), and
         with not-yet if the set has no free way (degrades to polling); otherwise parked with
         deadline now + TAU (R1: the request leaves the queue). A consuming read of a full line
         empties it.
  WRITE  a write of line `tag`: wakes the parked reads of that line, oldest first; non-consuming
         reads all wake, a consuming write window wakes only the oldest consuming read (and the
         line stays empty), as in the paper's two address windows. Never blocked (R2).
  SCAN   the timeout scanner: every entry whose deadline has passed is answered not-yet (R3, R4).
  DROP   host `port` failed or was reset: its entries are freed without a response.
Entry: valid, line tag, requester port, CXL tag, deadline, age sequence number, consuming (77 bits,
under the 16 B of the paper).

  python rtl/parked_table.py test       random operations against a Python reference
  python rtl/parked_table.py synth      Verilog, then Yosys/ABC on NanGate45 -> rtl/synth.txt
"""
import random
import subprocess
import sys
from pathlib import Path

from amaranth import Cat, Const, Elaboratable, Module, Signal
from amaranth.back import verilog

HERE = Path(__file__).resolve().parent
WAYS, TAGW, PORTW, CTAGW, TW, SEQW = 8, 31, 4, 16, 8, 16
TAU = 25                         # microsecond ticks: half the 50 us minimum Root Port timeout
READ, WRITE, SCAN, DROP = 0, 1, 2, 3
FIELDS = [("valid", 1), ("tag", TAGW), ("port", PORTW), ("ctag", CTAGW), ("dl", TW), ("seq", SEQW),
          ("cons", 1)]
EW = sum(w for _, w in FIELDS)


def unpack(row, i):
    out, off = {}, i * EW
    for name, w in FIELDS:
        out[name] = row[off:off + w]
        off += w
    return out


class ParkedRow(Elaboratable):
    def __init__(self):
        self.op = Signal(2)
        self.tag = Signal(TAGW)
        self.port = Signal(PORTW)
        self.ctag = Signal(CTAGW)
        self.cons = Signal()          # consuming window
        self.own = Signal()           # read marked as a read for ownership (R5)
        self.full = Signal()          # full/empty bit of the line
        self.now = Signal(TW)
        self.seq = Signal(SEQW)       # global arrival counter, for FIFO wake-up
        self.row = Signal(WAYS * EW)
        self.row_out = Signal(WAYS * EW)
        self.resp_valid = Signal()    # immediate response to a READ
        self.resp_data = Signal()     # 1: data, 0: not-yet
        self.full_out = Signal()      # new full/empty bit
        self.wake = Signal(WAYS)      # entries answered with data (WRITE)
        self.expire = Signal(WAYS)    # entries answered not-yet (SCAN)

    def elaborate(self, platform):
        m = Module()
        e = [unpack(self.row, i) for i in range(WAYS)]
        match = Signal(WAYS)
        free = Signal(WAYS)
        expired = Signal(WAYS)
        mine = Signal(WAYS)
        for i in range(WAYS):
            m.d.comb += [
                match[i].eq(e[i]["valid"] & (e[i]["tag"] == self.tag)),
                free[i].eq(~e[i]["valid"]),
                # deadline passed: (now - dl) mod 2^TW in [0, 2^(TW-1))
                expired[i].eq(e[i]["valid"] & ~(self.now - e[i]["dl"])[TW - 1]),
                mine[i].eq(e[i]["valid"] & (e[i]["port"] == self.port)),
            ]
        # first free way
        alloc = Signal(WAYS)
        taken = Const(0, 1)
        for i in range(WAYS):
            m.d.comb += alloc[i].eq(free[i] & ~taken)
            taken = taken | free[i]
        # oldest consuming match: entry i wins if no other consuming match is older (wrap-around)
        cm = Signal(WAYS)
        oldest = Signal(WAYS)
        for i in range(WAYS):
            m.d.comb += cm[i].eq(match[i] & e[i]["cons"])
        for i in range(WAYS):
            older = Const(0, 1)
            for j in range(WAYS):
                if j == i:
                    continue
                d = Signal(SEQW, name=f"d_{i}_{j}")
                m.d.comb += d.eq(e[j]["seq"] - e[i]["seq"])
                # j older than i, ties broken by way index
                j_older = d[SEQW - 1] if j > i else (d[SEQW - 1] | (d == 0))
                older = older | (cm[j] & j_older)
            m.d.comb += oldest[i].eq(cm[i] & ~older)

        new = Signal(WAYS * EW)
        m.d.comb += [new.eq(self.row), self.full_out.eq(self.full)]
        clear = Signal(WAYS)
        entry = Cat(Const(1, 1), self.tag, self.port, self.ctag, (self.now + TAU)[:TW], self.seq, self.cons)
        with m.Switch(self.op):
            with m.Case(READ):
                with m.If(self.full | self.own):
                    m.d.comb += [self.resp_valid.eq(1), self.resp_data.eq(1)]
                    with m.If(self.cons & ~self.own):
                        m.d.comb += self.full_out.eq(0)
                with m.Elif(free == 0):
                    m.d.comb += [self.resp_valid.eq(1), self.resp_data.eq(0)]
                with m.Else():
                    for i in range(WAYS):
                        with m.If(alloc[i]):
                            m.d.comb += new[i * EW:(i + 1) * EW].eq(entry)
            with m.Case(WRITE):
                with m.If(self.cons):
                    m.d.comb += [self.full_out.eq(0)]
                    m.d.comb += clear.eq(oldest)
                with m.Else():
                    m.d.comb += [self.full_out.eq(1), clear.eq(match & ~cm)]
                    # a non-consuming write fills the line: one consuming waiter takes it
                    with m.If(oldest != 0):
                        m.d.comb += [clear.eq((match & ~cm) | oldest), self.full_out.eq(0)]
                m.d.comb += self.wake.eq(clear)
            with m.Case(SCAN):
                m.d.comb += [clear.eq(expired), self.expire.eq(expired)]
            with m.Case(DROP):
                m.d.comb += clear.eq(mine)
        with m.If(self.op != READ):
            for i in range(WAYS):
                with m.If(clear[i]):
                    m.d.comb += new[i * EW].eq(0)
        m.d.comb += self.row_out.eq(new)
        return m


# Python reference
def ref(op, row, tag, port, ctag, cons, own, full, now, seq):
    row = [dict(x) for x in row]
    resp, full_out, wake, expire = None, full, [0] * WAYS, [0] * WAYS
    if op == READ:
        if full or own:
            resp = 1
            if cons and not own:
                full_out = 0
        elif all(x["valid"] for x in row):
            resp = 0
        else:
            i = next(k for k, x in enumerate(row) if not x["valid"])
            row[i] = dict(valid=1, tag=tag, port=port, ctag=ctag, dl=(now + TAU) % 2**TW, seq=seq, cons=cons)
    elif op == WRITE:
        match = [x["valid"] and x["tag"] == tag for x in row]
        cmatch = [k for k in range(WAYS) if match[k] and row[k]["cons"]]
        old = None
        if cmatch:
            old = min(cmatch, key=lambda k: ((row[k]["seq"] - row[cmatch[0]]["seq"]) % 2**SEQW
                                             if (row[k]["seq"] - row[cmatch[0]]["seq"]) % 2**SEQW < 2**(SEQW - 1)
                                             else (row[k]["seq"] - row[cmatch[0]]["seq"]) % 2**SEQW - 2**SEQW, k))
        if cons:
            full_out = 0
            if old is not None:
                wake[old] = 1
        else:
            full_out = 1
            for k in range(WAYS):
                if match[k] and not row[k]["cons"]:
                    wake[k] = 1
            if old is not None:
                wake[old] = 1
                full_out = 0
        for k in range(WAYS):
            if wake[k]:
                row[k]["valid"] = 0
    elif op == SCAN:
        for k, x in enumerate(row):
            if x["valid"] and ((now - x["dl"]) % 2**TW) < 2**(TW - 1):
                expire[k] = 1
                x["valid"] = 0
    else:
        for x in row:
            if x["valid"] and x["port"] == port:
                x["valid"] = 0
    return row, resp, full_out, wake, expire


def pack(row):
    v, off = 0, 0
    for x in row:
        for name, w in FIELDS:
            v |= (x[name] & (2**w - 1)) << off
            off += w
    return v


def test(n=4000):
    from amaranth.sim import Simulator
    dut = ParkedRow()
    sim = Simulator(dut)
    rng = random.Random(1)

    async def bench(ctx):
        row = [dict(valid=0, tag=0, port=0, ctag=0, dl=0, seq=0, cons=0) for _ in range(WAYS)]
        now, seq, full, bad = 0, 0, 0, 0
        for _ in range(n):
            op = rng.choice([READ, READ, READ, WRITE, SCAN, DROP])
            tag, port = rng.randrange(3), rng.randrange(4)     # few tags and ports: many collisions
            ctag, cons, own = rng.randrange(2**CTAGW), rng.randrange(2), int(rng.random() < 0.1)
            now = (now + rng.randrange(6)) % 2**TW
            seq = (seq + 1) % 2**SEQW
            for sig, val in ((dut.op, op), (dut.tag, tag), (dut.port, port), (dut.ctag, ctag),
                             (dut.cons, cons), (dut.own, own), (dut.full, full), (dut.now, now),
                             (dut.seq, seq), (dut.row, pack(row))):
                ctx.set(sig, val)
            exp_row, resp, full_out, wake, expire = ref(op, row, tag, port, ctag, cons, own, full, now, seq)
            got = (ctx.get(dut.row_out), ctx.get(dut.resp_valid), ctx.get(dut.resp_data),
                   ctx.get(dut.full_out), ctx.get(dut.wake), ctx.get(dut.expire))
            want = (pack(exp_row), int(resp is not None), resp or 0, full_out,
                    sum(b << k for k, b in enumerate(wake)), sum(b << k for k, b in enumerate(expire)))
            if got != want:
                bad += 1
                if bad < 5:
                    print("mismatch op", op, got[1:], want[1:])
            row, full = exp_row, full_out
        print(f"{n} random operations, {bad} mismatches")
        assert bad == 0

    sim.add_testbench(bench)
    sim.run()


def synth():
    """Registered inputs and outputs around the logic, so the timing is register to register."""
    import os
    ybin = Path(sys.executable).parent / "yowasp-yosys"
    os.environ.setdefault("YOSYS", str(ybin))

    class Top(Elaboratable):
        def __init__(self):
            self.i = Signal(2 + TAGW + PORTW + CTAGW + 3 + TW + SEQW + WAYS * EW)
            self.o = Signal(WAYS * EW + 3 + 2 * WAYS)

        def elaborate(self, platform):
            m = Module()
            m.submodules.core = core = ParkedRow()
            ins = [core.op, core.tag, core.port, core.ctag, core.cons, core.own, core.full, core.now,
                   core.seq, core.row]
            outs = [core.row_out, core.resp_valid, core.resp_data, core.full_out, core.wake, core.expire]
            ri = Signal.like(self.i)
            m.d.sync += [ri.eq(self.i), self.o.eq(Cat(*outs))]
            m.d.comb += Cat(*ins).eq(ri)
            return m
    top = Top()
    v = HERE / "parked_row.v"
    v.write_text(verilog.convert(top, name="parked_row", ports=[top.i, top.o]))
    lib = HERE / "lib" / "NangateOpenCellLibrary_typical.lib"
    # paths are absolute: the WebAssembly Yosys does not see relative subdirectories
    (HERE / "synth.ys").write_text(
        f"read_verilog {v}\nsynth -top parked_row -flatten -noabc\ndfflibmap -liberty {lib}\n"
        f"abc -liberty {lib} -D 1000 -script {HERE / 'abc.script'}\nopt_clean\n"
        f"tee -o {HERE / 'stat.txt'} stat -liberty {lib}\nwrite_json {HERE / 'mapped.json'}\n")
    p = subprocess.run([str(ybin), "-s", str(HERE / "synth.ys")], capture_output=True, text=True)
    (HERE / "synth.log").write_text(p.stdout + p.stderr)
    if p.returncode:
        sys.exit("yosys failed, see rtl/synth.log")
    subprocess.run([sys.executable, str(HERE / "sta.py")], check=True)


if __name__ == "__main__":
    {"test": test, "synth": synth}[sys.argv[1]]()
