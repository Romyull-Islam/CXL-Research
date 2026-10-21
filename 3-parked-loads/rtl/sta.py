"""Static timing of the mapped controller (rtl/mapped.json) with the NanGate45 liberty tables.

Arrival times start at clock-to-Q of every flip-flop and propagate through the combinational cells in
topological order; each arc's delay and output slew come from the cell_rise/cell_fall and
rise/fall_transition tables, interpolated at the input slew and the output load (sum of fanout pin
capacitances; no wire load). The critical path is the largest arrival at a flip-flop D pin plus its
setup time. Rise and fall are not tracked separately: each arc takes the worse of the two.
  python rtl/sta.py
"""
import json
import re
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE / "lib" / "NangateOpenCellLibrary_typical.lib"


def parse_liberty(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S).replace("\\\n", " ")
    tok = re.findall(r'"[^"]*"|[{}();:,]|[^\s{}();:,"]+', text)
    pos = 0

    def group():
        nonlocal pos
        node = {"attrs": {}, "groups": []}
        while pos < len(tok):
            t = tok[pos]
            if t == "}":
                pos += 1
                return node
            name = t
            pos += 1
            if tok[pos] == ":":
                node["attrs"][name] = tok[pos + 1].strip('"')
                pos += 2
                if pos < len(tok) and tok[pos] == ";":
                    pos += 1
            elif tok[pos] == "(":
                args = []
                pos += 1
                while tok[pos] != ")":
                    if tok[pos] != ",":
                        args.append(tok[pos].strip('"'))
                    pos += 1
                pos += 1
                if tok[pos] == "{":
                    pos += 1
                    g = group()
                    g["type"], g["args"] = name, args
                    node["groups"].append(g)
                else:
                    node["attrs"].setdefault(name, args)
                    if tok[pos] == ";":
                        pos += 1
        return node
    return group()


def table(g, templates):
    def nums(s):
        return [float(x) for x in re.split(r"[,\s]+", " ".join(s).strip()) if x]
    a = dict(templates.get(g["args"][0] if g["args"] else "", {}), **g["attrs"])
    i1 = nums(a["index_1"]) if "index_1" in a else [0.0]   # scalar table
    i2 = nums(a["index_2"]) if "index_2" in a else [0.0]
    vals = nums(g["attrs"]["values"])
    rows = [vals[k * len(i2):(k + 1) * len(i2)] for k in range(len(i1))]
    return i1, i2, rows


def interp(t, x, y):
    i1, i2, rows = t

    def idx(a, v):
        k = min(max(bisect_right(a, v) - 1, 0), max(len(a) - 2, 0))
        return k
    def lerp(a, k, v):
        return 0.0 if len(a) == 1 else (v - a[k]) / (a[k + 1] - a[k])
    a, b = idx(i1, x), idx(i2, y)
    fa, fb = lerp(i1, a, x), lerp(i2, b, y)
    a1, b1 = min(a + 1, len(i1) - 1), min(b + 1, len(i2) - 1)
    v00, v01 = rows[a][b], rows[a][b1]
    v10, v11 = rows[a1][b], rows[a1][b1]
    return (v00 * (1 - fa) * (1 - fb) + v10 * fa * (1 - fb) + v01 * (1 - fa) * fb + v11 * fa * fb)


def load_cells():
    lib = parse_liberty(LIB.read_text())
    top = lib["groups"][0]["groups"]
    templates = {g["args"][0]: g["attrs"] for g in top if g["type"].endswith("_template")}
    cells = {}
    for c in top:
        if c["type"] != "cell":
            continue
        pins, dirs, arcs, setup = {}, {}, {}, None
        for p in c["groups"]:
            if p["type"] != "pin":
                continue
            name = p["args"][0]
            pins[name] = float(p["attrs"].get("capacitance", 0) or 0)
            dirs[name] = p["attrs"].get("direction", "input")
            for t in p["groups"]:
                if t["type"] != "timing":
                    continue
                tt = t["attrs"].get("timing_type", "combinational")
                tabs = {g["type"]: table(g, templates) for g in t["groups"]}
                if tt.startswith("setup"):
                    setup = tabs
                elif "cell_rise" in tabs or "cell_fall" in tabs:
                    arcs[(t["attrs"]["related_pin"], name)] = tabs
        cells[c["args"][0]] = dict(pins=pins, dirs=dirs, arcs=arcs, setup=setup, area=float(c["attrs"]["area"]))
    return cells


def arc_eval(tabs, slew, load):
    d = max(interp(tabs[k], slew, load) for k in ("cell_rise", "cell_fall") if k in tabs)
    s = max(interp(tabs[k], slew, load) for k in ("rise_transition", "fall_transition") if k in tabs)
    return d, s


def main():
    cells = load_cells()
    mod = json.loads((HERE / "mapped.json").read_text())["modules"]["parked_row"]
    insts = {n: c for n, c in mod["cells"].items() if c["type"] in cells}   # drops $scopeinfo markers
    drivers, sinks = {}, defaultdict(list)
    for n, c in insts.items():
        for pin, bits in c["connections"].items():
            for b in bits:
                if isinstance(b, str):
                    continue
                if cells[c["type"]]["dirs"][pin] == "output":
                    drivers[b] = (n, pin)
                else:
                    sinks[b].append((n, pin))
    load = {b: sum(cells[insts[n]["type"]]["pins"].get(p, 0) for n, p in s) for b, s in sinks.items()}
    arr, slew, prev = {}, {}, {}
    seq = {n for n, c in insts.items() if cells[c["type"]]["setup"] is not None}
    clk_slew = 0.02
    for n in seq:
        c = insts[n]
        for (rel, out), tabs in cells[c["type"]]["arcs"].items():
            for b in c["connections"].get(out, []):
                if isinstance(b, int):
                    d, s = arc_eval(tabs, clk_slew, load.get(b, 0))
                    if d > arr.get(b, -1):
                        arr[b], slew[b], prev[b] = d, s, (n, rel)
    comb = [n for n in insts if n not in seq]
    indeg = {n: 0 for n in comb}
    for n in comb:
        for pin, bits in insts[n]["connections"].items():
            if cells[insts[n]["type"]]["dirs"][pin] == "input":
                indeg[n] += sum(1 for b in bits if isinstance(b, int) and b in drivers and drivers[b][0] in indeg)
    ready = [n for n in comb if indeg[n] == 0]
    order = []
    while ready:
        n = ready.pop()
        order.append(n)
        for pin, bits in insts[n]["connections"].items():
            if cells[insts[n]["type"]]["dirs"][pin] != "output":
                continue
            for b in bits:
                for m, _ in sinks.get(b, []):
                    if m in indeg:
                        indeg[m] -= 1
                        if indeg[m] == 0:
                            ready.append(m)
    assert len(order) == len(comb), "combinational loop"
    for n in order:
        c = insts[n]
        for (rel, out), tabs in cells[c["type"]]["arcs"].items():
            src = [b for b in c["connections"].get(rel, []) if isinstance(b, int)]
            if not src:
                continue
            a0, s0 = arr.get(src[0], 0.0), slew.get(src[0], clk_slew)
            for b in c["connections"].get(out, []):
                if isinstance(b, int):
                    d, s = arc_eval(tabs, s0, load.get(b, 0))
                    if a0 + d > arr.get(b, -1):
                        arr[b], slew[b], prev[b] = a0 + d, s, (n, rel)
    worst, wend = 0.0, None
    for n in seq:
        c = insts[n]
        st = cells[c["type"]]["setup"]
        for b in c["connections"].get("D", []):
            if isinstance(b, int):
                t = arr.get(b, 0.0) + max(interp(st[k], slew.get(b, clk_slew), clk_slew)
                                          for k in ("rise_constraint", "fall_constraint") if k in st)
                if t > worst:
                    worst, wend = t, b
    depth, b = 0, wend
    while b in prev:
        n, rel = prev[b]
        depth += 1
        src = [x for x in insts[n]["connections"].get(rel, []) if isinstance(x, int)]
        if n in seq or not src:
            break
        b = src[0]
    area = sum(cells[c["type"]]["area"] for c in insts.values())
    ff = sum(cells[insts[n]["type"]]["area"] for n in seq)
    out = (f"cells {len(insts)} (flip-flops {len(seq)}), area {area:.0f} um2 (flip-flops {ff:.0f}, "
           f"logic {area - ff:.0f}), critical path {worst:.3f} ns over {depth} cells, fmax {1 / worst:.2f} GHz")
    print(out)
    (HERE / "timing.txt").write_text(out + "\n")


if __name__ == "__main__":
    main()
