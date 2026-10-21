"""gem5 against the real machine for the software protocols (flush, lines). Native numbers come from
bench/sync_bench.c built with -DNATIVE on a two-socket EPYC 7742 (hosts on one socket, the shared
memory on the other; results/native_*.log). Step 1 finds the link delay at which gem5's flush round
trip equals the native one; step 2 runs oneway and rpc at that delay.
  python calibrate_native.py rtt          link delays 150 to 300 ns -> results/cal_rtt.csv
  python calibrate_native.py run DELAY     oneway and rpc, flush and lines -> results/cal_gem5.csv
  python calibrate_native.py hw DELAY      the same for uc, parked, wbpark -> results/cal_gem5_hw.csv
  BENCH=bench/sync_bench_cal python calibrate_native.py wb DELAY   writer-side write-back -> results/cal_wb.csv
  python calibrate_native.py report        native medians beside gem5 -> results/cal_summary.csv

A coherent reader takes a fresh line from the writer's cache, which a host sharing CXL memory cannot:
there the line is visible only after the writer's write-back. The report therefore also gives the
native latency plus one native write-back per message direction ("adjusted")."""
import csv
import itertools
import sys
from multiprocessing import Pool

from run_gem5 import OUT, WORKERS, gem5


def rtt(delay):
    res, _ = gem5(f"cal_rtt_{delay}", ["rtt - - 2000"], "uc", delay)
    r = next(x for x in res if x["test"] == "rtt")
    return dict(link_delay=delay, uc_ns=float(r["uc_ns"]), flush_ns=float(r["flush_ns"]))


def case(c):
    delay, test, proto, arg = c
    if test == "oneway":
        res, _ = gem5(f"cal_ow_{delay}_{proto}_{arg}", [f"oneway consumer {proto} 30 {arg}",
                                                         f"oneway producer {proto} 30 {arg}"], proto, delay)
        r = next(x for x in res if x["test"] == "oneway")
        return dict(link_delay=delay, test=test, protocol=proto, arg=arg, ns=float(r["lat_ns"]))
    res, _ = gem5(f"cal_rpc_{delay}_{proto}_{arg}", [f"rpc client {proto} 30 {arg}",
                                                      f"rpc server {proto} 30 {arg}"], proto, delay)
    r = next(x for x in res if x["test"] == "rpc")
    return dict(link_delay=delay, test=test, protocol=proto, arg=arg, ns=float(r["rtt_ns"]))


def write(name, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r, flush=True)


def report():
    """Median of the native repetitions beside gem5 at the matched delay; the parked speedups are
    computed against the gem5 baselines (as in the paper) and against the native ones."""
    import re
    from statistics import median
    nat = {}
    for line in (OUT / "native" / "native_epyc7742.log").read_text().splitlines():
        kv = dict(x.split("=", 1) for x in line.split()[1:]) if line.startswith("RESULT") else None
        if not kv:
            continue
        if kv["test"] == "rtt":
            nat.setdefault(("rtt", "flush", 0), []).append(float(kv["flush_ns"]))
        elif kv["test"] == "oneway":
            nat.setdefault(("oneway", kv["proto"], int(kv["size"])), []).append(float(kv["lat_ns"]))
        elif kv["test"] == "rpc":
            nat.setdefault(("rpc", kv["proto"], int(kv["service_ns"])), []).append(float(kv["rtt_ns"]))
    g = {}
    for name in ("cal_gem5.csv", "cal_gem5_hw.csv"):
        for r in csv.DictReader(open(OUT / name)):
            g[(r["test"], r["protocol"], int(r["arg"]))] = float(r["ns"])
    rows = []
    for test, args in (("oneway", (64, 256, 1024, 4096)), ("rpc", (0, 2000, 10000))):
        for a in args:
            n_fl, n_li = median(nat[(test, "flush", a)]), median(nat[(test, "lines", a)])
            g_fl, g_li = g[(test, "flush", a)], g[(test, "lines", a)]
            row = dict(test=test, arg=a, native_flush=n_fl, native_lines=n_li, gem5_flush=g_fl, gem5_lines=g_li,
                       gem5_flush_over_native=g_fl / n_fl, gem5_lines_over_native=g_li / n_li)
            for hw in ("uc", "parked", "wbpark"):
                row[f"gem5_{hw}"] = g[(test, hw, a)]
            best_g, best_n = min(g_fl, g_li), min(n_fl, n_li)
            for hw in ("parked", "wbpark"):
                row[f"{hw}_vs_best_gem5"] = best_g / g[(test, hw, a)]
                row[f"{hw}_vs_best_native"] = best_n / g[(test, hw, a)]
            rows.append(row)
    wb = median(float(line.split("wb_ns=")[1]) for line in (OUT / "native" / "native_wb.log").read_text().splitlines()
                if line.startswith("RESULT"))
    for r in rows:
        k = 1 if r["test"] == "oneway" else 2       # write-backs on the critical path: one per direction
        for proto in ("flush", "lines"):
            r[f"native_{proto}_adj"] = r[f"native_{proto}"] + k * wb
            r[f"gem5_{proto}_over_adj"] = r[f"gem5_{proto}"] / r[f"native_{proto}_adj"]
        best_a = min(r["native_flush_adj"], r["native_lines_adj"])
        for hw in ("parked", "wbpark"):
            r[f"{hw}_vs_best_native_adj"] = best_a / r[f"gem5_{hw}"]
    print(f"native write-back, median of 5: {wb:.0f} ns")
    rtt = median(nat[("rtt", "flush", 0)])
    print(f"native flush round trip, median of {len(nat[('rtt', 'flush', 0)])}: {rtt:.0f} ns")
    write("cal_summary.csv", [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
        sys.exit()
    if sys.argv[1] == "rtt":
        with Pool(WORKERS) as p:
            write("cal_rtt.csv", p.map(rtt, ["150ns", "175ns", "200ns", "225ns", "250ns", "300ns"]))
    elif sys.argv[1] == "wb":
        res, _ = gem5(f"cal_wb_{sys.argv[2]}", ["wb - - 2000"], "uc", sys.argv[2])
        write("cal_wb.csv", [dict(link_delay=sys.argv[2], wb_ns=float(res[0]["wb_ns"]))])
    elif sys.argv[1] == "hw":
        # the hardware protocols at the same delay, to compare against the native software baselines
        d = sys.argv[2]
        cases = [(d, "oneway", p, s) for p, s in itertools.product(("uc", "parked", "wbpark"), (64, 256, 1024, 4096))]
        cases += [(d, "rpc", p, s) for p, s in itertools.product(("uc", "parked", "wbpark"), (0, 2000, 10000))]
        with Pool(WORKERS) as p:
            write("cal_gem5_hw.csv", p.map(case, cases))
    else:
        d = sys.argv[2]
        cases = [(d, "oneway", p, s) for p, s in itertools.product(("flush", "lines"), (64, 256, 1024, 4096))]
        cases += [(d, "rpc", p, s) for p, s in itertools.product(("flush", "lines"), (0, 2000, 10000))]
        with Pool(WORKERS) as p:
            write("cal_gem5.csv", p.map(case, cases))
