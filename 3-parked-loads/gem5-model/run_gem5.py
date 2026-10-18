"""gem5 experiments for parked loads. Each experiment writes results/<name>.csv.

Devices: idle uncached round trip 250 ns (expander), 280 ns (multi-headed device) and 600 ns
(switch), set by the one-way link delay (round trip = 124.5 ns + 2 x delay in this config).
Protocols: flush (clflushopt + mfence + load of a write-back flag), uc (uncached load, parking
off), parked (uncached flag load parked), wbpark (cacheable line fill parked, the line carries the
message), and for oneway/rpc lines (poll the payload lines themselves, no flag); bN is uncached
polling with exponential backoff capped at N ns (experiments bo_*, written to results/bo_*.csv). Run: python run_gem5.py [experiment ...]   (default: all)
"""
import csv
import itertools
import os
import re
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEM5 = HERE.parent / "gem5" / "build" / "X86" / "gem5.opt"
OUT = HERE / "results"
DEVICES = {"expander": "63ns", "mhd": "78ns", "switch": "238ns"}
PROTOS = ["flush", "uc", "parked", "wbpark"]
WORKERS = int(os.environ.get("WORKERS", 5))


def gem5(tag, hosts, proto, delay, extra=()):
    rundir = HERE / "runs" / tag
    cmd = [str(GEM5), "-d", str(rundir), str(HERE / "configs" / "multihost.py"),
           "--park", "1" if "park" in proto else "0", "--link-delay", delay, *extra,
           "--hosts", *hosts]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    res = [dict(kv.split("=", 1) for kv in line.split()[1:])
           for line in p.stdout.splitlines() if line.startswith("RESULT")]
    stats = {}
    st = rundir / "stats.txt"
    if st.exists():
        for line in st.read_text().splitlines():
            m = re.match(r"dev\.parked\.(\w+)\s+([\d.]+)", line)
            if m:
                stats[m.group(1)] = float(m.group(2))
    if p.returncode or not res:
        sys.stderr.write(f"FAILED {tag}\n{p.stdout[-800:]}\n{p.stderr[-1500:]}\n")
    return res, stats


def oneway(a):
    dev, proto, size = a
    res, st = gem5(f"ow_{dev}_{proto}_{size}",
                   [f"oneway consumer {proto} 30 {size}", f"oneway producer {proto} 30 {size}"],
                   proto, DEVICES[dev])
    r = next(x for x in res if x["test"] == "oneway")
    return dict(device=dev, protocol=proto, size=size, latency_ns=float(r["lat_ns"]),
                polls_per_msg=float(r["polls"]))


def rpc(a):
    dev, proto, service = a
    res, _ = gem5(f"rpc_{dev}_{proto}_{service}",
                  [f"rpc client {proto} 30 {service}", f"rpc server {proto} 30 {service}"],
                  proto, DEVICES[dev])
    r = next(x for x in res if x["test"] == "rpc")
    return dict(device=dev, protocol=proto, service_ns=service, rtt_ns=float(r["rtt_ns"]))


def barrier(a):
    dev, proto, n = a
    hosts = [f"barrier {i} {n} {proto} 20" for i in range(n)]
    res, _ = gem5(f"bar_{dev}_{proto}_{n}", hosts, proto, DEVICES[dev], ("--cpu", "timing"))
    r = next(x for x in res if x["test"] == "barrier")
    return dict(device=dev, protocol=proto, hosts=n, round_ns=float(r["round_ns"]))


def contention(a, mean_wait=20000, table=4096, tag="ct"):
    proto, waiters = a[:2]
    if len(a) > 2:
        mean_wait = a[2]
    if len(a) > 3:
        table = a[3]
    msgs = max(4, int(300_000 / mean_wait))
    t_us = int(msgs * mean_wait / 1000)
    hosts = [f"4*tenant {4 * i} - {t_us} 4" for i in range(4)]
    per = 16
    for h in range(0, waiters, per):
        n = min(per, waiters - h)
        hosts.append(f"T:{n}*waiter {h} {proto} {msgs}")
        hosts.append(f"T:poster {h} {proto} {msgs} {mean_wait} {n}")
    res, st = gem5(f"{tag}_{proto}_{waiters}_{mean_wait}_{table}", hosts, proto,
                   DEVICES["switch"], ("--table", str(table)))
    ten = [float(x["GBs"]) for x in res if x["test"] == "tenant"]
    wai = [x for x in res if x["test"] == "waiter"]
    lat = sum(float(x["lat_ns"]) for x in wai) / len(wai) if wai else float("nan")
    rpm = sum(float(x["reads_per_msg"]) for x in wai) / len(wai) if wai else float("nan")
    return dict(protocol=proto, waiters=waiters, mean_wait_us=mean_wait / 1000, table=table,
                tenant_GBs=sum(ten), handoff_ns=lat, reads_per_msg=rpm,
                parks=st.get("parks", 0), timeouts=st.get("timeouts", 0),
                table_full=st.get("tableFull", 0), max_parked=st.get("maxParked", 0))


def longwait(a):
    return contention(a, tag="lw")


def table(a):
    return contention(a, tag="tb")


BACKOFF = ["b1000", "b4000"]   # uncached polling with exponential backoff, cap 1 and 4 us

EXPERIMENTS = {
    "bo_oneway": (oneway, list(itertools.product(DEVICES, BACKOFF, (64, 256, 1024, 4096)))),
    "bo_rpc": (rpc, list(itertools.product(DEVICES, BACKOFF, (0, 2000, 10000)))),
    "bo_barrier": (barrier, list(itertools.product(("mhd", "switch"), BACKOFF, (4, 8, 16)))),
    "bo_contention": (contention, list(itertools.product(BACKOFF, (16, 64, 128, 256)))),
    "bo_longwait": (longwait, list(itertools.product(BACKOFF, (64,), (20000, 100000, 500000)))),
    "oneway": (oneway, list(itertools.product(DEVICES, PROTOS + ["lines"], (64, 256, 1024, 4096)))),
    "rpc": (rpc, list(itertools.product(DEVICES, PROTOS + ["lines"], (0, 2000, 10000)))),
    "barrier": (barrier, list(itertools.product(("mhd", "switch"), PROTOS, (4, 8, 16)))),
    "contention": (contention, list(itertools.product(PROTOS, (0, 16, 64, 128, 256)))),
    "longwait": (longwait,
                 list(itertools.product(PROTOS, (64,), (20000, 100000, 500000)))),
    "table": (table,
              [(p, 256, 20000, t) for p in ("parked", "wbpark") for t in (4096, 128, 32)]),
}


def main():
    OUT.mkdir(exist_ok=True)
    names = sys.argv[1:] or list(EXPERIMENTS)
    for name in names:
        fn, cases = EXPERIMENTS[name]
        if name == "contention":
            cases = [c for c in cases if c[1] or c[0] == "parked"]
        with Pool(WORKERS) as p:
            rows = p.map(fn, cases)
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"== {name}")
        for r in rows:
            print(r, flush=True)


if __name__ == "__main__":
    main()
