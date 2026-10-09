"""Set up all cases, run MQSim (2 at a time), compute steady-state WAF by differencing.

usage: driver.py setup | run | report
"""
import csv
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
WORK = os.environ.get("MQSIM_WORK", os.path.join(PROJECT, "data", "mqsim"))   # build and runs, not in git
ROOT = WORK
RUNS = os.environ.get("RUNS", os.path.join(WORK, "runs"))
MQSIM = os.environ.get("MQSIM", os.path.join(WORK, "MQSim", "MQSim"))
STREAMS = os.path.join(PROJECT, "data", "streams")
DELTAS = [0, 9]

idx = {r["name"]: r for r in csv.DictReader(open(os.path.join(STREAMS, "index.csv")))}
cases = []
for name, r in idx.items():
    L, n = int(r["logical_units"]), int(r["writes"])
    for op, key in [(0.07, "waf_op7"), (0.28, "waf_op28")]:
        cases.append(dict(name=name, src=os.path.join(STREAMS, name + ".npy"), L=L, op=op, N=n, ref=float(r[key])))
for op in (0.07, 0.28):
    L = 50000
    P = int(np.ceil(L * (1 + op) / 64)) * 64
    cases.append(dict(name="uniform50k", src=f"uniform:{L}:{16 * P}:1", L=L, op=op, N=16 * P, ref=None))


def n_eb(c):
    return max(int(np.ceil(c["L"] * (1 + c["op"]) / 64)), c["L"] // 64 + 4)


def dirs(c):
    W = 2 * n_eb(c) * 64
    for d in DELTAS:
        for tag, n in (("warm", W), ("full", c["N"])):
            yield d, tag, n, os.path.join(RUNS, f"{c['name']}_op{int(round(c['op'] * 100))}_d{d}_{tag}")


def parse(case_dir):
    x = open(os.path.join(case_dir, "workload_scenario_1.xml")).read()
    f = lambda k: float(re.search(rf"<{k}>([0-9.]+)</{k}>", x).group(1)) if f"<{k}>" in x else float(re.search(rf'{k}="([-0-9.na]+)"', x).group(1))
    return dict(host=f("Write_Request_Count"), prog=f("Issued_Flash_Program_CMD"), mprog=f("Issued_Flash_Multiplane_Program_CMD"),
                gc=f("Total_GC_Executions"), mov=f("Average_Page_Movement_For_GC"), erase=f("Issued_Flash_Erase_CMD"),
                mapprog=f("Issued_Flash_Program_CMD_For_Mapping"))


cmd = sys.argv[1]
if cmd == "setup":
    for c in cases:
        for d, tag, n, cd in dirs(c):
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "mk.py"), cd, c["src"], str(c["L"]), str(c["op"]), str(d), str(n)], check=True)
elif cmd == "run":
    todo = [cd for c in cases for _, _, _, cd in dirs(c) if not os.path.exists(os.path.join(cd, "workload_scenario_1.xml"))]
    open(os.path.join(RUNS, "todo.txt"), "w").write("\n".join(todo) + "\n")
    subprocess.run(f"xargs -P2 -I{{}} sh -c 'cd {{}} && /usr/bin/time -v {MQSIM} -i ssdconfig.xml -w workload.xml > out.log 2> time.log' < {RUNS}/todo.txt", shell=True, check=True)
elif cmd == "report":
    rows = []
    for c in cases:
        res = {}
        for d, tag, n, cd in dirs(c):
            p = parse(cd)
            t = open(os.path.join(cd, "time.log")).read()
            p["wall"] = re.search(r"Elapsed \(wall clock\) time .*: (\S+)", t).group(1)
            p["rss_mb"] = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", t).group(1)) // 1024
            res[(d, tag)] = p
        row = dict(case=c["name"], op=c["op"], L=c["L"], N=c["N"], ref=c["ref"])
        for d in DELTAS:
            w, f = res[(d, "warm")], res[(d, "full")]
            assert w["mprog"] == 0 and f["mprog"] == 0 and f["mapprog"] == 0
            row[f"waf_d{d}"] = (f["prog"] - w["prog"]) / (f["host"] - w["host"])
            row[f"waf_gc_d{d}"] = 1 + (f["gc"] * f["mov"] - w["gc"] * w["mov"]) / (f["host"] - w["host"])
            row[f"overall_d{d}"] = f["prog"] / f["host"]
            row[f"wall_d{d}"] = f["wall"]
            row[f"rss_d{d}"] = f["rss_mb"]
            row[f"raw_d{d}"] = dict(warm=w, full=f)
        rows.append(row)
    json.dump(rows, open(os.path.join(ROOT, "results.json"), "w"), indent=1)
    for r in rows:
        ref = f"{r['ref']:.3f}" if r["ref"] is not None else "-"
        print(f"{r['case']:16s} op={r['op']:.2f} N={r['N']:8d} ref={ref:>6s} "
              + " ".join(f"d{d}: waf={r[f'waf_d{d}']:.3f} (gc-based {r[f'waf_gc_d{d}']:.3f}, whole-run {r[f'overall_d{d}']:.3f}, {r[f'wall_d{d}']}, {r[f'rss_d{d}']}MB)" for d in DELTAS))
