"""Model-check ParkedLoads.tla with all four rules and with each rule removed.

Each (variant, property) pair is one TLC run, so every violated property is reported, not just
the first. Writes tla/results.csv and prints a table. Run: python tla/run_tlc.py
"""
import csv
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
JAR = HERE / "tools" / "tla2tools.jar"

VARIANTS = {
    "all rules": dict(R1=True, R2=True, R3=True, R4=True),
    "all rules, write-back writer": dict(R1=True, R2=True, R3=True, R4=True, WBWrite=True),
    "no R5 (ownership read parked), write-back writer":
        dict(R1=True, R2=True, R3=True, R4=True, R5=False, WBWrite=True),
    "no R1 (parked read keeps its credit)": dict(R1=False, R2=True, R3=True, R4=True),
    "no R2 (write needs a request credit)": dict(R1=True, R2=False, R3=True, R4=True),
    "no R1, no R2": dict(R1=False, R2=False, R3=True, R4=True),
    "no R3 (no device timeout)": dict(R1=True, R2=True, R3=False, R4=True),
    "no R4 (timeout reply looks like data)": dict(R1=True, R2=True, R3=True, R4=False),
}
INVARIANTS = ["TypeOK", "NoRootPortTimeout", "NoStaleRead", "ProducerReadNotBlocked",
              "ProducerWriteNotBlocked"]
SIZES = {
    "2 consumers, 2 credits, 2-entry table": dict(Cons="{c1, c2}", Credits=2, TableSize=2),
    "3 consumers, 2 credits, 2-entry table": dict(Cons="{c1, c2, c3}", Credits=2, TableSize=2),
}
CONST = dict(Timeout=2, RPTimeout=4, ProducerMayFail=True, R5=True, WBWrite=False)


def run(size, variant, prop, temporal=False):
    consts = {**CONST, **SIZES[size], **VARIANTS[variant]}
    cfg = HERE / "states" / "run.cfg"
    cfg.parent.mkdir(exist_ok=True)
    lines = ["SPECIFICATION Spec", "CONSTANTS"]
    lines += [f"  {k} = {str(v).upper() if isinstance(v, bool) else v}" for k, v in consts.items()]
    lines += [("PROPERTY " if temporal else "INVARIANT ") + prop]
    cfg.write_text("\n".join(lines) + "\n")
    out = subprocess.run(
        ["java", "-XX:+UseParallelGC", "-Xmx2g", "-cp", str(JAR), "tlc2.TLC", "-workers", "4",
         "-deadlock", "-metadir", str(HERE / "states"), "-config", str(cfg),
         str(HERE / "ParkedLoads.tla")],
        capture_output=True, text=True, cwd=HERE).stdout
    states = re.findall(r"([\d,]+) distinct states found", out)
    if "Model checking completed. No error has been found" in out:
        verdict = "holds"
    elif "is violated" in out or "Temporal properties were violated" in out:
        depth = len(re.findall(r"^State \d+:", out, re.M))
        verdict = f"violated (trace of {depth} states)"
    else:
        verdict = "ERROR: " + out[-400:]
    return verdict, states[-1] if states else "?"


def main():
    rows = []
    for size in SIZES:
        for v in VARIANTS:
            for p in INVARIANTS:
                verdict, states = run(size, v, p)
                rows.append(dict(size=size, variant=v, property=p, verdict=verdict,
                                 distinct_states=states))
                print(f"{size[:12]} {v:40} {p:25} {verdict} ({states} states)", flush=True)
        for v in ("all rules", "all rules, write-back writer"):
            verdict, states = run(size, v, "EventuallyDone", temporal=True)
            rows.append(dict(size=size, variant=v, property="EventuallyDone (liveness)",
                             verdict=verdict, distinct_states=states))
            print(f"{size[:12]} {v:40} {'EventuallyDone':25} {verdict} ({states} states)")
    with open(HERE / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
