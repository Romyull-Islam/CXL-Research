"""Peak bandwidth and unloaded latency vs read/write mix, CXL ASIC vs DDR5 (ByteDance EuroSys'24 MLC logs)."""
import glob
import re
from collections import defaultdict

MIX = {0: "read only", 2: "2R:1W", 3: "3R:1W", 4: "3R:2W", 5: "1R:1W", 6: "NT write only",
       7: "2R:1NTW", 8: "1R:1NTW", 9: "3R:1NTW", 10: "W10 (unlabeled)", 11: "W11 (unlabeled)",
       12: "W12 (unlabeled)"}
NAME = re.compile(r"log_d(\d+)_c(\d+)_w(\d+)_r(\d)$")


def parse(path):
    rows = []
    for line in open(path, errors="ignore"):
        m = re.match(r"\s*(\d{5})\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return rows


runs = defaultdict(dict)
for dev in ("cxl", "ddr5"):
    for path in glob.glob(f"data/bytedance/repo/data/{dev}/log_d*"):
        m = NAME.search(path)
        if not m:
            continue
        d, c, w, r = map(int, m.groups())
        rows = parse(path)
        if rows:
            runs[dev][(d, c, w, r)] = (max(b for _, _, b in rows) / 1000, rows[-1][1])

for dev in runs:
    nodes = sorted({k[0] for k in runs[dev]})
    conc = sorted({k[1] for k in runs[dev]})
    print(f"{dev}: nodes {nodes}, concurrencies {conc}, runs {len(runs[dev])}")

print("\nPeak bandwidth GB/s (best node and thread count per mix), sequential access")
print(f"{'mix':18} {'DDR5':>8} {'CXL':>8} {'CXL/DDR5':>9}   {'CXL vs its read-only':>20}")
best = {dev: {} for dev in runs}
for dev in runs:
    for (d, c, w, r), (bw, lat) in runs[dev].items():
        if r == 0 and bw > best[dev].get(w, (0,))[0]:
            best[dev][w] = (bw, lat, d, c)
for w in sorted(MIX):
    if w in best["ddr5"] and w in best["cxl"]:
        dd, cx = best["ddr5"][w][0], best["cxl"][w][0]
        print(f"{MIX[w]:18} {dd:8.1f} {cx:8.1f} {cx / dd:9.2f}   {cx / best['cxl'][0][0]:20.2f}")
