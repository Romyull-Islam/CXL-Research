"""Figures and LaTeX tables for the paper, generated from results/ so no number is copied by hand.
Run: python paper/make_assets.py  (needs data/ for the trace table)"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R = ROOT / "results"
sys.path.insert(0, str(ROOT / "analysis"))
import figfont  # noqa: E402

figfont.use(figfont.LIBERTINE, "Linux Libertine TT")

plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.5,
                     "legend.fontsize": 7, "legend.frameon": False})
TRACES = {"bailian_B_api": "Bailian API", "bailian_thinking": "Bailian thinking",
          "mooncake_toolagent": "Mooncake tool-agent", "mooncake_conversation": "Mooncake conversation",
          "bailian_coder": "Bailian coder", "bailian_A_chat": "Bailian chat"}
MODELS = {"DeepSeek-V3": "DeepSeek-V3 (MLA)", "Llama-3.1-70B": "Llama-3.1-70B", "Qwen2.5-7B": "Qwen2.5-7B"}
ADMIT_C = {"all": "#999999", "2nd hit": "#4C72B0", "oracle": "#C44E52"}
BUDGETS = [("TLC-1", 2.0), ("TLC-3", 6.0), ("pSLC", 23.3), ("SLC", 100.0)]


def tex(df, path, fmt=None):
    path.write_text(df.to_latex(index=False, escape=True, float_format=fmt or (lambda x: f"{x:.2f}")))


def fig_wear():
    """Dot plot: NAND TB/day per node by trace and admission policy, one panel per model, with the
    five-year budgets of a 2 TB device as vertical lines."""
    d = pd.read_csv(R / "endurance_sweep_2000GB.csv")
    d = d[(d.precision == "BF16") & (d.duty == 1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.3), sharey=True)
    y = np.arange(len(TRACES))
    for ax, (m, mlab) in zip(axes, MODELS.items()):
        for name, b in BUDGETS:
            ax.axvline(b, color="0.55", lw=0.6, ls=":")
            ax.text(b, len(TRACES) + 0.75, name.replace(" DWPD", ""), rotation=90, fontsize=6,
                    ha="right", va="top", color="0.35")
        for i, t in enumerate(TRACES):
            v = [d[(d.model == m) & (d.trace == t) & (d.admit == a)].TBday_op28.iloc[0] for a in ADMIT_C]
            ax.plot([min(v), max(v)], [i, i], color="0.75", lw=1.0, zorder=1)
            for (a, c), val in zip(ADMIT_C.items(), v):
                ax.scatter(val, i, s=14, color=c, zorder=3, label=f"admit {a}" if i == 0 else None)
        ax.set_xscale("log")
        ax.set_xlim(1, 2000)
        ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
        ax.set_title(mlab, fontsize=8)
        ax.grid(axis="x", alpha=0.3, lw=0.4)
        ax.set_xlabel("NAND writes (TB/day)")
    axes[0].set_yticks(y, list(TRACES.values()))
    axes[0].set_ylim(-0.6, len(TRACES) + 0.8)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.02))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.92))
    fig.subplots_adjust(left=0.26, wspace=0.12)
    fig.savefig(HERE / "fig_wear.pdf")


def fig_concept():
    """How KV dies: erase blocks at garbage-collection time under three regimes."""
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.75))
    cols = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}
    ttl = ["(a) LRU prefix cache", "(b) per request, TRIM", "(c) per request, no TRIM"]
    waf = ["measured WAF 1.00 to 1.02", "measured WAF 2.1", "measured WAF 5.2"]
    for k, ax in enumerate(axes):
        for e in range(3):
            for j in range(8):
                x, yy = j, 2 - e
                if k == 0:
                    owner = "A"
                    dead = e == 0
                    shade = 0.35 + 0.08 * (e * 8 + j) / 3
                    face = plt.cm.Blues(min(0.3 + 0.03 * (e * 8 + j), 0.9))
                else:
                    owner = "ABC"[(j + e) % 3]
                    dead = owner == "B"
                    face = cols[owner]
                trimmed = dead and k < 2
                ax.add_patch(Rectangle((x, yy), 0.9, 0.85, facecolor="white" if trimmed else face,
                                       edgecolor="0.4", lw=0.4, hatch="////" if trimmed else None))
                if dead and k == 2:
                    ax.text(x + 0.45, yy + 0.42, "x", ha="center", va="center", fontsize=7, color="white")
            ax.text(-0.25, 2 - e + 0.42, f"EB{e + 1}", ha="right", va="center", fontsize=6.5)
        ax.set_xlim(-1.3, 8.1)
        ax.set_ylim(-0.75, 3.35)
        ax.axis("off")
        ax.set_title(ttl[k], fontsize=7.5)
        ax.text(3.9, -0.45, waf[k], ha="center", fontsize=7)
    axes[0].text(3.9, 3.05, "evicted in write order: EB1 empties", ha="center", fontsize=6)
    axes[1].text(3.9, 3.05, "request B ends: holes in every EB", ha="center", fontsize=6)
    axes[2].text(3.9, 3.05, "B's pages dead but still valid", ha="center", fontsize=6)
    fig.tight_layout(pad=0.2)
    fig.savefig(HERE / "fig_concept.pdf")


def fig_prov():
    """Cheapest flash class that lasts five years with second-hit admission."""
    from matplotlib.colors import ListedColormap
    t = pd.read_csv(R / "provisioning.csv")
    t = t[t.admit == "2nd hit"]
    classes = ["TLC-1", "TLC-3", "pSLC", "SLC", "none"]
    cmap = ListedColormap(["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"])
    cols = [(1.0, "BF16"), (1.0, "FP8"), (1.0, "INT4"), (0.3, "BF16"), (0.3, "FP8"), (0.3, "INT4")]
    rows, labels = [], []
    for tr, lab in TRACES.items():
        for m in MODELS:
            rows.append([classes.index(t[(t.trace == tr) & (t.model == m) & (t.duty == du) &
                                          (t.precision == pr)].cheapest.iloc[0]) for du, pr in cols])
            labels.append(f"{lab}, {m}")
    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=4.5, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, classes[M[i, j]], ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] in (0, 4) else "black")
    ax.set_xticks(range(6), [f"{int(du * 100)}% duty\n{pr}" for du, pr in cols], fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
    for i in range(3, len(labels), 3):
        ax.axhline(i - 0.5, color="white", lw=2)
    ax.axvline(2.5, color="white", lw=2)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "fig_prov.pdf")


def fig_waf():
    d = pd.concat([pd.read_csv(R / f"endurance_sweep_{c}GB.csv") for c in (2000, 667)])
    lt = pd.read_csv(R / "lifetime_waf.csv")
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2), gridspec_kw={"width_ratios": [1.1, 1]})
    for a, c in ADMIT_C.items():
        for op, ls in ((7, "-"), (28, "--")):
            v = np.sort(d[(d.admit == a)][f"waf_op{op}"].dropna().values)
            axes[0].plot(v, np.arange(1, len(v) + 1) / len(v), color=c, ls=ls, label=f"{a}, {op}% OP")
    axes[0].set_xlabel("write amplification (prefix-cache sweep)")
    axes[0].set_ylabel("share of configurations")
    axes[0].set_xlim(0.98, 2.0)
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].grid(alpha=0.3, lw=0.4)
    labels = {"prefix LRU (all)": "LRU,\nadmit all", "prefix LRU (oracle)": "LRU,\noracle"}
    rows = []
    for _, r in lt.iterrows():
        name = labels.get(r.policy, "request,\n" + ("TRIM" if r.trim else "no TRIM"))
        rows.append((name, r.op, r.waf))
    names = list(dict.fromkeys(n for n, _, _ in rows))
    xx = np.arange(len(names))
    for j, (op, c) in enumerate(((0.07, "0.3"), (0.28, "0.7"))):
        y = [next(w for n, o, w in rows if n == nm and abs(o - op) < 1e-9) for nm in names]
        axes[1].bar(xx + (j - 0.5) * 0.38, y, 0.38, color=c, label=f"{int(op * 100)}% OP")
        for xv, yv in zip(xx + (j - 0.5) * 0.38, y):
            axes[1].text(xv, yv + 0.06, f"{yv:.2f}", ha="center", va="bottom", fontsize=5.5)
    axes[1].set_xticks(xx, names, fontsize=6.5)
    axes[1].set_ylabel("write amplification")
    axes[1].set_ylim(0, 5.8)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.4)
    fig.savefig(HERE / "fig_waf.pdf")


def fig_pool():
    d = pd.read_csv(R / "pool_sharing.csv")
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.0))
    sty = {"pooled": ("#C44E52", "o"), "prefix-aware": ("#4C72B0", "^"), "random": ("#999999", "s")}
    for ax, t in zip(axes[:2], ("mooncake_conversation", "bailian_A_chat")):
        x = d[(d.trace == t) & (d.hosts == 8)]
        for s, (c, mk) in sty.items():
            y = x[x.scheme == s].sort_values("cap_frac")
            ax.plot(y.cap_frac * 100, y.hit_share, color=c, marker=mk, label=s)
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
        ax.minorticks_off()
        ax.set_xlabel("total cache, % of unique blocks")
        ax.set_title(TRACES[t] + ", 8 hosts", fontsize=8)
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel("prefix hit share")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, ["shared pool", "per-host, cache-aware routing", "per-host, random routing"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
    lines = (R / "mooncake_repro.txt").read_text().splitlines()
    rows = []
    trace = None
    for ln in lines:
        if not ln.startswith(" "):
            trace = ln.split(":")[0]
            continue
        parts = ln.split()
        if parts[0] == "prefix-aware":
            rows.append((trace, float(parts[2].rstrip(":")), float(parts[7].rstrip("%")), float(parts[9])))
    rp = pd.DataFrame(rows, columns=["trace", "slack", "advantage", "imbalance"])
    for t, c, mk in (("conversation", "0.15", "o"), ("toolagent", "0.6", "s")):
        y = rp[(rp.trace == t) & (rp.slack < 1e6)]
        axes[2].plot(y.imbalance, y.advantage, color=c, marker=mk, label={"conversation": "Mooncake conversation", "toolagent": "Mooncake tool-agent"}[t])
    axes[2].axhline(136, color="k", lw=0.6, ls=":")
    axes[2].text(1.05, 122, "Mooncake paper: +136%", fontsize=6)
    axes[2].set_ylim(0, 150)
    axes[2].set_xlabel("allowed imbalance (max/mean)", fontsize=7)
    axes[2].set_ylabel("pool advantage in hits (%)")
    axes[2].set_title("Mooncake setup", fontsize=8)
    axes[2].legend(fontsize=5.5)
    axes[2].grid(alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.4, w_pad=1.2, rect=(0, 0, 1, 0.9))
    fig.savefig(HERE / "fig_pool.pdf", bbox_inches="tight", pad_inches=0.02)


def fig_load():
    from matplotlib.lines import Line2D
    d = pd.read_csv(R / "load_vs_recompute.csv")
    d = d[d.gpu == "8x H100"]
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    cols = {"DeepSeek-V3": "#C44E52", "Llama-3.1-70B": "#4C72B0", "Qwen2.5-7B": "#55A868"}
    ls = {"BF16": "-", "FP8": "--", "INT4": ":"}
    for (m, p), g in d.groupby(["model", "precision"]):
        ax.plot(g.context, g.breakeven_GBs, color=cols[m], ls=ls[p], marker="o")
    for name, b in (("CMM-H NAND 5", 5.0), ("CMM-H DRAM 27", 27.0)):
        ax.axhline(b, color="k", lw=0.5, ls=":")
        above = b < 10
        ax.text(2 ** 10.85, b * 1.08 if above else b / 1.08, name + " GB/s", fontsize=6,
                va="bottom" if above else "top")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(2 ** 10.7, 2 ** 15.3)
    ax.set_xticks([2 ** k for k in range(11, 16)], ["2K", "4K", "8K", "16K", "32K"])
    ax.set_yticks([0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlabel("context tokens")
    ax.set_ylabel("break-even bandwidth (GB/s)")
    h = [Line2D([], [], color=c, marker="o", label=MODELS[m]) for m, c in cols.items()]
    h += [Line2D([], [], color="0.3", ls=v, label=f"{k} KV") for k, v in ls.items()]
    ax.legend(handles=h, fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.4)
    fig.savefig(HERE / "fig_load.pdf")


def table_traces():
    from nand_stream import load, oracle_flags
    rows = []
    for t, lab in TRACES.items():
        path = {"mooncake_conversation": "mooncake/conversation_trace.jsonl",
                "mooncake_toolagent": "mooncake/toolagent_trace.jsonl",
                "bailian_A_chat": "bailian/qwen_traceA_blksz_16.jsonl", "bailian_B_api": "bailian/qwen_traceB_blksz_16.jsonl",
                "bailian_thinking": "bailian/qwen_thinking_blksz_16.jsonl",
                "bailian_coder": "bailian/qwen_coder_blksz_16.jsonl"}[t]
        reqs = [json.loads(line) for line in open(ROOT / "data" / path)]
        tr = load(ROOT / "data" / path)
        orc = oracle_flags(tr["seqs"])
        flat = np.concatenate(tr["seqs"])
        first = np.zeros(len(flat), dtype=bool)
        seen = set()
        for i, b in enumerate(flat.tolist()):
            first[i] = b not in seen
            seen.add(b)
        rows.append({"trace": lab, "requests": len(reqs), "hours": round(tr["duration"] / 3600, 2),
                     "block tokens": 512 if "mooncake" in t else 16,
                     "mean input": round(np.mean([r["input_length"] for r in reqs])),
                     "mean output": round(np.mean([r["output_length"] for r in reqs])),
                     "unique blocks": tr["n_blocks"],
                     "blocks reused": f"{orc[first].mean() * 100:.0f}%"})
    tex(pd.DataFrame(rows), HERE / "tab_traces.tex")


def table_provisioning():
    t = pd.read_csv(R / "provisioning.csv")
    t = t[t.admit == "2nd hit"]
    rows = []
    for tr, lab in TRACES.items():
        for m in MODELS:
            row = {"trace": lab, "model": m}
            for duty in (1.0, 0.3):
                for p in ("BF16", "FP8", "INT4"):
                    row[f"{int(duty * 100)}% {p}"] = t[(t.trace == tr) & (t.model == m) & (t.duty == duty) &
                                                      (t.precision == p)].cheapest.iloc[0]
            rows.append(row)
    tex(pd.DataFrame(rows), HERE / "tab_provisioning.tex")


def table_ftl():
    from ftl import FTL, fifo_waf, greedy_waf
    rows = []
    for op in (0.07, 0.28):
        for pol in ("fifo", "greedy"):
            L = 40000
            f = FTL(L, op, 64, policy=pol)
            r = f.run(np.random.default_rng(3).integers(0, L, 15 * L), warmup=5 * L)
            ref = fifo_waf(f.op_effective) if pol == "fifo" else greedy_waf(f.op_effective, 64)
            rows.append({"over-provisioning": f"{int(op * 100)}%", "victim": pol,
                         "effective OP": round(f.op_effective, 4), "simulated WAF": round(r["waf"], 3),
                         "closed form": round(ref, 3)})
    tex(pd.DataFrame(rows), HERE / "tab_ftl.tex", fmt=lambda x: f"{x:.3f}")


def table_mqsim():
    from ftl import FTL
    m = pd.read_csv(R / "mqsim_crosscheck.csv")
    idx = pd.read_csv(ROOT / "data" / "streams" / "index.csv").set_index("name")
    names = {"lru_chat_oracle": "LRU cache, oracle admission", "lru_chat_all": "LRU cache, admit all",
             "scoped_notrim": "request-scoped, no TRIM", "uniform50k": "uniform random (reference)"}
    rows = []
    for _, r in m.iterrows():
        L = int(r.logical_units)
        if r.case == "uniform50k":
            n_eb = int(np.ceil(L * (1 + r.op) / 64))
            w = np.random.default_rng(1).integers(0, L, int(r.host_writes))
            warm = 2 * n_eb * 64
        else:
            w = np.load(ROOT / "data" / "streams" / f"{r.case}.npy")
            warm = 2 * L
        ours = FTL(L, r.op, 64, reserve=10).run(w, warmup=warm)["waf"]
        rows.append({"KV write stream": names[r.case], "OP": f"{int(round(r.op * 100))}%",
                     "MQSim": round(r.mqsim_waf, 3), "this FTL": round(ours, 3),
                     "difference": f"{(ours / r.mqsim_waf - 1) * 100:+.1f}%"})
    tex(pd.DataFrame(rows), HERE / "tab_mqsim.tex", fmt=lambda x: f"{x:.3f}")


def table_ssd():
    """Real drive (SanDisk Extreme, WD SN580E inside, over USB): steady-state host write bandwidth
    (median of each phase's last third) and its slowdown against sequential overwrites, next to
    the write amplification this FTL measures for the same stream at 7% over-provisioning."""
    import csv
    import statistics as st
    d = R / "ssd"

    def steady(name):
        rows = list(csv.DictReader(open(d / f"{name}.csv")))
        return st.median(float(r["interval_MBps"]) for r in rows[len(rows) * 2 // 3:]), \
            float(rows[-1]["host_bytes"]) / 1e12
    lt = pd.read_csv(R / "lifetime_waf.csv")
    lt7 = lt[lt.op == 0.07]
    sim = {"lru_chat": float(lt7[lt7.policy == "prefix LRU (all)"].waf.iloc[0]),
           "scoped_trim": float(lt7[(lt7.policy == "request-scoped") & lt7.trim].waf.iloc[0]),
           "scoped_notrim": float(lt7[(lt7.policy == "request-scoped") & ~lt7.trim].waf.iloc[0])}
    from ftl import greedy_waf, FTL
    sim["rand"] = greedy_waf(FTL(40000, 0.07, 64).op_effective, 64)
    sim["seq"] = 1.0
    names = {"seq": "sequential overwrite", "lru_chat": "LRU prefix cache, admit all",
             "scoped_trim": "request-scoped, TRIM", "scoped_notrim": "request-scoped, no TRIM",
             "rand": "uniform random"}
    base = steady("seq")[0]
    rows = []
    for k, lab in names.items():
        bw, tb = steady(k)
        rows.append({"write stream": lab, "written (TB)": f"{tb:.2f}", "steady MB/s": f"{bw:.0f}",
                     "slowdown": f"{base / bw:.2f}", "simulated WAF": f"{sim[k]:.2f}"})
    tex(pd.DataFrame(rows), HERE / "tab_ssd.tex")
    t = (HERE / "tab_ssd.tex").read_text().replace("{lllll}", "{lrrrr}", 1)
    (HERE / "tab_ssd.tex").write_text(t)
    disc = list(csv.DictReader(open(d / "discard.csv")))
    idle = {int(r["bytes"]): [] for r in disc}
    for r in disc:
        if r["load"] == "idle":
            idle[int(r["bytes"])].append(float(r["seconds"]))
    load = {}
    for r in csv.DictReader(open(d / "discard_load.csv")):
        load.setdefault(int(r["bytes"]), []).append(float(r["seconds"]))
    print("discard idle ms:", {b >> 20: round(st.median(v) * 1e3, 1) for b, v in idle.items()})
    print("discard writing ms:", {b >> 20: round(st.median(v) * 1e3, 1) for b, v in load.items()})


if __name__ == "__main__":
    table_mqsim()
    fig_wear()
    # fig_concept.pdf is drawn in PowerPoint: paper/fig_concept.js, paper/diagrams.sh
    fig_prov()
    fig_waf()
    fig_pool()
    fig_load()
    table_provisioning()
    table_ftl()
    table_traces()
    table_ssd()
    print("ok")
