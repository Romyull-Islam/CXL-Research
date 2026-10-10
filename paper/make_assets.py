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

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.5,
                     "legend.fontsize": 7, "legend.frameon": False, "pdf.fonttype": 42})
TRACES = {"bailian_B_api": "Bailian API", "bailian_thinking": "Bailian thinking",
          "mooncake_toolagent": "Mooncake tool-agent", "mooncake_conversation": "Mooncake conversation",
          "bailian_coder": "Bailian coder", "bailian_A_chat": "Bailian chat"}
MODELS = {"DeepSeek-V3": "DeepSeek-V3 (MLA)", "Llama-3.1-70B": "Llama-3.1-70B", "Qwen2.5-7B": "Qwen2.5-7B"}
ADMIT_C = {"all": "#999999", "2nd hit": "#4C72B0", "oracle": "#C44E52"}
BUDGETS = [("TLC 1 DWPD", 2.0), ("TLC 3 DWPD", 6.0), ("pSLC", 23.3), ("SLC", 100.0)]


def tex(df, path, fmt=None):
    path.write_text(df.to_latex(index=False, escape=True, float_format=fmt or (lambda x: f"{x:.2f}")))


def fig_wear():
    d = pd.read_csv(R / "endurance_sweep_2000GB.csv")
    d = d[(d.precision == "BF16") & (d.duty == 1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    x = np.arange(len(TRACES))
    for ax, (m, mlab) in zip(axes, MODELS.items()):
        for j, (a, c) in enumerate(ADMIT_C.items()):
            y = [d[(d.model == m) & (d.trace == t) & (d.admit == a)].TBday_op28.iloc[0] for t in TRACES]
            ax.bar(x + (j - 1) * 0.27, y, 0.27, color=c, label=a)
        for name, b in BUDGETS:
            ax.axhline(b, color="k", lw=0.5, ls=":")
            if m == "Qwen2.5-7B":
                ax.text(len(TRACES) - 0.45, b, " " + name, fontsize=6, va="center", ha="left")
        ax.set_yscale("log")
        ax.set_xticks(x, list(TRACES.values()), rotation=40, ha="right", fontsize=6.5)
        ax.set_title(mlab, fontsize=8)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
    axes[0].set_ylabel("NAND writes per node (TB/day)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, [f"admit {x}" for x in lab], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, rect=(0, 0, 0.93, 0.92))
    fig.savefig(HERE / "fig_wear.pdf")


def fig_waf():
    d = pd.concat([pd.read_csv(R / f"endurance_sweep_{c}GB.csv") for c in (2000, 667)])
    lt = pd.read_csv(R / "lifetime_waf.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw={"width_ratios": [1.2, 1]})
    for a, c in ADMIT_C.items():
        for op, ls in ((7, "-"), (28, "--")):
            v = np.sort(d[(d.admit == a)][f"waf_op{op}"].dropna().values)
            axes[0].plot(v, np.arange(1, len(v) + 1) / len(v), color=c, ls=ls, label=f"{a}, {op}% OP")
    axes[0].set_xlabel("write amplification (prefix-cache sweep)")
    axes[0].set_ylabel("share of configurations")
    axes[0].set_xlim(0.98, 2.0)
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].grid(alpha=0.3, lw=0.4)
    labels = {"prefix LRU (all)": "LRU cache,\nadmit all", "prefix LRU (oracle)": "LRU cache,\noracle"}
    rows = []
    for _, r in lt.iterrows():
        name = labels.get(r.policy, "per request,\n" + ("TRIM" if r.trim else "no TRIM (CXL)"))
        rows.append((name, r.op, r.waf))
    names = list(dict.fromkeys(n for n, _, _ in rows))
    xx = np.arange(len(names))
    for j, (op, c) in enumerate(((0.07, "#4C72B0"), (0.28, "#DD8452"))):
        y = [next(w for n, o, w in rows if n == nm and abs(o - op) < 1e-9) for nm in names]
        axes[1].bar(xx + (j - 0.5) * 0.38, y, 0.38, color=c, label=f"{int(op * 100)}% OP")
    axes[1].set_xticks(xx, names, fontsize=6.5)
    axes[1].set_ylabel("write amplification")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.4)
    fig.savefig(HERE / "fig_waf.pdf")


def fig_pool():
    d = pd.read_csv(R / "pool_sharing.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
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
    axes[0].legend()
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
    for t, c in (("conversation", "#C44E52"), ("toolagent", "#4C72B0")):
        y = rp[(rp.trace == t) & (rp.slack < 1e6)]
        axes[2].plot(y.imbalance, y.advantage, color=c, marker="o", label=f"Mooncake {t}")
    axes[2].axhline(136, color="k", lw=0.6, ls=":")
    axes[2].text(1.05, 120, "Mooncake FAST'25: up to +136%", fontsize=6)
    axes[2].set_ylim(0, 150)
    axes[2].set_xlabel("allowed imbalance (max/mean)")
    axes[2].set_ylabel("pool advantage in hits (%)")
    axes[2].set_title("10 nodes x 3M tokens", fontsize=8)
    axes[2].legend(fontsize=6)
    axes[2].grid(alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.4)
    fig.savefig(HERE / "fig_pool.pdf")


def fig_load():
    d = pd.read_csv(R / "load_vs_recompute.csv")
    d = d[d.gpu == "8x H100"]
    fig, ax = plt.subplots(figsize=(4.8, 2.3))
    cols = {"DeepSeek-V3": "#C44E52", "Llama-3.1-70B": "#4C72B0", "Qwen2.5-7B": "#55A868"}
    ls = {"BF16": "-", "FP8": "--", "INT4": ":"}
    for (m, p), g in d.groupby(["model", "precision"]):
        ax.plot(g.context, g.breakeven_GBs, color=cols[m], ls=ls[p], marker="o",
                label=f"{m.split('-')[0]} {p}" if p == "BF16" else None)
    for name, b in (("CMM-H NAND 5", 5.0), ("CMM-H DRAM 27", 27.0)):
        ax.axhline(b, color="k", lw=0.5, ls=":")
        ax.text(2100, b * 1.07, name + " GB/s", fontsize=6)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("context tokens")
    ax.set_ylabel("break-even bandwidth (GB/s)")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              title="BF16 solid\nFP8 dashed\nINT4 dotted", title_fontsize=6)
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


if __name__ == "__main__":
    table_mqsim()
    fig_wear()
    fig_waf()
    fig_pool()
    fig_load()
    table_provisioning()
    table_ftl()
    table_traces()
    print("ok")
