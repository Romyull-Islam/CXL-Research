"""Figures for the letter, generated from results/. Run: python paper/make_assets.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import figfont  # noqa: E402

HERE = Path(__file__).resolve().parent
R = HERE.parent / "results"
figfont.use(figfont.TIMES, "Nimbus Roman TT")
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3,
                     "legend.fontsize": 6, "legend.frameon": False})
DEP = ["70B-8xH100", "70B-8xH200", "70B-8xB200", "QwQ32B-2xH100", "QwQ32B-4xH100", "Qwen3-235B-FP8-8xH100",
       "R1-FP8-8xH200"]
WCOL = {"bailian": "#999999", "bo1": "#0072B2", "bo16": "#E69F00", "bo64": "#D55E00"}   # Okabe-Ito
LAB = ["70B\nH100", "70B\nH200", "70B\nB200", "QwQ\n2×H100", "QwQ\n4×H100", "Q3-235B\nH100", "R1\nH200"]
WL = {"bailian": "Bailian", "bo1": "best-of-1", "bo16": "best-of-16", "bo64": "best-of-64"}
# real-vLLM reuse tests: model, architecture, short name
NAMES = {"DeepSeek-R1-Distill-Qwen-1.5B": ("Qwen2", "Qwen2 1.5B"), "DeepSeek-R1-Distill-Qwen-7B": ("Qwen2", "Qwen2 7B"),
         "Llama-3.2-1B-Instruct": ("Llama", "Llama 1B"), "granite-3.1-1b-a400m-instruct": ("MoE", "Granite MoE 1B"),
         "DeepSeek-R1-Distill-Llama-8B": ("Llama", "Llama 8B"), "DeepSeek-R1-Distill-Qwen-14B": ("Qwen2", "Qwen2 14B"),
         "Qwen3-30B-A3B": ("MoE", "Qwen3 MoE 30B"), "QwQ-32B": ("Qwen2", "QwQ 32B")}
GCOL = {"RTX 4500 Ada": "#CC79A7", "A100": "#009E73"}   # Okabe-Ito


def real():
    """Real-vLLM reuse summaries of both GPUs, one row per model, benchmark and answer length."""
    a = pd.read_csv(R / "reuse" / "summary.csv").assign(gpu="RTX 4500 Ada")
    b = pd.read_csv(R / "reuse_a100" / "summary.csv").assign(gpu="A100")
    return pd.concat([a, b], ignore_index=True)


def gains():
    d = pd.read_csv(R / "gate2_saturation.csv")
    fig, ax = plt.subplots(figsize=(3.5, 1.6))
    x = np.arange(len(DEP))
    cols = [WCOL[w] for w in WL]
    for j, (w, lab) in enumerate(WL.items()):
        y, e = [], []
        for dep in DEP:
            g = {}
            for k in ("fa3", "fa2"):
                r = d[(d.deploy == dep) & (d.wname == w) & (d.kernel == k)].set_index("policy").throughput
                g[k] = (r["swap"] / r["recompute"] - 1) * 100
            y.append(g["fa3"])
            e.append(g["fa2"] - g["fa3"])
        xs = x + (j - 1.5) * 0.2
        ax.bar(xs, y, 0.2, color=cols[j], label=lab)
        top = np.array(y) + np.array(e)
        ax.vlines(xs, y, top, color="k", lw=0.5)                 # FlashAttention-2 calibration
        ax.hlines(top, xs - 0.04, xs + 0.04, color="k", lw=0.5)
    ax.set_xticks(x, LAB, fontsize=6)
    ax.set_ylabel("throughput gain (%)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "fig_gain.pdf", bbox_inches="tight", pad_inches=0.02)


def first_order():
    d = pd.read_csv(R / "gate2_saturation.csv")
    r = d[d.policy == "recompute"].copy()
    r["rho"] = ((r.busy_time - r.recompute_time) / r.out_tokens) / (r.recompute_time / r.recomputed_tokens)
    s = d[d.policy == "swap"].set_index(["deploy", "wname", "kernel"]).throughput
    r["gain"] = (s.loc[list(zip(r.deploy, r.wname, r.kernel))].values / r.throughput.values - 1) * 100
    fig, ax = plt.subplots(figsize=(3.5, 1.7))
    rr = np.linspace(0.3, 1.0, 50)
    for g in (2, 5, 10, 20):
        w = g / (100 + g)
        ax.plot(rr, rr * (1 - w) / w, color="k", lw=0.4, ls=":")
        ax.text(1.0, min(1.0 * (1 - w) / w, 34), f" {g}%", fontsize=5.5, va="center")
    sc = ax.scatter(r.recomputed_per_out, r.rho, c=r.gain, cmap="viridis", s=9, vmin=0, vmax=14)
    cb = fig.colorbar(sc, ax=ax, pad=0.08)
    cb.set_label("simulated gain (%)", fontsize=6)
    ax.set_xlabel("R, recomputed tokens per generated token")
    ax.set_ylabel(r"$\rho$, decode / recompute cost")
    ax.set_ylim(0, 35)
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "fig_model.pdf")


def design():
    """Design space of Eq. 1: gain against R for rho = 5, 30, 300 and two load-to-recompute cost
    ratios, with the simulated preemption runs and the real-vLLM reuse runs at their measured R."""
    d = pd.read_csv(R / "gate2_saturation.csv")
    r = d[d.policy == "recompute"].copy()
    s = d[d.policy == "swap"].set_index(["deploy", "wname", "kernel"]).throughput
    r["gain"] = (s.loc[list(zip(r.deploy, r.wname, r.kernel))].values / r.throughput.values - 1) * 100
    q = real()
    fig, ax = plt.subplots(figsize=(3.5, 1.6))
    rr = np.logspace(-1, 3.5, 200)
    for rho, c in ((5, "0.15"), (30, "0.45"), (300, "0.7")):
        for k, ls in ((1 / 30, "-"), (1 / 5, "--")):
            g = rr * (1 - k) / (rho + rr * k) * 100
            ax.plot(rr, g, color=c, ls=ls, lw=0.9)
        ax.text(2.0, 2.0 * (1 - 1 / 30) / (rho + 2.0 / 30) * 100 * 1.6, rf"$\rho={rho}$", fontsize=6.5,
                color=c, ha="center", va="bottom")
    ax.scatter(r.recomputed_per_out, r.gain, s=8, color="#0072B2", zorder=3, label="preempted, simulated")
    for gpu, mk in (("RTX 4500 Ada", "D"), ("A100", "s")):
        x = q[q.gpu == gpu]
        ax.scatter(x.R, x.gain_measured * 100, s=14, marker=mk, color=GCOL[gpu], zorder=3,
                   label=f"reused, vLLM, {gpu}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 3000)
    ax.set_ylim(0.1, 3000)
    t = [0.1, 1, 10, 100, 1000]
    ax.set_xticks(t, ["0.1", "1", "10", "100", "1000"])
    ax.set_yticks(t, ["0.1", "1", "10", "100", "1000"])
    ax.minorticks_off()
    ax.set_xlabel("$R$, tokens recomputed per generated token")
    ax.set_ylabel("throughput gain (%)")
    ax.legend(loc="upper left", fontsize=6.5, handletextpad=0.3, borderaxespad=0.3)
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "fig_design.pdf", bbox_inches="tight", pad_inches=0.02)


def teaser():
    """Model-predicted against measured gain: simulated preemption runs (gain = R / rho) and real
    vLLM document questions (Eq. 1 with the measured c_r and c_l)."""
    d = pd.read_csv(R / "gate2_saturation.csv")
    r = d[d.policy == "recompute"].copy()
    r["rho"] = ((r.busy_time - r.recompute_time) / r.out_tokens) / (r.recompute_time / r.recomputed_tokens)
    s = d[d.policy == "swap"].set_index(["deploy", "wname", "kernel"]).throughput
    r["gain"] = (s.loc[list(zip(r.deploy, r.wname, r.kernel))].values / r.throughput.values - 1) * 100
    r["model"] = r.recomputed_per_out / r.rho * 100
    q = real()
    q["arch"] = q.model.map(lambda m: NAMES[m][0])
    fig, ax = plt.subplots(figsize=(3.5, 1.6))
    lim = (1, 600)
    ax.plot(lim, lim, color="k", lw=0.5)
    ax.fill_between(lim, [v / 1.2 for v in lim], [v * 1.2 for v in lim], color="0.85", lw=0)
    ax.scatter(r.model, r.gain, s=7, color="0.45", label="preempted KV, simulated (56 runs)", zorder=3)
    for gpu in GCOL:
        x = q[q.gpu == gpu]
        ax.scatter(x.gain_predicted * 100, x.gain_measured * 100, s=14, marker="D" if gpu == "A100" else "o",
                   color=GCOL[gpu], zorder=4, linewidths=0.6,
                   facecolors=[GCOL[gpu] if a != "MoE" else "white" for a in x.arch],
                   label=f"reused KV, vLLM, {gpu} ({x.model.nunique()} models)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ticks = [1, 3, 10, 30, 100, 300]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("predicted gain (%)")
    ax.set_ylabel("observed gain (%)")
    ax.text(1.3, 300, "line: predicted = observed;\nband: within 20%", fontsize=6, va="top")
    ax.legend(loc="lower right", fontsize=6.5, handletextpad=0.2, borderaxespad=0.2)
    ax.text(1.3, 95, "filled: dense\nhollow: mixture of experts", fontsize=6.5, va="top")
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "fig_teaser.pdf", bbox_inches="tight", pad_inches=0.02)


def capacity():
    c = pd.read_csv(R / "gate2_capacity.csv")
    s = pd.read_csv(R / "gate2_saturation.csv")
    s = s[s.kernel == "fa3"]
    rec = s[s.policy == "recompute"].set_index(["deploy", "wname"]).throughput
    full = s[s.policy == "swap"].set_index(["deploy", "wname"]).throughput
    k = list(zip(c.deploy, c.wname))
    c["kept"] = (c.throughput.values / rec.loc[k].values - 1) / (full.loc[k].values / rec.loc[k].values - 1)
    fig, ax = plt.subplots(figsize=(3.5, 1.7))
    for (w, pol), g in c.groupby(["wname", "policy"]):
        m = g.groupby("tier_mult").kept.median()
        ax.plot(m.index, m.values * 100, ls="-" if pol == "swap" else "--",
                color=WCOL[w], marker={"bailian": "o", "bo16": "s", "bo64": "^"}[w],
                label=f"{WL[w]}, {'swap' if pol == 'swap' else 'LRU'}")
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1, 2, 4], ["0.05", "0.1", "0.25", "0.5", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_xlabel("tier capacity / HBM KV capacity")
    ax.set_ylabel("gain kept (%)")
    ax.legend(fontsize=6.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=1.6, handletextpad=0.3)
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "fig_capacity.pdf", bbox_inches="tight", pad_inches=0.02)


def reuse_table():
    """Table II: real vLLM, one row per model over both benchmarks, from the two GPUs' summaries."""
    d = real()

    def pct(x):
        v = int(round(x * 100))
        return f"{v:+d}" if v else "0"

    def rng(v, f=pct, sep=" to "):
        lo, hi = f(v.min()), f(v.max())
        return lo if lo == hi else f"{lo}{sep}{hi}"
    lines = []
    for gpu in GCOL:
        if lines:
            lines.append("\\midrule")
        for m, (_, name) in NAMES.items():
            g = d[(d.gpu == gpu) & (d.model == m)]
            if g.empty:
                continue
            gains = " / ".join(rng(g[g.out == o].gain_measured, lambda v: f"{v * 100:.0f}", "--")
                               for o in sorted(g.out.unique()))
            lines.append(f"{name} & {gains} & {rng(g.error)} & "
                         f"{rng(g[g.out > g.out.min()].error_insitu)} \\\\")
    body = "\n".join(x.replace("--", "\x00").replace("-", "$-$").replace("\x00", "--") for x in lines)
    head = ("\\begin{tabular}{@{}lrrr@{}}\n\\toprule\n & measured gain (\\%) & \\multicolumn{2}{c}{error (\\%), $c_l$ from} "
            "\\\\\nmodel & 4 / 64 / 512 tokens & bandwidth & 4-token run \\\\\n\\midrule\n")
    (HERE / "tab_reuse.tex").write_text(head + body + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    design()
    teaser()
    capacity()
    reuse_table()
    print("ok")
