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
LAB = ["70B\nH100", "70B\nH200", "70B\nB200", "QwQ\n2×H100", "QwQ\n4×H100", "Q3-235B\nH100", "R1\nH200"]
WL = {"bailian": "Bailian", "bo1": "best-of-1", "bo16": "best-of-16", "bo64": "best-of-64"}


def gains():
    d = pd.read_csv(R / "gate2_saturation.csv")
    fig, ax = plt.subplots(figsize=(3.5, 1.6))
    x = np.arange(len(DEP))
    cols = ["#999999", "#4C72B0", "#DD8452", "#C44E52"]
    for j, (w, lab) in enumerate(WL.items()):
        y, e = [], []
        for dep in DEP:
            g = {}
            for k in ("fa3", "fa2"):
                r = d[(d.deploy == dep) & (d.wname == w) & (d.kernel == k)].set_index("policy").throughput
                g[k] = (r["swap"] / r["recompute"] - 1) * 100
            y.append(g["fa3"])
            e.append(g["fa2"] - g["fa3"])
        ax.bar(x + (j - 1.5) * 0.2, y, 0.2, color=cols[j], label=lab, yerr=[np.zeros(len(e)), e],
               error_kw=dict(lw=0.5, capsize=1))
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


def teaser():
    """Model-predicted against measured gain: simulated preemption runs (gain = R / rho) and real
    vLLM document questions (Eq. 1 with the measured c_r and c_l)."""
    d = pd.read_csv(R / "gate2_saturation.csv")
    r = d[d.policy == "recompute"].copy()
    r["rho"] = ((r.busy_time - r.recompute_time) / r.out_tokens) / (r.recompute_time / r.recomputed_tokens)
    s = d[d.policy == "swap"].set_index(["deploy", "wname", "kernel"]).throughput
    r["gain"] = (s.loc[list(zip(r.deploy, r.wname, r.kernel))].values / r.throughput.values - 1) * 100
    r["model"] = r.recomputed_per_out / r.rho * 100
    q = pd.read_csv(R / "reuse" / "summary.csv")
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    lim = (1, 600)
    ax.plot(lim, lim, color="k", lw=0.5)
    ax.fill_between(lim, [v / 1.2 for v in lim], [v * 1.2 for v in lim], color="0.85", lw=0)
    ax.scatter(r.model, r.gain, s=7, color="#4C72B0", label="preempted KV, simulated (56 runs)", zorder=3)
    ax.scatter(q.gain_predicted * 100, q.gain_measured * 100, s=22, marker="D", color="#C44E52",
               label="reused KV, real vLLM", zorder=4)
    for _, row in q.iterrows():
        ax.annotate(rf"$R$ = {row.R:.0f}", (row.gain_predicted * 100, row.gain_measured * 100),
                    xytext=(-6, 3), textcoords="offset points", fontsize=6, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ticks = [1, 3, 10, 30, 100, 300]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("model gain (%)")
    ax.set_ylabel("observed gain (%)")
    ax.text(1.3, 300, "line: model = observed;\nband: within 20%", fontsize=6, va="top")
    ax.legend(loc="lower right")
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
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    for (w, pol), g in c.groupby(["wname", "policy"]):
        m = g.groupby("tier_mult").kept.median()
        ax.plot(m.index, m.values * 100, marker="o", ls="-" if pol == "swap" else "--",
                color={"bailian": "#4C72B0", "bo16": "#DD8452", "bo64": "#C44E52"}[w],
                label=f"{WL[w]}, {'swap' if pol == 'swap' else 'LRU write-through'}")
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1, 2, 4], ["0.05", "0.1", "0.25", "0.5", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_xlabel("tier capacity / HBM KV capacity")
    ax.set_ylabel("gain kept (%)")
    ax.legend(fontsize=6, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "fig_capacity.pdf", bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    gains()
    teaser()
    capacity()
    print("ok")
