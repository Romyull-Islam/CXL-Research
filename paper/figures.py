"""Figures for the CAL letter from the gem5 results (gem5-model/results/*.csv). Latency is reported as
Table II in the letter; latency() is kept for reference.
Run: python paper/figures.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
G = HERE.parent / "gem5-model" / "results"
STYLE = {"flush": ("flush poll", "#888888", "s"), "uc": ("uncached poll", "#55A868", "v"),
         "lines": ("line poll", "#4C72B0", "^"), "parked": ("parked flag", "#DD8452", "D"),
         "wbpark": ("parked line fill", "#C44E52", "o")}
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3,
                     "legend.fontsize": 6, "legend.frameon": False})


def latency():
    d = pd.read_csv(G / "oneway.csv")
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.6), sharey=False)
    for ax, dev, title in zip(axes, ("expander", "switch"), ("250 ns expander", "600 ns switch")):
        x = d[d.device == dev]
        for p, (lab, c, mk) in STYLE.items():
            y = x[x.protocol == p].sort_values("size")
            ax.plot(y["size"], y.latency_ns / 1e3, color=c, marker=mk, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 256, 1024, 4096], ["64", "256", "1K", "4K"])
        ax.set_xlabel("message bytes")
        ax.set_title(title, fontsize=7)
        ax.grid(alpha=0.3, linewidth=0.4)
    axes[0].set_ylabel("one-way latency (µs)")
    axes[0].legend(loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "fig_latency.pdf")


def contention():
    d = pd.read_csv(G / "contention.csv")
    lw = pd.read_csv(G / "longwait.csv")
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.55))
    base = d[d.waiters == 0]
    for p in ("flush", "uc", "parked", "wbpark"):
        lab, c, mk = STYLE[p]
        x = pd.concat([base, d[d.protocol == p]]).sort_values("waiters")
        axes[0].plot(x.waiters, x.tenant_GBs, color=c, marker=mk, label=lab)
        y = lw[lw.protocol == p].sort_values("mean_wait_us")
        axes[1].plot(y.mean_wait_us, y.reads_per_msg, color=c, marker=mk, label=lab)
    axes[0].set_ylabel("tenant GB/s")
    axes[0].set_xlabel("waiting threads")
    axes[0].set_ylim(0, 50)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xticks([20, 100, 500], ["20", "100", "500"])
    axes[1].set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    axes[1].minorticks_off()
    axes[1].set_xlabel("mean wait (µs)")
    axes[1].set_ylabel("reads per message")
    for ax in axes:
        ax.grid(alpha=0.3, linewidth=0.4)
    axes[0].legend(loc="lower left")
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "fig_contention.pdf")


if __name__ == "__main__":
    contention()
