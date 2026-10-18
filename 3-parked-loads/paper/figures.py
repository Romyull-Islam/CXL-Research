"""Figures for the CAL letter from the gem5 results (gem5-model/results/*.csv). Latency is reported as
Table II in the letter; latency() is kept for reference.
Run: python paper/figures.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import figfont  # noqa: E402

HERE = Path(__file__).resolve().parent
G = HERE.parent / "gem5-model" / "results"
# Okabe-Ito colors; polling dashed, parked solid
STYLE = {"flush": ("flush poll", "#000000", "s"), "uc": ("uncached poll", "#56B4E9", "v"),
         "lines": ("line poll", "#009E73", "^"), "parked": ("parked flag", "#E69F00", "D"),
         "wbpark": ("parked line fill", "#D55E00", "o"),
         "b1000": ("backoff, 1 us cap", "#0072B2", "<"), "b4000": ("backoff, 4 us cap", "#CC79A7", ">")}
figfont.use(figfont.TIMES, "Nimbus Roman TT")
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3,
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
    d = pd.concat([pd.read_csv(G / "contention.csv"), pd.read_csv(G / "bo_contention.csv")])
    lw = pd.concat([pd.read_csv(G / "longwait.csv"), pd.read_csv(G / "bo_longwait.csv")])
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75))
    base = d[d.waiters == 0]
    for p in ("flush", "uc", "b1000", "b4000", "parked", "wbpark"):
        lab, c, mk = STYLE[p]
        x = pd.concat([base, d[d.protocol == p]]).sort_values("waiters")
        kw = dict(color=c, marker=mk, label=lab.replace(" us", " \u00b5s"),
                  mfc="white" if p in ("flush", "parked", "b4000") else c, mew=0.8,
                  ls="-" if p in ("parked", "wbpark") else "--")
        axes[0].plot(x.waiters, x.tenant_GBs, **kw)
        y = lw[lw.protocol == p].sort_values("mean_wait_us")
        axes[1].plot(y.mean_wait_us, y.reads_per_msg, **kw)
    axes[0].axhline(float(base.tenant_GBs.iloc[0]), color="0.5", lw=0.6, ls=":")
    axes[0].text(260, float(base.tenant_GBs.iloc[0]) + 1.2, "alone", fontsize=6.5, ha="right", color="0.35")
    axes[0].set_ylabel("tenant bandwidth (GB/s)")
    axes[0].set_xlabel("waiting threads")
    axes[0].set_ylim(0, 50)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xticks([20, 100, 500], ["20", "100", "500"])
    axes[1].set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    axes[1].minorticks_off()
    axes[1].set_xlabel("mean wait (µs)")
    axes[1].set_ylabel("waiting reads per message")
    for ax, t in zip(axes, "ab"):
        ax.set_title(f"({t})", fontsize=7, loc="left", pad=2)
    for ax in axes:
        ax.grid(alpha=0.3, linewidth=0.4)
    axes[0].legend(loc="lower left", fontsize=6.5, handlelength=2.2, ncol=1, borderaxespad=0.2,
                   labelspacing=0.25)
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "fig_contention.pdf", bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    contention()
