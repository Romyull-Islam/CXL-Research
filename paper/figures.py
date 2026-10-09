"""Figures for the CAL letter from model/results_latency.csv and model/results_contention.csv.
Run: python paper/figures.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
M = HERE.parent / "model"
STYLE = {"poll_flag": ("flag poll", "#888888", "s"), "poll_lines": ("line poll", "#4C72B0", "^"),
         "coherent_spin": ("coherent spin", "#DD8452", "D"), "parked": ("parked", "#C44E52", "o")}
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3,
                     "legend.fontsize": 6, "legend.frameon": False})


def latency():
    d = pd.read_csv(M / "results_latency.csv")
    d = d[(d.exp == "one_way") & (d.rtt == 600.0)]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.55), sharey=True)
    for ax, ovh in zip(axes, (50.0, 1400.0)):
        x = d[d.poll_ovh == ovh]
        for p, (lab, c, mk) in STYLE.items():
            ax.plot(x["size"], x[p] / 1e3, color=c, marker=mk, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 1024, 4096, 16384], ["64", "1K", "4K", "16K"])
        ax.set_xlabel("message bytes")
        ax.set_title(f"poll cost {ovh / 1e3:g} us" if ovh >= 1000 else f"poll cost {ovh:g} ns", fontsize=7)
        ax.grid(alpha=0.3, linewidth=0.4)
    axes[0].set_ylabel("one-way latency (us)")
    axes[0].legend(loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "fig_latency.pdf")


def contention():
    d = pd.read_csv(M / "results_contention.csv")
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.5))
    for p in ("poll_flag", "poll_lines", "parked"):
        lab, c, mk = STYLE[p]
        x = d[(d.protocol == p) | ((d.waiters == 0) & (d.protocol == "parked"))].sort_values("waiters")
        axes[0].plot(x.waiters, x.tenant_GBs, color=c, marker=mk, label=lab)
        y = d[d.protocol == p].sort_values("waiters")
        axes[1].plot(y.waiters, y.handoff_ns / 1e3, color=c, marker=mk, label=lab)
    axes[0].set_ylabel("tenant GB/s")
    axes[1].set_ylabel("handoff (us)")
    for ax in axes:
        ax.set_xlabel("waiting threads")
        ax.grid(alpha=0.3, linewidth=0.4)
    axes[0].set_ylim(0, 50)
    axes[1].legend(loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "fig_contention.pdf")


if __name__ == "__main__":
    latency()
    contention()
