"""
Live plot for tribal-village autoresearch.
Run in a separate terminal: watch -n 5 python plot.py
Or: python plot.py --watch (re-renders every 5s)
Saves plot.png in the repo root.
"""
import argparse
import os
import re
import time
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
RUN_LOG = os.path.join(REPO, "run.log")
RESULTS_TSV = os.path.join(REPO, "results.tsv")
OUT_PNG = os.path.join(REPO, "plot.png")


def parse_run_log(path):
    """Return list of (update, effrank_n, d_act, elapsed) from in-progress run.log."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            m = re.match(
                r"update=(\d+) steps=\d+ elapsed=(\d+)s effrank_n=([0-9.nan]+) d_act=([0-9.nan]+)",
                line.strip(),
            )
            if m:
                rows.append((int(m.group(1)), float(m.group(3)),
                             float(m.group(4)), int(m.group(2))))
    return rows


def parse_results_tsv(path):
    """Return list of dicts from results.tsv."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return rows
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) < 6:
            continue
        try:
            rows.append({
                "commit": parts[0],
                "effrank_n": float(parts[1]),
                "probe_acc": float(parts[2]),
                "reward_type": parts[3],
                "status": parts[4],
                "desc": parts[5],
            })
        except ValueError:
            continue
    return rows


def render(once=False):
    while True:
        run_rows = parse_run_log(RUN_LOG)
        exp_rows = parse_results_tsv(RESULTS_TSV)

        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor("#0f0f0f")
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        ax_rank  = fig.add_subplot(gs[0, 0])  # effrank_n this run
        ax_dact  = fig.add_subplot(gs[0, 1])  # d_act this run
        ax_hill  = fig.add_subplot(gs[1, 0])  # hillclimb across experiments
        ax_probe = fig.add_subplot(gs[1, 1])  # probe_acc across experiments

        for ax in [ax_rank, ax_dact, ax_hill, ax_probe]:
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="#cccccc", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")

        def style(ax, title, xlabel, ylabel):
            ax.set_title(title, color="#ffffff", fontsize=9, pad=4)
            ax.set_xlabel(xlabel, color="#888888", fontsize=8)
            ax.set_ylabel(ylabel, color="#888888", fontsize=8)
            ax.grid(True, color="#2a2a2a", linewidth=0.5)

        # — current run: effrank_n over updates —
        style(ax_rank, "effrank_n  (this run)", "update", "effrank_n")
        if run_rows:
            updates = [r[0] for r in run_rows]
            ranks   = [r[1] for r in run_rows]
            ax_rank.plot(updates, ranks, color="#00ccff", linewidth=1.5, marker="o", markersize=3)
            ax_rank.axhline(1.0, color="#ff6666", linewidth=0.8, linestyle="--", label="n_agents threshold")
            ax_rank.set_ylim(bottom=0)
            ax_rank.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="#cccccc")
        else:
            ax_rank.text(0.5, 0.5, "waiting for data…", transform=ax_rank.transAxes,
                         ha="center", color="#666666", fontsize=9)

        # — current run: d_act over updates —
        style(ax_dact, "d_act  (action diversity, this run)", "update", "mean KL")
        if run_rows:
            updates = [r[0] for r in run_rows]
            dacts   = [r[2] for r in run_rows]
            ax_dact.plot(updates, dacts, color="#ffaa00", linewidth=1.5, marker="o", markersize=3)
            ax_dact.set_ylim(bottom=0)
        else:
            ax_dact.text(0.5, 0.5, "waiting for data…", transform=ax_dact.transAxes,
                         ha="center", color="#666666", fontsize=9)

        # — hillclimb: effrank_n per completed experiment —
        style(ax_hill, "effrank_n  (all experiments)", "experiment #", "effrank_n")
        if exp_rows:
            colors = {"individual": "#00ccff", "shared": "#ff4444", "mixed": "#ffaa00"}
            for i, row in enumerate(exp_rows):
                c = colors.get(row["reward_type"], "#aaaaaa")
                marker = "o" if row["status"] == "keep" else "x"
                ax_hill.scatter(i, row["effrank_n"], color=c, marker=marker, s=60, zorder=3)
            ax_hill.axhline(1.0, color="#ff6666", linewidth=0.8, linestyle="--")
            ax_hill.set_xticks(range(len(exp_rows)))
            ax_hill.set_xticklabels([r["commit"][:5] for r in exp_rows], rotation=45, fontsize=7)
            # legend
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=v, label=k, markersize=6)
                       for k, v in colors.items()]
            ax_hill.legend(handles=handles, fontsize=7, facecolor="#1a1a1a", labelcolor="#cccccc")
        else:
            ax_hill.text(0.5, 0.5, "no completed experiments yet", transform=ax_hill.transAxes,
                         ha="center", color="#666666", fontsize=9)

        # — probe_acc per experiment —
        style(ax_probe, "probe_acc  (all experiments)", "experiment #", "accuracy")
        if exp_rows:
            colors = {"individual": "#00ccff", "shared": "#ff4444", "mixed": "#ffaa00"}
            for i, row in enumerate(exp_rows):
                c = colors.get(row["reward_type"], "#aaaaaa")
                marker = "o" if row["status"] == "keep" else "x"
                ax_probe.scatter(i, row["probe_acc"], color=c, marker=marker, s=60, zorder=3)
            # chance line
            chance = 1.0 / 3 if any(r.get("assign_roles") for r in exp_rows) else 0.5
            ax_probe.axhline(0.5, color="#ff6666", linewidth=0.8, linestyle="--", label="chance (binary)")
            ax_probe.axhline(1/3, color="#ff9900", linewidth=0.8, linestyle=":", label="chance (3-way)")
            ax_probe.set_ylim(0, 1.05)
            ax_probe.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="#cccccc")
        else:
            ax_probe.text(0.5, 0.5, "no completed experiments yet", transform=ax_probe.transAxes,
                         ha="center", color="#666666", fontsize=9)

        # timestamp
        fig.text(0.01, 0.01, f"updated {time.strftime('%H:%M:%S')}", color="#444444", fontsize=7)

        plt.savefig(OUT_PNG, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[plot] saved {OUT_PNG}", flush=True)

        if once:
            break
        time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="re-render every 5s until Ctrl-C")
    args = parser.parse_args()
    render(once=not args.watch)
