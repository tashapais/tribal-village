"""
Autoresearch hillclimb plot — mirrors karpathy/autoresearch style.
Reads results.tsv and live run.log. Saves plot.png.

Usage:
  python plot.py              # render once
  python plot.py --watch      # re-render every 5s
"""
import argparse
import os
import re
import time
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

REPO    = os.path.dirname(os.path.abspath(__file__))
RUN_LOG = os.path.join(REPO, "run.log")
TSV     = os.path.join(REPO, "results.tsv")
OUT     = os.path.join(REPO, "plot.png")

METRIC  = "effrank_n"   # higher is better
CHANCE  = 1/3           # probe_acc chance level for 3-way classification


def parse_tsv():
    rows = []
    if not os.path.exists(TSV):
        return rows
    with open(TSV) as f:
        lines = f.readlines()
    for line in lines[1:]:
        p = line.strip().split("\t")
        if len(p) < 6:
            continue
        try:
            rows.append({
                "commit":      p[0],
                "effrank_n":   float(p[1]),
                "probe_acc":   float(p[2]),
                "reward_type": p[3],
                "status":      p[4],
                "desc":        p[5],
            })
        except ValueError:
            continue
    return rows


def parse_live_run():
    """Return (reward_type, list of (update, effrank_n)) from run.log."""
    reward_type = "?"
    pts = []
    if not os.path.exists(RUN_LOG):
        return reward_type, pts
    with open(RUN_LOG) as f:
        for line in f:
            m = re.search(r"reward=(\w+)", line)
            if m:
                reward_type = m.group(1)
            m = re.match(r"update=(\d+).*effrank_n=([0-9.]+)", line)
            if m:
                pts.append((int(m.group(1)), float(m.group(2))))
    return reward_type, pts


def render(once=False):
    while True:
        rows = parse_tsv()
        live_reward, live_pts = parse_live_run()

        n_total = len(rows)
        n_kept  = sum(1 for r in rows if r["status"] == "keep")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                                 gridspec_kw={"width_ratios": [2, 1]})
        fig.patch.set_facecolor("#ffffff")

        # ── left panel: hillclimb ──────────────────────────────────────────
        ax = axes[0]
        ax.set_facecolor("#ffffff")

        COLOR_KEPT    = "#2ecc71"
        COLOR_DISCARD = "#cccccc"
        COLOR_BEST    = "#27ae60"
        REWARD_COLORS = {"individual": "#2980b9", "shared": "#e74c3c", "mixed": "#f39c12"}

        if rows:
            xs    = list(range(len(rows)))
            ys    = [r[METRIC] for r in rows]
            stati = [r["status"] for r in rows]

            # scatter all points
            for i, (x, y, s, r) in enumerate(zip(xs, ys, stati, rows)):
                rc = REWARD_COLORS.get(r["reward_type"], "#888888")
                if s == "keep":
                    ax.scatter(x, y, color=COLOR_KEPT, s=80, zorder=4,
                               edgecolors=rc, linewidths=2)
                else:
                    ax.scatter(x, y, color=COLOR_DISCARD, s=40, zorder=3,
                               alpha=0.6)

            # running best line through kept points
            best_x, best_y, best_rows = [], [], []
            cur_best = -np.inf
            for i, r in enumerate(rows):
                if r["status"] == "keep" and r[METRIC] > cur_best:
                    cur_best = r[METRIC]
                    best_x.append(i)
                    best_y.append(r[METRIC])
                    best_rows.append(r)

            if len(best_x) >= 1:
                # draw step line
                step_x = [best_x[0]]
                step_y = [best_y[0]]
                for bx, by in zip(best_x[1:], best_y[1:]):
                    step_x += [step_x[-1], bx]
                    step_y += [step_y[-1], by]
                ax.plot(step_x, step_y, color=COLOR_BEST, linewidth=2,
                        zorder=5, label="Running best")

                # labels on kept improvements
                for bx, by, br in zip(best_x, best_y, best_rows):
                    label = textwrap.shorten(br["desc"], width=32)
                    ax.annotate(
                        label, xy=(bx, by),
                        xytext=(10, 8), textcoords="offset points",
                        fontsize=7, color="#1a1a1a",
                        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8),
                    )

        # live run inset (current run progress)
        if live_pts:
            up  = [p[0] for p in live_pts]
            er  = [p[1] for p in live_pts]
            ax2 = ax.inset_axes([0.02, 0.02, 0.32, 0.28])
            ax2.set_facecolor("#f8f8f8")
            rc = REWARD_COLORS.get(live_reward, "#888888")
            ax2.plot(up, er, color=rc, linewidth=1.2)
            ax2.set_title(f"live: {live_reward}", fontsize=6, pad=2, color=rc)
            ax2.tick_params(labelsize=5)
            ax2.set_xlabel("update", fontsize=5)
            ax2.grid(True, linewidth=0.3, alpha=0.5)

        ax.set_title(
            f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements",
            fontsize=12, pad=10,
        )
        ax.set_xlabel("Experiment #", fontsize=10)
        ax.set_ylabel(f"{METRIC}  (higher is better)", fontsize=10)
        ax.grid(True, linewidth=0.4, alpha=0.4)
        if rows:
            all_y = [r[METRIC] for r in rows]
            pad = (max(all_y) - min(all_y)) * 0.3 + 0.05
            ax.set_ylim(max(0, min(all_y) - pad), max(all_y) + pad)

        # legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_DISCARD,
                   markersize=7, label="Discarded"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_KEPT,
                   markersize=7, label="Kept"),
            Line2D([0], [0], color=COLOR_BEST, linewidth=2, label="Running best"),
        ]
        for rtype, rc in REWARD_COLORS.items():
            handles.append(Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=rc, markersize=6,
                                  markeredgecolor=rc, label=rtype))
        ax.legend(handles=handles, fontsize=8, loc="lower right")

        # ── right panel: probe_acc ─────────────────────────────────────────
        ax3 = axes[1]
        ax3.set_facecolor("#ffffff")
        if rows:
            for i, r in enumerate(rows):
                rc = REWARD_COLORS.get(r["reward_type"], "#888888")
                mk = "o" if r["status"] == "keep" else "x"
                ax3.scatter(i, r["probe_acc"], color=rc, marker=mk, s=60, zorder=3)
            ax3.axhline(0.333, color="#e74c3c", linewidth=0.8, linestyle="--",
                        label="chance (3-way, 0.33)")
            ax3.axhline(0.500, color="#f39c12", linewidth=0.8, linestyle=":",
                        label="chance (binary, 0.50)")
            ax3.set_ylim(0, 1.05)
        ax3.set_title("probe_acc by experiment", fontsize=11, pad=10)
        ax3.set_xlabel("Experiment #", fontsize=10)
        ax3.set_ylabel("role probe accuracy", fontsize=10)
        ax3.grid(True, linewidth=0.4, alpha=0.4)
        ax3.legend(fontsize=7)

        fig.suptitle("autoresearch — tribal village", fontsize=14,
                     fontweight="bold", x=0.02, ha="left")
        fig.text(0.99, 0.01, f"updated {time.strftime('%H:%M:%S')}",
                 ha="right", fontsize=7, color="#888888")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(OUT, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {OUT}  ({n_total} experiments, {n_kept} kept)", flush=True)

        if once:
            break
        time.sleep(5)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true")
    render(once=not p.parse_args().watch)
