"""Aggregate per-run result JSONs into paper table numbers (mean +/- std).

Tribal Village (Table 1): individual / mixed / shared over EffRank/n, D_act, probe.
SMAC (Table 2): individual / shared over EffRank/n, D_act, probe (+ win rate).
"""
import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"


def load(pattern):
    out = []
    for f in sorted(glob.glob(str(RES / pattern))):
        try:
            out.append(json.load(open(f)))
        except Exception as e:
            print(f"skip {f}: {e}")
    return out


def agg(rows, keys):
    vals = {k: np.array([r[k] for r in rows if r.get(k) is not None], float) for k in keys}
    return {k: (float(v.mean()), float(v.std())) for k, v in vals.items() if len(v)}


def fmt(m, k):
    if k not in m:
        return "--"
    mu, sd = m[k]
    return f"{mu:.3f} \\pm {sd:.3f}"


def tribal():
    print("\n=== TABLE 1: Tribal Village ===")
    print(f"{'Condition':<11} {'EffRank/n':<20} {'D_act':<22} {'Probe':<20} gate")
    rows_by = {}
    for cond in ("individual", "mixed", "shared"):
        rows = load(f"tv_{cond}_seed*.json")
        if not rows:
            print(f"{cond:<11} (no results yet)")
            continue
        m = agg(rows, ["effrank_per_agent", "d_act", "probe_accuracy"])
        gate = sum(int(r.get("gate_passed", False)) for r in rows)
        print(f"{cond:<11} {fmt(m,'effrank_per_agent'):<20} {fmt(m,'d_act'):<22} "
              f"{fmt(m,'probe_accuracy'):<20} {gate}/{len(rows)}")
        rows_by[cond] = m
    return rows_by


def smac():
    print("\n=== TABLE 2: SMACv2 10gen_terran ===")
    print(f"{'Condition':<11} {'EffRank/n':<20} {'D_act':<22} {'Probe':<20} {'WinRate':<12}")
    for cond in ("individual", "shared"):
        rows = load(f"smac_{cond}_seed*.json")
        if not rows:
            print(f"{cond:<11} (no results yet)")
            continue
        m = agg(rows, ["effrank_per_agent", "d_act", "probe_accuracy", "win_rate"])
        print(f"{cond:<11} {fmt(m,'effrank_per_agent'):<20} {fmt(m,'d_act'):<22} "
              f"{fmt(m,'probe_accuracy'):<20} {fmt(m,'win_rate'):<12}")


if __name__ == "__main__":
    tribal()
    smac()
