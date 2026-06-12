#!/usr/bin/env python3
"""
make_progression_figures.py — Experiment 4: the story of the project in figures.

Two analyses, both from existing eval sweeps (no new rollouts):

  (A) Progression — err_ratio across the chronological sequence of policies,
      one panel per (material, profile) group. Shows how each idea moved each
      group, and that cloth ramp is the standing frontier. The deployable
      ensemble (analysis_ensemble.py) is drawn as the final point.

  (B) Per-axis error — which force component (Fx/Fy/Fz) is hardest, aggregated
      by material. Computed from the achieved-vs-goal force traces in the npz
      rollouts of the deployable-selected policy per case.

Outputs:
  figures/ensemble/02_progression.png
  figures/ensemble/03_per_axis_error.png
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("progression")

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = SCRIPT_DIR.parent / "results"
FIG_DIR = RESULTS / "figures" / "ensemble"

# Chronological policy sequence (the actual development order).
PROGRESSION = [
    ("BC base",            "eval_closed_loop"),
    ("BC+1D",              "eval_closed_loop_fixE"),
    ("BC+2D",              "eval_closed_loop_fixF"),
    ("RL+2D",              "eval_rl_fixF_multi"),
    ("Transformer\n-noise", "eval_closed_loop_v2_noise"),
    ("FiLM-11D",           "eval_closed_loop_v2_fixK_film"),
]

GROUP_ORDER = [
    ("rope", "replay"), ("rope", "ramp"),
    ("cloth", "replay"), ("cloth", "ramp"),
    ("sloth", "replay"), ("sloth", "ramp"),
]
MAT_COLOR = {"rope": "#1f77b4", "cloth": "#ff7f0e", "sloth": "#2ca02c"}


def prof_bucket(p):
    return "ramp" if "ramp" in p else "replay"


def group_means(eval_dir):
    summ = json.load(open(RESULTS / eval_dir / "summary.json"))["per_rollout"]
    g = defaultdict(list)
    for v in summ.values():
        g[(v["material"], prof_bucket(v["profile"]))].append(v["force_err_ratio"])
    return {k: float(np.mean(x)) for k, x in g.items()}


# ============================ (A) progression ==============================
seq = [(name, group_means(d)) for name, d in PROGRESSION]
ens = json.load(open(RESULTS / "ensemble" / "ensemble_summary.json"))["per_group"]

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
xs = np.arange(len(PROGRESSION))
for ax, g in zip(axes.flat, GROUP_ORDER):
    ys = [gm.get(g, np.nan) for _, gm in seq]
    col = MAT_COLOR[g[0]]
    ax.plot(xs, ys, "-o", color=col, lw=2, ms=6)
    # annotate each point
    for x, y in zip(xs, ys):
        if not np.isnan(y):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color="#333")
    # deployable ensemble as a dashed reference line
    ens_val = ens[f"{g[0]}_{g[1]}"]["Selector-B (goal+mat)"]
    ax.axhline(ens_val, color="#2ca02c", ls="--", lw=1.3, alpha=0.8)
    ax.text(0.02, ens_val, f"ensemble {ens_val:.2f}", color="#2ca02c",
            fontsize=7, va="bottom", ha="left", transform=ax.get_yaxis_transform())
    ax.axhline(1.0, color="#d62728", lw=0.7, ls=":", alpha=0.5)
    ax.set_title(f"{g[0]} — {g[1]}", fontsize=11, color=col, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    best = np.nanmin(ys)
    ax.set_ylim(0, max(best * 1.4, max([y for y in ys if not np.isnan(y)]) * 1.15))

for ax in axes[-1]:
    ax.set_xticks(xs)
    ax.set_xticklabels([n for n, _ in PROGRESSION], fontsize=8, rotation=30, ha="right")
for ax in axes[:, 0]:
    ax.set_ylabel("err_ratio")

fig.suptitle("Project progression — err_ratio per group across the development sequence\n"
             "Green dashed = deployable goal-aware ensemble (final). Red dotted = uncontrolled (1.0).",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / "02_progression.png", dpi=150)
logger.info("wrote %s", FIG_DIR / "02_progression.png")

# ============================ (B) per-axis error ===========================
# Use the deployable-selected policy per case (from ensemble_summary picks).
picks = json.load(open(RESULTS / "ensemble" / "ensemble_summary.json"))["selector_b_picks"]
POLICY_DIR = {
    "BC+2D": "eval_closed_loop_fixF",
    "RL+2D": "eval_rl_fixF_multi",
    "Transformer-noise": "eval_closed_loop_v2_noise",
    "FiLM-11D": "eval_closed_loop_v2_fixK_film",
}
AXES = ["Fx", "Fy", "Fz"]
# per material: accumulate axis-wise abs error and axis-wise goal magnitude
err_acc = {m: np.zeros(3) for m in MAT_COLOR}
goal_acc = {m: np.zeros(3) for m in MAT_COLOR}
for case_key, pol in picks.items():
    d = np.load(RESULTS / POLICY_DIR[pol] / f"{case_key}.npz", allow_pickle=True)
    mat = str(d["material"])
    F = d["forces"].reshape(d["forces"].shape[0], -1, 3).sum(axis=1)   # net force [T,3]
    Fg = d["F_goal"].reshape(d["F_goal"].shape[0], -1, 3).sum(axis=1)  # net goal  [T,3]
    err_acc[mat] += np.abs(F - Fg).mean(axis=0)
    goal_acc[mat] += np.abs(Fg).mean(axis=0) + 1e-6

# normalize per-axis error by per-axis goal magnitude (per material)
norm_err = {m: err_acc[m] / goal_acc[m] for m in MAT_COLOR}

fig, ax = plt.subplots(figsize=(8.5, 4.8))
x = np.arange(3)
w = 0.25
for i, m in enumerate(["rope", "cloth", "sloth"]):
    vals = norm_err[m]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=m, color=MAT_COLOR[m],
                  edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(AXES)
ax.set_ylabel("normalized axis error\n(mean |F−F*| / mean |F*|, per axis)")
ax.set_title("Which force axis is hardest to control?\n"
             "Per-axis tracking error by material (deployable ensemble rollouts)")
ax.legend(title="material")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "03_per_axis_error.png", dpi=150)
logger.info("wrote %s", FIG_DIR / "03_per_axis_error.png")

# log the per-axis numbers
logger.info("\nPer-axis normalized error (material x [Fx,Fy,Fz]):")
for m in ["rope", "cloth", "sloth"]:
    logger.info("  %-6s %s", m, "  ".join(f"{a}={v:.2f}" for a, v in zip(AXES, norm_err[m])))
