#!/usr/bin/env python3
"""
make_open_vs_closed_figures.py — open-loop vs closed-loop ablation figures

Reads TWO eval sweep dirs produced by eval_closed_loop.py — one closed-loop and
one open-loop (--open_loop) run of the SAME Fix F architecture — and renders a
side-by-side comparison. Pure numpy + matplotlib; runs on a login node.

The comparison is the controlled experiment: closed-loop reads the achieved
force F(t) each frame; open-loop is trained and evaluated with that feedback
channel blanked (train_policy.py --zero_force_now + run_closed_loop.py
--open_loop). Everything else (data, split, architecture, hyperparameters) is
identical, so the gap isolates the value of force feedback.

Inputs:
  --closed_dir  (default results/eval_closed_loop_fixF)   summary.json + *.npz
  --open_dir    (default results/eval_open_loop_fixF)     summary.json + *.npz
Outputs (results/figures/open_vs_closed/):
  01_open_vs_closed_bars.png    grouped bars, err_ratio per material × profile
  02_tracking_overlays.png      achieved-force magnitude, closed vs open vs goal
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("figures_ovc")

SCRIPT_DIR = Path(__file__).resolve().parent
MY_WORK = SCRIPT_DIR.parent
RESULTS = MY_WORK / "results"

MATERIALS = ["rope", "cloth", "sloth"]
PROFILES = ["policy_recorded_goal", "policy_ramp"]
PROF_LABEL = {"policy_recorded_goal": "replay", "policy_ramp": "ramp"}

CLOSED_COLOR = "#1f77b4"   # blue  — closed-loop (with feedback)
OPEN_COLOR = "#d62728"     # red   — open-loop (no feedback)

# Representative cases for the tracking overlays: one per material + the OOD
# rope ramp (the release-edge story). (case, profile, title).
OVERLAY_CASES = [
    ("single_push_rope_1", "policy_recorded_goal", "rope replay"),
    ("double_lift_cloth_1", "policy_recorded_goal", "cloth replay (best case)"),
    ("single_lift_sloth", "policy_recorded_goal", "sloth replay"),
    ("single_push_rope_1", "policy_ramp", "rope ramp (OOD / release edge)"),
]


def load_summary(eval_dir: Path) -> dict:
    with open(eval_dir / "summary.json") as f:
        return json.load(f)


def find_rollout(eval_dir: Path, case: str, profile: str):
    """Load the per-case npz, tolerating the __openloop suffix."""
    for name in (f"{case}__{profile}.npz", f"{case}__{profile}__openloop.npz"):
        p = eval_dir / name
        if p.exists():
            return np.load(p, allow_pickle=True)
    # last resort: glob
    hits = sorted(eval_dir.glob(f"{case}__{profile}*.npz"))
    return np.load(hits[0], allow_pickle=True) if hits else None


def force_mag(forces: np.ndarray) -> np.ndarray:
    """forces [T, n_ctrl, 3] -> ‖sum of groups‖ per frame [T]."""
    return np.linalg.norm(forces.sum(axis=1), axis=-1)


def goal_mag(F_goal: np.ndarray, n: int) -> np.ndarray:
    """F_goal [T, 2, 3] -> ‖sum of active groups‖ per frame [T]."""
    return np.linalg.norm(F_goal[:, :n].sum(axis=1), axis=-1)


# ---------- Figure 01: grouped err_ratio bars (closed vs open) -------------

def fig_bars(closed: dict, open_: dict, out_path: Path):
    cm = closed["per_material"]
    om = open_["per_material"]

    groups, c_means, c_stds, o_means, o_stds = [], [], [], [], []
    for mat in MATERIALS:
        for prof in PROFILES:
            if mat not in cm or prof not in cm[mat]:
                continue
            groups.append(f"{mat}\n{PROF_LABEL[prof]}")
            c_means.append(cm[mat][prof]["mean_force_err_ratio"])
            c_stds.append(cm[mat][prof].get("std_force_err_ratio", 0.0))
            if mat in om and prof in om[mat]:
                o_means.append(om[mat][prof]["mean_force_err_ratio"])
                o_stds.append(om[mat][prof].get("std_force_err_ratio", 0.0))
            else:
                o_means.append(np.nan)
                o_stds.append(0.0)

    x = np.arange(len(groups))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x - w / 2, c_means, w, yerr=c_stds, capsize=3,
                color=CLOSED_COLOR, edgecolor="black", alpha=0.88,
                label="closed-loop (with force feedback)")
    b2 = ax.bar(x + w / 2, o_means, w, yerr=o_stds, capsize=3,
                color=OPEN_COLOR, edgecolor="black", alpha=0.88,
                label="open-loop (no force feedback)")

    # annotate each bar with its value (clipping label height for tall bars)
    ymax = np.nanmax([np.nanmax(c_means), np.nanmax(o_means)])
    cap = max(7.0, ymax * 1.18)
    for bars, vals in ((b1, c_means), (b2, o_means)):
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, min(v, cap) + cap * 0.012,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.6,
               label="err_ratio = 1 (control breakdown)")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel("force error ratio  (mean ‖F_a − F_g‖ / mean ‖F_g‖)")
    ax.set_title("Force feedback ablation: same Fix F policy, with vs without "
                 "feedback  (lower is better)")
    ax.set_ylim(0, cap)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


# ---------- Figure 02: per-case force-magnitude overlays -------------------

def fig_overlays(closed_dir: Path, open_dir: Path, summaries, out_path: Path):
    closed_pr = summaries[0]["per_rollout"]
    open_pr = summaries[1]["per_rollout"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    for ax, (case, profile, title) in zip(axes, OVERLAY_CASES):
        dc = find_rollout(closed_dir, case, profile)
        do = find_rollout(open_dir, case, profile)
        if dc is None or do is None:
            ax.set_title(f"{title}\n(missing rollout)")
            ax.axis("off")
            continue
        n = int(dc["n_ctrl_parts"])
        g = goal_mag(dc["F_goal"], n) / 1e3                  # kN
        fc = force_mag(dc["forces"]) / 1e3
        fo = force_mag(do["forces"]) / 1e3
        t = np.arange(len(g))

        ax.plot(t, g, color="black", linestyle="--", linewidth=2, label="goal F*")
        ax.plot(t[:len(fc)], fc, color=CLOSED_COLOR, linewidth=1.8,
                label="closed-loop")
        ax.plot(t[:len(fo)], fo, color=OPEN_COLOR, linewidth=1.8,
                label="open-loop")

        key = f"{case}__{profile}"
        er_c = closed_pr.get(key, {}).get("force_err_ratio", float("nan"))
        er_o = open_pr.get(key, {}).get("force_err_ratio", float("nan"))
        ax.set_title(f"{title}\nerr_ratio: closed {er_c:.2f}  ·  open {er_o:.2f}",
                     fontsize=10)
        ax.set_xlabel("frame")
        ax.set_ylabel("‖F‖ (kN)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Achieved force vs goal — closed-loop corrects, open-loop "
                 "drifts off", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--closed_dir", type=Path,
                    default=RESULTS / "eval_closed_loop_fixF")
    ap.add_argument("--open_dir", type=Path,
                    default=RESULTS / "eval_open_loop_fixF")
    ap.add_argument("--out_dir", type=Path,
                    default=RESULTS / "figures" / "open_vs_closed")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    closed = load_summary(args.closed_dir)
    open_ = load_summary(args.open_dir)
    if not open_.get("open_loop", False):
        logger.warning("--open_dir summary.json has open_loop != true (%s); "
                        "is this really the open-loop sweep?", args.open_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_bars(closed, open_, args.out_dir / "01_open_vs_closed_bars.png")
    fig_overlays(args.closed_dir, args.open_dir, (closed, open_),
                 args.out_dir / "02_tracking_overlays.png")
    logger.info("done — figures in %s", args.out_dir)


if __name__ == "__main__":
    main()
