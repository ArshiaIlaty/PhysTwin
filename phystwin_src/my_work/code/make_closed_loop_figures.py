#!/usr/bin/env python3
"""
make_closed_loop_figures.py — Step 5 preliminary figures

Generates 5 PNGs from the Step 4 rollout npzs + summary.json. Static plots
only (no video). Pure numpy + matplotlib; runs on login node.

Plan:    my_work/docs/closed_loop_control/step5_plan.md
Inputs:  my_work/results/eval_closed_loop/*.npz + summary.json
Outputs: my_work/results/figures/closed_loop/0[1-5]_*.png
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("figures_cl")

SCRIPT_DIR = Path(__file__).resolve().parent
MY_WORK = SCRIPT_DIR.parent
RESULTS = MY_WORK / "results"
EVAL_DIR = RESULTS / "eval_closed_loop"
OUT_DIR = RESULTS / "figures" / "closed_loop"

MATERIAL_COLORS = {"rope": "#1f77b4", "cloth": "#ff7f0e", "sloth": "#2ca02c", "toy": "#9467bd"}


def load_rollout(case, profile):
    path = EVAL_DIR / f"{case}__{profile}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return d


def total_force_mag(forces):
    """forces: [T, n_ctrl, 3]; returns ‖sum of groups‖ per frame [T]."""
    return np.linalg.norm(forces.sum(axis=1), axis=-1)


def goal_total_mag(F_goal, n_ctrl):
    """F_goal: [T, 2, 3]; returns ‖sum of active groups‖ per frame."""
    return np.linalg.norm(F_goal[:, :n_ctrl].sum(axis=1), axis=-1)


# ---------- Figure 01: per-material bars --------------------------------

def fig_per_material_bars(summary, out_path):
    pm = summary["per_material"]
    materials = ["rope", "cloth", "sloth"]
    profiles = ["policy_recorded_goal", "policy_ramp"]
    prof_label = {"policy_recorded_goal": "replay", "policy_ramp": "ramp"}

    fig, ax = plt.subplots(figsize=(9, 5))
    x = []
    means = []
    stds = []
    colors = []
    labels = []
    n_uncontrolled = []
    n_cases = []

    pos = 0
    xticks_pos, xticks_lab = [], []
    for mat in materials:
        if mat not in pm:
            continue
        for prof in profiles:
            if prof not in pm[mat]:
                continue
            agg = pm[mat][prof]
            x.append(pos)
            means.append(agg["mean_force_err_ratio"])
            stds.append(agg["std_force_err_ratio"])
            colors.append(MATERIAL_COLORS.get(mat, "gray"))
            labels.append(f"{mat}\n{prof_label[prof]}")
            n_uncontrolled.append(agg["n_uncontrolled"])
            n_cases.append(agg["n_cases"])
            xticks_pos.append(pos)
            xticks_lab.append(f"{mat}\n{prof_label[prof]}")
            pos += 1
        pos += 0.5  # spacer between materials

    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor="black",
                   width=0.8, capsize=4, alpha=0.85)
    # annotate uncontrolled count above each bar
    for i, b in enumerate(bars):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2,
                h + max(stds[i], 0.05) + 0.1,
                f"{n_uncontrolled[i]}/{n_cases[i]} ✗" if n_uncontrolled[i] > 0
                else f"{n_cases[i]} OK",
                ha="center", fontsize=9, color="black")

    ax.axhline(1.0, color="red", linestyle=":", linewidth=1, alpha=0.6,
               label="err_ratio = 1 (control breakdown)")
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks_lab, fontsize=10)
    ax.set_ylabel("mean force error ratio  (mean ‖F_a − F_g‖ / mean ‖F_g‖)")
    ax.set_title("Closed-loop tracking error by material × profile  (lower is better)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, max(7, max(means) * 1.2))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


# ---------- Figure 02: force tracking grid (controlled replay) ----------

def fig_tracking_grid(per_rollout, out_path):
    # 9 controlled replay rollouts (err_ratio < 1.0 AND profile == replay)
    controlled = [
        (k, m) for k, m in per_rollout.items()
        if m["profile"] == "policy_recorded_goal"
           and not m["any_nan"]
           and m["force_err_ratio"] < 1.0
    ]
    # sort by material, then by err_ratio
    controlled.sort(key=lambda kv: (kv[1]["material"], kv[1]["force_err_ratio"]))

    n = len(controlled)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.0 * nrows), sharex=False)
    axes = axes.flatten()

    for i, (k, m) in enumerate(controlled):
        case = m["case_name"]
        profile = m["profile"]
        ax = axes[i]
        d = load_rollout(case, profile)
        if d is None:
            ax.axis("off")
            continue
        forces = d["forces"]
        F_goal = d["F_goal"]
        n_ctrl = int(d["n_ctrl_parts"])
        T = forces.shape[0]
        t = np.arange(T)
        F_a = total_force_mag(forces) / 1000.0     # kN
        F_g = goal_total_mag(F_goal, n_ctrl) / 1000.0
        col = MATERIAL_COLORS.get(m["material"], "gray")
        ax.plot(t, F_g, linestyle="--", color="black", linewidth=1.2,
                label="goal", alpha=0.7)
        ax.plot(t, F_a, color=col, linewidth=1.8, label="achieved")
        ax.set_title(f"{case}\n{m['material']} · err={m['force_err_ratio']:.2f}",
                     fontsize=9)
        ax.set_xlabel("frame")
        ax.set_ylabel("‖F‖  (kN)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=8)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Closed-loop force tracking — controlled replay rollouts "
                  "(9 of 11)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s (%d controlled cases)", out_path, n)


# ---------- Figure 03: limitations panel --------------------------------

def fig_limitations(per_rollout, out_path):
    # Three failure modes: rope outlier, sloth outlier (replay), cloth ramp (blow-up)
    pick = [
        ("single_push_rope_4",  "policy_recorded_goal", "rope outlier\n(very low recorded force)"),
        ("double_stretch_sloth", "policy_recorded_goal", "sloth long-trajectory drift\n(192 frames)"),
        ("double_lift_cloth_3",  "policy_ramp",          "cloth ramp blow-up\n(novel target)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (case, profile, label) in zip(axes, pick):
        d = load_rollout(case, profile)
        if d is None:
            ax.set_title(f"missing: {case}__{profile}")
            ax.axis("off")
            continue
        forces = d["forces"]
        F_goal = d["F_goal"]
        n_ctrl = int(d["n_ctrl_parts"])
        T = forces.shape[0]
        t = np.arange(T)
        F_a = total_force_mag(forces) / 1000.0
        F_g = goal_total_mag(F_goal, n_ctrl) / 1000.0
        m = per_rollout.get(f"{case}__{profile}")
        col = MATERIAL_COLORS.get(str(d["material"]), "gray")
        ax.plot(t, F_g, linestyle="--", color="black", linewidth=1.2, label="goal")
        ax.plot(t, F_a, color=col, linewidth=2.0, label="achieved")
        title = f"{case}\n{label}"
        if m:
            title += f"\nerr_ratio = {m['force_err_ratio']:.2f}  overshoot = {m['peak_overshoot_ratio']:.1f}×"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("frame")
        ax.set_ylabel("‖F‖  (kN)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("Failure modes — 3 representative uncontrolled rollouts",
                  fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


# ---------- Figure 04: highlight (cloth best case) ----------------------

def fig_highlight_cloth(per_rollout, out_path):
    case = "double_lift_cloth_1"
    profile = "policy_recorded_goal"
    d = load_rollout(case, profile)
    if d is None:
        logger.warning("missing highlight case %s", case)
        return
    forces = d["forces"]
    F_goal = d["F_goal"]
    n_ctrl = int(d["n_ctrl_parts"])
    T = forces.shape[0]
    t = np.arange(T)
    # Per-axis decomposition: sum across groups
    F_a_sum = forces.sum(axis=1)              # [T, 3]
    F_g_sum = F_goal[:, :n_ctrl].sum(axis=1)  # [T, 3]
    m = per_rollout.get(f"{case}__{profile}", {})

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axis_names = ["Fx", "Fy", "Fz"]
    for ax_i, label in enumerate(axis_names):
        ax = axes[ax_i]
        ax.plot(t, F_g_sum[:, ax_i] / 1000, "k--", linewidth=1.3, label="goal", alpha=0.7)
        ax.plot(t, F_a_sum[:, ax_i] / 1000, color=MATERIAL_COLORS["cloth"],
                linewidth=2.0, label="achieved")
        ax.set_title(f"{label}", fontsize=11)
        ax.set_xlabel("frame")
        ax.set_ylabel(f"{label}  (kN)")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        if ax_i == 0:
            ax.legend(loc="best", fontsize=9)

    # 4th panel: magnitude
    F_a_mag = total_force_mag(forces) / 1000
    F_g_mag = goal_total_mag(F_goal, n_ctrl) / 1000
    axes[3].plot(t, F_g_mag, "k--", linewidth=1.3, label="goal", alpha=0.7)
    axes[3].plot(t, F_a_mag, color=MATERIAL_COLORS["cloth"], linewidth=2.0,
                  label="achieved")
    axes[3].set_title("‖F‖ (magnitude)", fontsize=11)
    axes[3].set_xlabel("frame")
    axes[3].set_ylabel("‖F‖  (kN)")
    axes[3].grid(True, alpha=0.3)

    err = m.get("force_err_ratio", float("nan"))
    fig.suptitle(f"Highlight: {case} (cloth) — closed-loop force tracking, "
                 f"per-axis decomposition\nerr_ratio = {err:.2f}  (best controlled case)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


# ---------- Figure 05: ramp comparison ----------------------------------

def fig_ramps(per_rollout, out_path):
    pick = [
        ("single_push_rope_1",    "rope (borderline, err=0.66)"),
        ("double_lift_cloth_3",   "cloth (blow-up, err=6.55)"),
        ("double_stretch_sloth",  "sloth (blow-up, err=5.06)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (case, label) in zip(axes, pick):
        d = load_rollout(case, "policy_ramp")
        if d is None:
            ax.set_title(f"missing: {case}__policy_ramp")
            ax.axis("off")
            continue
        forces = d["forces"]
        F_goal = d["F_goal"]
        n_ctrl = int(d["n_ctrl_parts"])
        T = forces.shape[0]
        t = np.arange(T)
        F_a = total_force_mag(forces) / 1000.0
        F_g = goal_total_mag(F_goal, n_ctrl) / 1000.0
        col = MATERIAL_COLORS.get(str(d["material"]), "gray")
        ax.plot(t, F_g, "k--", linewidth=1.4, label="goal (ramp 0→peak→0)")
        ax.plot(t, F_a, color=col, linewidth=2.0, label="achieved")
        ax.set_title(f"{case}\n{label}", fontsize=10)
        ax.set_xlabel("frame")
        ax.set_ylabel("‖F‖  (kN)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("Ramp profile — policy is one-directional, fails on release "
                  "and compliant materials", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


# ---------- main --------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = EVAL_DIR / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    per_rollout = summary["per_rollout"]

    fig_per_material_bars(summary,      OUT_DIR / "01_per_material_bars.png")
    fig_tracking_grid(per_rollout,      OUT_DIR / "02_force_tracking_grid.png")
    fig_limitations(per_rollout,        OUT_DIR / "03_limitations_panel.png")
    fig_highlight_cloth(per_rollout,    OUT_DIR / "04_highlight_cloth.png")
    fig_ramps(per_rollout,              OUT_DIR / "05_ramp_failure.png")

    print(f"\nwrote 5 figures to {OUT_DIR}")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
