#!/usr/bin/env python3
"""make_presentation_assets.py — closed-loop diagram + BC vs RL bar chart.

Outputs PNGs into my_work/presentation_results/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation_results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Light theme defaults
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#222",
    "axes.titlecolor": "#222",
    "xtick.color": "#222",
    "ytick.color": "#222",
    "text.color": "#222",
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ----------------------------- closed-loop diagram -------------------------

def closed_loop_diagram(path: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5)
    ax.axis("off")

    # Two boxes
    def box(x, y, w, h, label, sub, fill):
        r = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.15",
            linewidth=1.6, edgecolor="#222", facecolor=fill,
        )
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2 + 0.25, label,
                ha="center", va="center", fontsize=15, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.35, sub,
                ha="center", va="center", fontsize=11, color="#444")

    box(0.6, 1.7, 3.2, 1.6, "POLICY", "MLP / RL actor", "#dbeafe")
    box(7.2, 1.7, 3.2, 1.6, "SIMULATOR", "PhysTwin (warp)", "#fde68a")

    # Forward arrow (top)
    ax.annotate("",
                xy=(7.2, 3.0), xytext=(3.8, 3.0),
                arrowprops=dict(arrowstyle="-|>", color="#222", lw=2))
    ax.text(5.5, 3.25, "action  Δ gripper",
            ha="center", fontsize=12, color="#222")

    # Feedback arrow (bottom)
    ax.annotate("",
                xy=(3.8, 2.0), xytext=(7.2, 2.0),
                arrowprops=dict(arrowstyle="-|>", color="#c2410c", lw=2))
    ax.text(5.5, 1.65,
            "feedback: state + achieved force F(t)",
            ha="center", fontsize=12, color="#c2410c")

    # Inputs to policy
    ax.text(2.2, 0.85, "+ goal F*(t+1)",
            ha="center", fontsize=11, color="#0369a1")
    ax.annotate("",
                xy=(2.2, 1.7), xytext=(2.2, 1.05),
                arrowprops=dict(arrowstyle="-|>", color="#0369a1", lw=1.4))

    # Title strip
    ax.text(5.5, 4.5, "Closed-loop force tracking",
            ha="center", fontsize=18, fontweight="bold", color="#111")
    ax.text(5.5, 4.1,
            "every frame: policy → action → sim → new force → back to policy",
            ha="center", fontsize=12, color="#555", style="italic")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ----------------------------- pipeline diagram ----------------------------

def pipeline_diagram(path: Path):
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3)
    ax.axis("off")

    stages = [
        ("Real PhysTwin\nrollouts (14)",  "#fde68a"),
        ("+ 262 synthetic\ntrajectories",  "#fbbf24"),
        ("Policy dataset\n33K rows",      "#fcd34d"),
        ("BC training\n(supervised)",     "#dbeafe"),
        ("PPO + BC\nwarm-start",          "#a7f3d0"),
        ("Closed-loop\nevaluation",  "#86efac"),
    ]
    w = 1.75
    x = 0.2
    gap = 0.18
    for i, (label, color) in enumerate(stages):
        r = mpatches.FancyBboxPatch(
            (x, 0.85), w, 1.3,
            boxstyle="round,pad=0.03,rounding_size=0.10",
            linewidth=1.2, edgecolor="#333", facecolor=color,
        )
        ax.add_patch(r)
        ax.text(x + w / 2, 1.5, label,
                ha="center", va="center", fontsize=11, fontweight="bold")
        if i < len(stages) - 1:
            ax.annotate("",
                        xy=(x + w + gap + 0.02, 1.5),
                        xytext=(x + w, 1.5),
                        arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.4))
        x += w + gap

    ax.text(6, 2.7, "Pipeline",
            ha="center", fontsize=16, fontweight="bold", color="#111")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ----------------------------- BC vs RL bar chart --------------------------

def bc_vs_rl_bars(path: Path):
    """Per-(material, profile) err_ratio, four series:
       BC base · BC + 2-d desc · RL base (no desc) · RL + 2-d desc."""
    summaries = {
        "BC base":          ROOT / "results" / "eval_closed_loop" / "summary.json",
        "BC + 2-d desc":    ROOT / "results" / "eval_closed_loop_fixF" / "summary.json",
        "RL base":          ROOT / "results" / "rl_eval_multi" / "summary.json",
        "RL + 2-d desc":    ROOT / "results" / "eval_rl_fixF_multi" / "summary.json",
    }
    data = {}
    for k, p in summaries.items():
        s = json.load(open(p))
        data[k] = s["per_material"]

    combos = [
        ("rope",  "policy_recorded_goal", "rope replay"),
        ("rope",  "policy_ramp",          "rope ramp"),
        ("cloth", "policy_recorded_goal", "cloth replay"),
        ("cloth", "policy_ramp",          "cloth ramp"),
        ("sloth", "policy_recorded_goal", "sloth replay"),
        ("sloth", "policy_ramp",          "sloth ramp"),
    ]
    labels = [c[2] for c in combos]
    series = ["BC base", "BC + 2-d desc", "RL base", "RL + 2-d desc"]
    # cool tones for BC, warm tones for RL; lighter = no descriptor.
    colors = ["#cbd5e1", "#2563eb", "#fdba74", "#16a34a"]

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    x = np.arange(len(combos))
    width = 0.20

    for i, name in enumerate(series):
        vals = []
        for (m, p, _lbl) in combos:
            d = data[name].get(m, {}).get(p)
            vals.append(d["mean_force_err_ratio"] if d else np.nan)
        offset = (i - 1.5) * width  # center 4 bars around the x tick
        bars = ax.bar(x + offset, vals, width,
                      label=name, color=colors[i], edgecolor="#222", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.08,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9, color="#222")

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("force-error ratio  (lower is better)")
    ax.set_title("Closed-loop force tracking — BC base → +descriptor → RL",
                 fontsize=15, pad=12, fontweight="bold")
    ax.set_ylim(0, max(7.0, ax.get_ylim()[1]))
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ----------------------------- release-problem cartoon --------------------

def release_cartoon(path: Path):
    """Stylized 'BC fails to retract, RL does' force trace cartoon."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    t = np.linspace(0, 1, 200)
    goal = np.piecewise(t, [t < 0.4, (t >= 0.4) & (t < 0.6), t >= 0.6],
                        [lambda tt: 2.0 * tt / 0.4,
                         lambda tt: 2.0 - 2.0 * (tt - 0.4) / 0.2,
                         lambda tt: 0.0])
    # BC: tracks rise, fails to release (clamps at peak then sags slowly)
    bc = np.copy(goal)
    bc[t > 0.4] = 2.0 - 0.4 * (t[t > 0.4] - 0.4)
    bc = np.clip(bc, 0, 2.0)
    # RL: tracks rise, releases reasonably
    rl = np.copy(goal)
    overshoot = 0.18 * np.exp(-((t - 0.42) / 0.04) ** 2)
    lag = 0.15 * np.exp(-((t - 0.62) / 0.06) ** 2)
    rl = goal + overshoot - lag
    rl = np.clip(rl, 0, 2.3)

    ax.plot(t, goal, color="#222", linewidth=2.5, label="goal  F*(t)")
    ax.plot(t, bc,   color="#94a3b8", linewidth=2.5,
            label="BC  (fails to release)")
    ax.plot(t, rl,   color="#16a34a", linewidth=2.5,
            label="RL  (tracks + releases)")

    ax.annotate("BC clamps", xy=(0.7, 1.62), xytext=(0.4, 1.95),
                fontsize=11, color="#475569",
                arrowprops=dict(arrowstyle="->", color="#475569"))
    ax.annotate("RL retracts", xy=(0.78, 0.18), xytext=(0.55, 0.55),
                fontsize=11, color="#15803d",
                arrowprops=dict(arrowstyle="->", color="#15803d"))

    ax.set_xlabel("time")
    ax.set_ylabel("force magnitude")
    ax.set_title("The release problem — closed-loop ramp",
                 fontsize=15, pad=12, fontweight="bold")
    # Put legend OUTSIDE the plotted region so it can't cover the curves
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 2.5)
    ax.grid(alpha=0.25)
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def observation_breakdown(path: Path):
    """Stacked-block diagram of the 45-dim observation (titles + dims only)."""
    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")

    ax.text(6, 3.6, "What the policy sees each frame  (45-dim observation)",
            ha="center", fontsize=18, fontweight="bold", color="#111")

    blocks = [
        ("Deformation\nfeatures",    18, "#fcd34d"),
        ("Control-point\nfeatures",  13, "#dbeafe"),
        ("Achieved force\nF(t)",      6, "#fecaca"),
        ("Goal force\nF*(t+1)",       6, "#a7f3d0"),
        ("Material\ndescriptor",      2, "#e9d5ff"),
    ]
    # Equal-width boxes — the dim count is inside the box, no need to
    # encode it through the box width too (that made the 6-d / 2-d boxes
    # too small for their labels).
    x = 0.4
    span = 11.2
    w = span / len(blocks)
    gap = 0.08
    box_w = w - gap
    for label, dim, color in blocks:
        r = mpatches.FancyBboxPatch(
            (x, 1.5), box_w, 1.55,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.2, edgecolor="#333", facecolor=color,
        )
        ax.add_patch(r)
        ax.text(x + box_w / 2, 2.35, label,
                ha="center", va="center", fontsize=14, fontweight="bold")
        ax.text(x + box_w / 2, 1.78, f"{dim}-d",
                ha="center", va="center", fontsize=13, color="#222",
                fontweight="bold")
        x += w

    ax.annotate("", xy=(11.5, 0.55), xytext=(0.5, 0.55),
                arrowprops=dict(arrowstyle="-", color="#222", lw=1.2))
    ax.text(6, 0.25, "→  MLP (43 / 44 / 45 → 256 / 512 → 6)  →  action Δgripper",
            ha="center", fontsize=12, color="#222", style="italic")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    closed_loop_diagram(OUT / "00_closed_loop_diagram.png")
    pipeline_diagram(OUT / "00_pipeline_diagram.png")
    bc_vs_rl_bars(OUT / "00_bc_vs_rl_bars.png")
    release_cartoon(OUT / "00_release_problem.png")
    observation_breakdown(OUT / "00_observation_breakdown.png")
