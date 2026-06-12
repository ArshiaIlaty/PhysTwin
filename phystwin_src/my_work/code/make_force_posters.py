#!/usr/bin/env python3
"""make_force_posters.py — generate force-tracking PNG posters for the
demo video slides. Each PNG plots achieved vs goal force magnitude over
time, per active gripper.

Used as poster_frame_image in build_demo_video_slides.py so the static
view (before play) shows the actual tracking story instead of an
unrelated cartoon.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "presentation_results" / "figures" / "posters"
OUT.mkdir(parents=True, exist_ok=True)


def force_poster(rollout_npz: Path, out_png: Path, title: str,
                  accent: str = "#16a34a"):
    """Plot achieved (colored) vs goal (black) force magnitude per gripper."""
    d = np.load(rollout_npz, allow_pickle=True)
    forces = d["forces"]               # [T, n_ctrl, 3]
    F_goal = d["F_goal"]               # [T, 2, 3]
    n_ctrl = int(d["n_ctrl_parts"])
    T = forces.shape[0]
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for g in range(n_ctrl):
        a_mag = np.linalg.norm(forces[:, g], axis=-1)
        g_mag = np.linalg.norm(F_goal[:, g], axis=-1)
        label_a = f"achieved" + (f"  (gripper {g+1})" if n_ctrl > 1 else "")
        label_g = f"goal"     + (f"  (gripper {g+1})" if n_ctrl > 1 else "")
        ax.plot(t, g_mag, color="#222", linewidth=2.5,
                linestyle="-" if g == 0 else "--",
                label=label_g)
        ax.plot(t, a_mag, color=accent, linewidth=2.5,
                linestyle="-" if g == 0 else "--",
                label=label_a)

    ax.set_xlabel("frame", fontsize=13, color="#222")
    ax.set_ylabel("force magnitude (N)", fontsize=13, color="#222")
    ax.set_title(title, fontsize=15, fontweight="bold", color="#111", pad=10)
    ax.tick_params(colors="#222")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#444")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=False, fontsize=11)

    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")


def main():
    results = ROOT / "results"
    # Slide A — descriptor effect on rope replay (single_lift_rope)
    force_poster(
        results / "eval_closed_loop" / "single_lift_rope__policy_recorded_goal.npz",
        OUT / "poster_A_left_BC_rope_replay.png",
        "BC  (no descriptor)  ·  rope replay  ·  err 0.36",
        accent="#6b7280",
    )
    force_poster(
        results / "eval_closed_loop_fixF" / "single_lift_rope__policy_recorded_goal.npz",
        OUT / "poster_A_right_BCdesc_rope_replay.png",
        "BC + 2-d descriptor  ·  rope replay  ·  err 0.13",
        accent="#2563eb",
    )

    # Slide B — RL effect on rope ramp (single_push_rope_1)
    force_poster(
        results / "eval_closed_loop_fixF" / "single_push_rope_1__policy_ramp.npz",
        OUT / "poster_B_left_BCdesc_rope_ramp.png",
        "BC + 2-d descriptor  ·  rope ramp  ·  release frac 0.04",
        accent="#2563eb",
    )
    force_poster(
        results / "eval_rl_fixF_multi" / "single_push_rope_1__policy_ramp.npz",
        OUT / "poster_B_right_RL_rope_ramp.png",
        "RL + 2-d descriptor  ·  rope ramp  ·  release frac 0.87",
        accent="#16a34a",
    )


if __name__ == "__main__":
    main()
