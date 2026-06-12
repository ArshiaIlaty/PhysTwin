#!/usr/bin/env python3
"""
make_release_real_figure.py — the release problem shown with real rollouts.

Replaces the stylized cartoon (00_release_problem.png) with measured force
traces from the actual eval rollouts on double_lift_cloth_3, ramp goal:
  BC  — results/eval_closed_loop/double_lift_cloth_3__policy_ramp.npz
  RL  — results/eval_rl_fixF_multi/double_lift_cloth_3__policy_ramp.npz

Output: results/figures/ensemble/00_release_problem_real.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS / "figures" / "ensemble"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CASE = "double_lift_cloth_3__policy_ramp.npz"


def net_mag(arr):
    """[T, P, 3] -> net |F| per frame in kN."""
    a = arr.reshape(arr.shape[0], -1, 3)
    return np.linalg.norm(a.sum(axis=1), axis=-1) / 1e3


def load(eval_dir):
    d = np.load(RESULTS / eval_dir / CASE, allow_pickle=True)
    F = net_mag(d["forces"])
    G = net_mag(d["F_goal"])
    err = float(np.mean(np.abs(F - G)) / (np.mean(np.abs(G)) + 1e-9))
    return F, G, err


bc, goal, bc_err = load("eval_closed_loop")
rl, _, rl_err = load("eval_rl_fixF_multi")
t = np.arange(len(goal))

fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.plot(t, goal, color="#222", lw=2.4, ls="--", label="goal F*(t)")
ax.plot(t, bc, color="#94a3b8", lw=2.4, label=f"BC (err_ratio {bc_err:.2f})")
ax.plot(t, rl, color="#16a34a", lw=2.4, label=f"RL (err_ratio {rl_err:.2f})")

peak = int(np.argmax(goal))
ax.axvspan(peak, t[-1], color="#fde68a", alpha=0.25, zorder=0)
ax.text((peak + t[-1]) / 2, max(bc.max(), rl.max()) * 1.02,
        "descending ramp: retraction required", ha="center", fontsize=10,
        color="#92400e")

ax.set_xlabel("frame")
ax.set_ylabel("net force (kN)")
ax.set_title("Measured release behavior on a cloth ramp rollout (double_lift_cloth_3)",
             fontsize=12, fontweight="bold")
ax.legend(frameon=False, fontsize=10, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(FIG_DIR / "00_release_problem_real.png", dpi=150, bbox_inches="tight")
print("wrote", FIG_DIR / "00_release_problem_real.png")
print(f"BC err_ratio {bc_err:.3f} | RL err_ratio {rl_err:.3f}")
