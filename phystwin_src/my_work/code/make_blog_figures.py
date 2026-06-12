#!/usr/bin/env python3
"""
make_blog_figures.py — figures requested by report feedback.

  (1) 11_data_overview.png — "look at the data": the actual force traces the
      labels come from, per material, plus the per-case (stiffness, force-scale)
      map that motivates the material descriptor.
  (2) 12_generalization_splits.png — schematic of the three train/test splits,
      including a precise visual definition of the "random block" split.

Pure numpy + matplotlib, reads results/policy_dataset_fixK.npz.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS / "figures" / "ensemble"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MAT_COLOR = {"rope": "#1f77b4", "cloth": "#ff7f0e", "sloth": "#2ca02c"}

d = np.load(RESULTS / "policy_dataset_fixK.npz", allow_pickle=True)
cases, mats = d["case_name"], d["material"]
stiff, fn, src, fidx = d["raw_stiffness"], d["force_now"], d["source"], d["frame_idx"]
real = src == "real"

# ---------------- (1a) recorded force traces, one row ----------------
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, mat in zip(axes, ["rope", "cloth", "sloth"]):
    for c in np.unique(cases[real & (mats == mat)]):
        m = real & (cases == c)
        order = np.argsort(fidx[m])
        F = fn[m][order].reshape(-1, 2, 3)
        mag = np.linalg.norm(F, axis=-1).sum(-1) / 1e3  # net |F| in kN
        ax.plot(mag, lw=1.4, alpha=0.85, color=MAT_COLOR[mat])
    ax.set_title(mat, color=MAT_COLOR[mat], fontweight="bold", fontsize=13)
    ax.set_xlabel("frame")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("net force (kN)")
fig.tight_layout()
fig.savefig(FIG_DIR / "11a_force_traces.png", dpi=150, bbox_inches="tight")
print("wrote", FIG_DIR / "11a_force_traces.png")

# ---------------- (1b) stiffness vs force-scale map ----------------
fig, ax = plt.subplots(figsize=(6.4, 4.8))
for mat in ["rope", "cloth", "sloth"]:
    xs, ys = [], []
    for c in np.unique(cases[real & (mats == mat)]):
        m = real & (cases == c)
        xs.append(float(np.mean(np.asarray(stiff[m][0]).ravel())))
        F = fn[m].reshape(-1, 2, 3)
        ys.append(np.linalg.norm(F, axis=-1).sum(-1).mean() / 1e3)
    ax.scatter(xs, ys, s=70, color=MAT_COLOR[mat], label=mat, edgecolor="white", zorder=3)
ax.set_yscale("log")
ax.set_xlabel("log spring stiffness")
ax.set_ylabel("mean episode force (kN, log)")
ax.set_title("Per-case stiffness vs force scale", fontweight="bold", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.25, which="both")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "11b_stiffness_force_map.png", dpi=150, bbox_inches="tight")
print("wrote", FIG_DIR / "11b_stiffness_force_map.png")

# ---------------- (2) generalization splits schematic ----------------
fig, axes = plt.subplots(3, 1, figsize=(9.5, 5.2))
TRAIN, TEST = "#bcd9f5", "#f5b8b1"

def bar(ax, x0, x1, color, y=0.22, h=0.56, label=None, fs=10):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y), x1 - x0, h, boxstyle="round,pad=0.005,rounding_size=0.04",
        facecolor=color, edgecolor="#555", lw=0.8))
    if label:
        ax.text((x0 + x1) / 2, y + h / 2, label, ha="center", va="center", fontsize=fs)

# Level 1: held-out frames
ax = axes[0]
b0, b1 = 0.55, 0.75
bar(ax, 0.0, b0, TRAIN, label="train")
bar(ax, b0, b1, TEST, label="test", fs=9)
bar(ax, b1, 1.0, TRAIN, label="train")
ax.text(1.06, 0.5, "contiguous 20% window,\nrandom start", fontsize=9,
        va="center", color="#444")
ax.set_title("Level 1   Held-out frames", fontsize=11, loc="left", fontweight="bold")

# Level 2: novel goal
ax = axes[1]
bar(ax, 0.0, 1.0, TRAIN, label="train: recorded + synthetic goals")
ax.add_patch(mpatches.FancyBboxPatch((1.06, 0.22), 0.36, 0.56,
             boxstyle="round,pad=0.005,rounding_size=0.04",
             facecolor=TEST, edgecolor="#555", lw=0.8))
t = np.linspace(0, 1, 50)
ramp = np.minimum(t, 1 - t) * 2
ax.plot(1.10 + 0.28 * t, 0.30 + 0.4 * ramp, color="#a33", lw=2)
ax.text(1.47, 0.5, "test: ramp goal,\nnever in training", fontsize=9,
        va="center", color="#444")
ax.set_title("Level 2   Novel goal trajectory", fontsize=11, loc="left", fontweight="bold")

# Level 3: held-out material
ax = axes[2]
bar(ax, 0.0, 0.31, MAT_COLOR["rope"] + "55", label="rope: train")
bar(ax, 0.345, 0.655, MAT_COLOR["cloth"] + "55", label="cloth: train")
bar(ax, 0.69, 1.0, TEST, label="sloth: test")
ax.text(1.06, 0.5, "training never\nsees sloth data", fontsize=9,
        va="center", color="#444")
ax.set_title("Level 3   Held-out material", fontsize=11, loc="left", fontweight="bold")

for ax in axes:
    ax.set_xlim(-0.02, 1.85); ax.set_ylim(0, 1.1); ax.axis("off")
fig.tight_layout(h_pad=2.0)
fig.savefig(FIG_DIR / "12_generalization_splits.png", dpi=150, bbox_inches="tight")
print("wrote", FIG_DIR / "12_generalization_splits.png")
