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

# ---------------- (1) data overview ----------------
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, mat in zip(axes[:3], ["rope", "cloth", "sloth"]):
    for c in np.unique(cases[real & (mats == mat)]):
        m = real & (cases == c)
        order = np.argsort(fidx[m])
        F = fn[m][order].reshape(-1, 2, 3)
        mag = np.linalg.norm(F, axis=-1).sum(-1) / 1e3  # net |F| in kN
        ax.plot(mag, lw=1.2, alpha=0.85, color=MAT_COLOR[mat])
    ax.set_title(f"{mat} — recorded |F|(t)", color=MAT_COLOR[mat], fontweight="bold")
    ax.set_xlabel("frame")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("net force (kN)")

ax = axes[3]
for mat in ["rope", "cloth", "sloth"]:
    xs, ys = [], []
    for c in np.unique(cases[real & (mats == mat)]):
        m = real & (cases == c)
        xs.append(float(np.mean(np.asarray(stiff[m][0]).ravel())))
        F = fn[m].reshape(-1, 2, 3)
        ys.append(np.linalg.norm(F, axis=-1).sum(-1).mean() / 1e3)
    ax.scatter(xs, ys, s=55, color=MAT_COLOR[mat], label=mat, edgecolor="white", zorder=3)
ax.set_yscale("log")
ax.set_xlabel("log spring stiffness (descriptor dim 1)")
ax.set_ylabel("mean episode force (kN, log)")
ax.set_title("per-case map: stiffness vs force scale", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, which="both")
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("The data behind the labels — every force trace is a simulator spring force, no physical sensor",
             fontsize=12, y=1.04)
fig.tight_layout()
fig.savefig(FIG_DIR / "11_data_overview.png", dpi=150, bbox_inches="tight")
print("wrote", FIG_DIR / "11_data_overview.png")

# ---------------- (2) generalization splits schematic ----------------
fig, axes = plt.subplots(3, 1, figsize=(10, 5.6))
TRAIN, TEST = "#bcd9f5", "#f5b8b1"

def bar(ax, x0, x1, color, y=0.25, h=0.5, label=None):
    ax.add_patch(mpatches.Rectangle((x0, y), x1 - x0, h, facecolor=color,
                                    edgecolor="#555", lw=0.8))
    if label:
        ax.text((x0 + x1) / 2, y + h / 2, label, ha="center", va="center", fontsize=8)

# Level 1: random block
ax = axes[0]
b0, b1 = 0.55, 0.75  # random contiguous 20% window
bar(ax, 0.0, b0, TRAIN); bar(ax, b0, b1, TEST, label="held-out block (20%)"); bar(ax, b1, 1.0, TRAIN)
ax.annotate("random start", xy=(b0, 0.78), xytext=(b0 - 0.18, 0.93),
            fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.text(0.27, 0.5, "train", ha="center", va="center", fontsize=8)
ax.set_title("Level 1 — frame-level: \"random block\" = one contiguous time window per case, "
             "random start (never interleaved frames)", fontsize=9, loc="left")

# Level 2: trajectory-level (novel goal)
ax = axes[1]
bar(ax, 0.0, 1.0, TRAIN, label="train: recorded + synthetic goal trajectories")
ax.add_patch(mpatches.Rectangle((1.08, 0.25), 0.42, 0.5, facecolor=TEST, edgecolor="#555", lw=0.8))
t = np.linspace(0, 1, 50)
ramp = np.minimum(t, 1 - t) * 2
ax.plot(1.08 + 0.42 * t, 0.30 + 0.4 * ramp, color="#a33", lw=1.6)
ax.text(1.29, 0.86, "test: invented ramp goal 0→peak→0\n(appears in no training trajectory)",
        ha="center", fontsize=8)
ax.set_title("Level 2 — trajectory-level: closed-loop rollout on a goal F*(t) the policy has never seen",
             fontsize=9, loc="left")

# Level 3: material-level (LOSO)
ax = axes[2]
bar(ax, 0.0, 0.33, MAT_COLOR["rope"] + "55", label="rope (train)")
bar(ax, 0.36, 0.69, MAT_COLOR["cloth"] + "55", label="cloth (train)")
bar(ax, 0.72, 1.05, TEST, label="sloth (never seen)")
ax.set_title("Level 3 — material-level: leave-sloth-out — weights and scalers never touch sloth data",
             fontsize=9, loc="left")

for ax in axes:
    ax.set_xlim(-0.02, 1.55); ax.set_ylim(0, 1.05); ax.axis("off")
fig.suptitle("Three nested generalization tests (weakest → strongest)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(FIG_DIR / "12_generalization_splits.png", dpi=150)
print("wrote", FIG_DIR / "12_generalization_splits.png")
