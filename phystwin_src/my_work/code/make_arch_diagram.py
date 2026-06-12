#!/usr/bin/env python3
"""
make_arch_diagram.py — token-level diagram of the transformer policy:
8 past-frame tokens + 4 goal-preview tokens, FiLM conditioning, noise
injection on prev_action inputs, 6-D action head.

Output: figures/ensemble/13_transformer_arch.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

FIG_DIR = Path(__file__).resolve().parent.parent / "results" / "figures" / "ensemble"
FIG_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 5.2))
ax.set_xlim(0, 12.4); ax.set_ylim(0, 6.6); ax.axis("off")

PAST, GOAL, BLOCK, COND = "#bcd9f5", "#f9c6a0", "#e4dcf5", "#cdeccd"

def box(x, y, w, h, color, label, fs=8.5, lw=1.0, weight="normal"):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                   facecolor=color, edgecolor="#444", lw=lw))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, fontweight=weight)

# ---- token strip ----
tw, th, y0 = 0.78, 1.0, 3.6
for i in range(8):
    box(0.3 + i * (tw + 0.08), y0, tw, th, PAST, f"t−{7 - i}", fs=8)
for j in range(4):
    box(0.3 + 8 * (tw + 0.08) + 0.25 + j * (tw + 0.08), y0, tw, th, GOAL,
        f"F*\n+{j}", fs=7.5)

ax.text(0.3 + 4 * (tw + 0.08) - 0.05, y0 + th + 0.55,
        "8 past-frame tokens\n[state 31 | force_now 6 | prev_action 6]",
        ha="center", fontsize=9)
ax.text(0.3 + 8 * (tw + 0.08) + 0.25 + 2 * (tw + 0.08), y0 + th + 0.55,
        "4 goal-preview tokens\n(commanded F*, deployment-legitimate)",
        ha="center", fontsize=9)

# noise annotation on prev_action
ax.annotate("training only: Gaussian noise σ=0.3\non prev_action slots (scheduled sampling)",
            xy=(0.3 + 5.5 * (tw + 0.08), y0), xytext=(1.0, 1.85),
            fontsize=8.5, color="#a33",
            arrowprops=dict(arrowstyle="->", color="#a33", lw=1.0))

# ---- transformer block ----
bx, by, bw, bh = 3.4, 0.35, 4.2, 1.05
box(bx, by, bw, bh, BLOCK, "Transformer encoder (2 layers, d=128)", fs=10, lw=1.2, weight="bold")
# arrows tokens -> block
ax.annotate("", xy=(bx + bw * 0.5, by + bh), xytext=(bx + bw * 0.5, y0 - 0.06),
            arrowprops=dict(arrowstyle="->", lw=1.4))

# ---- FiLM conditioning ----
cx, cy, cw, ch = 8.6, 0.35, 3.4, 1.05
box(cx, cy, cw, ch, COND,
    "FiLM conditioning  γ, β per layer\n[5-D stiffness stats | cmd_scale]\n(11-D physics vector in the FiLM variant)",
    fs=8)
ax.annotate("scale & shift\nactivations", xy=(bx + bw, by + bh / 2),
            xytext=(cx - 0.05, cy + ch / 2), ha="right", va="center", fontsize=8.5,
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#2a7"))

# ---- output ----
ax.annotate("", xy=(1.7, by + bh / 2), xytext=(bx - 0.05, by + bh / 2),
            arrowprops=dict(arrowstyle="->", lw=1.4))
box(0.3, by, 1.4, bh, "#f5e6bc", "action\nΔgripper (6-D)", fs=9, weight="bold")

ax.set_title("Transformer policy — what the network actually sees each frame\n"
             "(history + commanded-goal preview as tokens; material physics enters via FiLM, not as a token)",
             fontsize=11)
fig.tight_layout()
fig.savefig(FIG_DIR / "13_transformer_arch.png", dpi=150, bbox_inches="tight")
print("wrote", FIG_DIR / "13_transformer_arch.png")
