#!/usr/bin/env python3
"""make_material_encoder_figure.py — visualize the Fix H material encoder.

Two panels that together answer the professor's "visualize your data" and
"give intuition for the descriptor" comments:

  LEFT  — the raw stiffness *data* the encoder ingests: per-case
          log(spring_Y) distribution (mean ± [q10,q90]) the encoder sees,
          one row per case, grouped/coloured by material.
  RIGHT — the *learned* 2-D latent the encoder produces (PCA of the
          `latent_dim`-D embedding), one point per case, coloured by material.
          If the encoder is doing its job, materials separate here even though
          the input is stiffness-only (no material label, no force leakage).

Usage:
  python my_work/code/make_material_encoder_figure.py \
      --policy_dir my_work/results/models_policy_fixH/seed_1 \
      --out my_work/presentation_results/figures/06_material_encoder.png
"""

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from features import get_stiffness_stats, _STIFFNESS_QUANTILES  # noqa: E402
from train_policy import load_policy_model  # noqa: E402

RESULTS = SCRIPT_DIR.parent / "results"
MAT_COLORS = {"rope": "#1f77b4", "cloth": "#d62728", "sloth": "#2ca02c", "toy": "#7f7f7f"}


def case_category(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return str(d["object_category"]) if "object_category" in d.files else "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy_dir", type=Path,
                    default=RESULTS / "models_policy_fixH" / "seed_1")
    ap.add_argument("--dataset_v2_dir", type=Path, default=RESULTS / "dataset_v2")
    ap.add_argument("--out", type=Path,
                    default=SCRIPT_DIR.parent / "presentation_results" /
                    "figures" / "06_material_encoder.png")
    ap.add_argument("--exclude_materials", nargs="*", default=["toy"])
    args = ap.parse_args()

    model = load_policy_model(args.policy_dir)
    with open(args.policy_dir / "feat_scaler.pkl", "rb") as f:
        feat_scaler = pickle.load(f)
    base_dim = int(model.base_dim)
    raw_dim = int(model.raw_dim)
    f_mean = feat_scaler["mean"][base_dim:base_dim + raw_dim].astype(np.float32)
    f_std = feat_scaler["std"][base_dim:base_dim + raw_dim].astype(np.float32)

    # Collect per-case stiffness stats + latent embeddings.
    cases, mats, stats, latents = [], [], [], []
    for p in sorted(glob.glob(str(args.dataset_v2_dir / "*.npz"))):
        case = os.path.basename(p)[:-4]
        cat = case_category(p)
        if cat in args.exclude_materials:
            continue
        try:
            s = get_stiffness_stats(case)  # stiffness-only, same as training
        except Exception as e:
            print(f"skip {case}: {e}")
            continue
        x = (s - f_mean) / f_std
        with torch.no_grad():
            z = model.encoder(torch.from_numpy(x).float().unsqueeze(0)).numpy()[0]
        cases.append(case); mats.append(cat); stats.append(s); latents.append(z)

    stats = np.array(stats)          # [C, raw_dim] = [mean, std, q10, q50, q90]
    latents = np.array(latents)      # [C, latent_dim]
    print(f"{len(cases)} cases, raw_dim={raw_dim}, latent_dim={latents.shape[1]}")

    # PCA of the latent (centre + SVD → 2 comps).
    Z = latents - latents.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    pcs = Z @ Vt[:2].T               # [C, 2]
    var = (S[:2] ** 2) / (S ** 2).sum()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    # LEFT: raw stiffness data per case (mean with q10–q90 whiskers).
    order = np.argsort([list(MAT_COLORS).index(m) for m in mats])
    qlo = _STIFFNESS_QUANTILES.index(0.1) + 2  # stats index of q10
    qhi = _STIFFNESS_QUANTILES.index(0.9) + 2
    for row, i in enumerate(order):
        m = mats[i]
        mean = stats[i, 0]
        lo, hi = stats[i, qlo], stats[i, qhi]
        axL.plot([lo, hi], [row, row], color=MAT_COLORS[m], lw=2, alpha=0.6, zorder=1)
        axL.scatter([mean], [row], color=MAT_COLORS[m], s=40, zorder=2)
    axL.set_yticks(range(len(order)))
    axL.set_yticklabels([cases[i] for i in order], fontsize=7)
    axL.set_xlabel("log(spring_Y)  —  mean with q10–q90 spread")
    axL.set_title("(a) Encoder INPUT: per-case stiffness data\n"
                  "(stiffness only — no material label, no force scale)", fontsize=10)
    axL.grid(axis="x", alpha=0.3)

    # RIGHT: learned latent PCA, coloured by material.
    seen = set()
    for i in range(len(cases)):
        m = mats[i]
        axR.scatter(pcs[i, 0], pcs[i, 1], color=MAT_COLORS[m], s=70,
                    edgecolor="k", linewidth=0.5,
                    label=m if m not in seen else None)
        seen.add(m)
        axR.annotate(cases[i], (pcs[i, 0], pcs[i, 1]), fontsize=6,
                     xytext=(4, 2), textcoords="offset points")
    axR.set_xlabel(f"latent PC1 ({var[0] * 100:.0f}% var)")
    axR.set_ylabel(f"latent PC2 ({var[1] * 100:.0f}% var)")
    axR.set_title(f"(b) Encoder OUTPUT: learned {latents.shape[1]}-D material\n"
                  "latent (PCA) — materials separate from stiffness alone",
                  fontsize=10)
    axR.legend(title="material", loc="best")
    axR.grid(alpha=0.3)

    fig.suptitle("Fix H material encoder: stiffness distribution → learned latent",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
