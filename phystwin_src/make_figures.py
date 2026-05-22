"""make_figures.py — generate demo figures from trained models.

Produces:
  figures/force_over_time__{case}.png    — GT vs Ridge vs MLP magnitude over time
  figures/r2_by_model_split.png          — grouped bar chart: model × material, cross_case vs within_case
  figures/error_distribution.png         — per-axis error box plots by material
  figures/feature_correlation.png        — correlation of summary features with ‖force‖

Reads dataset/*.npz, models_cross_case/, models_within_case/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class ForceMLP(nn.Module):
    """Must match train_models.ForceMLP exactly."""
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_all_cases(dataset_dir: str):
    cases = {}
    for f in sorted(glob.glob(os.path.join(dataset_dir, "*.npz"))):
        if Path(f).name.startswith("_"):
            continue
        d = np.load(f, allow_pickle=True)
        cases[str(d["case_name"])] = {
            "X": d["X"].astype(np.float32),
            "y_net": d["y_net"].astype(np.float32),
            "y_per_ctrl": d["y_per_ctrl"].astype(np.float32),
            "material": str(d["material"]),
            "object_category": str(d["object_category"]),
            "n_ctrl_parts": int(d["n_ctrl_parts"]),
            "feature_names": list(d["feature_names"]),
        }
    return cases


def load_models(model_dir: str, input_dim: int, output_dim: int):
    with open(os.path.join(model_dir, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)
    with open(os.path.join(model_dir, "ridge.pkl"), "rb") as f:
        ridge = pickle.load(f)
    unified = ForceMLP(input_dim, output_dim)
    unified.load_state_dict(torch.load(os.path.join(model_dir, "mlp_unified.pt"), map_location="cpu"))
    unified.eval()
    per_cat = {}
    for ckpt in glob.glob(os.path.join(model_dir, "mlp_*.pt")):
        name = Path(ckpt).stem.replace("mlp_", "")
        if name == "unified":
            continue
        m = ForceMLP(input_dim, output_dim)
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m.eval()
        per_cat[name] = m
    return scalers, ridge, unified, per_cat


def predict(model_or_ridge, X: np.ndarray, scalers) -> np.ndarray:
    Xs = scalers["scaler_X"].transform(X).astype(np.float32)
    if isinstance(model_or_ridge, nn.Module):
        with torch.no_grad():
            ys = model_or_ridge(torch.from_numpy(Xs)).numpy()
    else:
        ys = model_or_ridge.predict(Xs)
    return scalers["scaler_y"].inverse_transform(ys)


def force_over_time_figure(cases, scalers, ridge, unified, per_cat, out_path: str, picks: dict):
    n = len(picks)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (cat, case_name) in zip(axes, picks.items()):
        info = cases[case_name]
        X, y_gt = info["X"], info["y_net"]
        y_ridge = predict(ridge, X, scalers)
        y_mlp_u = predict(unified, X, scalers)
        y_mlp_t = predict(per_cat[cat], X, scalers) if cat in per_cat else None

        T = np.arange(len(y_gt))
        ax.plot(T, np.linalg.norm(y_gt, axis=1),    label="GT (PhysTwin)", color="black", lw=2)
        ax.plot(T, np.linalg.norm(y_ridge, axis=1), label="Ridge",          ls="--", lw=1.2)
        ax.plot(T, np.linalg.norm(y_mlp_u, axis=1), label="MLP-unified",    ls="-",  lw=1.2)
        if y_mlp_t is not None:
            ax.plot(T, np.linalg.norm(y_mlp_t, axis=1), label=f"MLP-{cat}", ls="-",  lw=1.4)
        ax.set_title(f"{case_name} ({cat})")
        ax.set_xlabel("frame")
        ax.set_ylabel("‖force‖")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def r2_grouped_bar(metrics_cross: dict, metrics_within: dict, out_path: str):
    cats = sorted(set(metrics_cross["results"]["ridge"]) | set(metrics_within["results"]["ridge"]))
    models = ["ridge", "mlp_unified", "mlp_per_cat"]
    labels = ["Ridge", "MLP-unified", "MLP-per-type"]
    width = 0.12
    x = np.arange(len(cats))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (m, lbl) in enumerate(zip(models, labels)):
        # cross
        r2_cross = [metrics_cross["results"].get(m, {}).get(c, {}).get("r2", float("nan")) for c in cats]
        r2_within = [metrics_within["results"].get(m, {}).get(c, {}).get("r2", float("nan")) for c in cats]
        ax.bar(x + i * 2 * width - 2.5 * width, r2_cross, width, label=f"{lbl} (cross-case)", alpha=0.85)
        ax.bar(x + i * 2 * width - 2.5 * width + width, r2_within, width, label=f"{lbl} (within-case)", alpha=0.85, hatch="//")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("R² (higher is better)")
    ax.set_title("Force prediction R² by model × material × split")
    ax.legend(fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-2.0, 1.05)  # clip extreme negatives
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def error_box_plot(cases, scalers, unified, per_cat, out_path: str):
    """Per-axis force error distribution (residuals) for the unified MLP, grouped by material."""
    axes_labels = ["Fx", "Fy", "Fz"]
    data_by_cat = {}
    for name, info in cases.items():
        X, y_gt = info["X"], info["y_net"]
        y_pred = predict(unified, X, scalers)
        resid = y_pred - y_gt
        cat = info["object_category"]
        data_by_cat.setdefault(cat, []).append(resid)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    for ax_idx, axname in enumerate(axes_labels):
        cats = sorted(data_by_cat)
        flat = [np.concatenate(data_by_cat[c], axis=0)[:, ax_idx] for c in cats]
        axs[ax_idx].boxplot(flat, labels=cats, showfliers=False)
        axs[ax_idx].set_title(f"{axname} residual (MLP-unified)")
        axs[ax_idx].axhline(0, color="black", lw=0.7)
        axs[ax_idx].grid(axis="y", alpha=0.3)
    fig.suptitle("Per-axis force residual distribution by material")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def feature_correlation(cases, out_path: str):
    feat_names = next(iter(cases.values()))["feature_names"]
    X_all, mag_all = [], []
    for info in cases.values():
        X_all.append(info["X"])
        mag_all.append(np.linalg.norm(info["y_net"], axis=1))
    X_all = np.concatenate(X_all, axis=0)
    mag_all = np.concatenate(mag_all, axis=0)
    corrs = []
    for i in range(X_all.shape[1]):
        col = X_all[:, i]
        if col.std() < 1e-9:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(col, mag_all)[0, 1]))
    order = np.argsort(np.abs(corrs))[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feat_names[i] for i in order], [corrs[i] for i in order])
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Pearson r with ‖net force‖ (all cases pooled)")
    ax.set_title("Which summary features carry force signal?")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--cross_dir", type=str, default="models_cross_case")
    parser.add_argument("--within_dir", type=str, default="models_within_case")
    parser.add_argument("--fig_dir", type=str, default="figures")
    args = parser.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    cases = load_all_cases(args.dataset_dir)
    print(f"loaded {len(cases)} cases")

    input_dim = next(iter(cases.values()))["X"].shape[1]
    output_dim = next(iter(cases.values()))["y_net"].shape[1]

    scalers_w, ridge_w, unified_w, per_cat_w = load_models(args.within_dir, input_dim, output_dim)
    with open(os.path.join(args.cross_dir,  "metrics.json")) as f: m_cross  = json.load(f)
    with open(os.path.join(args.within_dir, "metrics.json")) as f: m_within = json.load(f)

    # Pick one representative case per category for Fig 1.
    # Prefer the case with median |force|.mean() per category.
    picks = {}
    by_cat: dict[str, list[str]] = {}
    for name, info in cases.items():
        by_cat.setdefault(info["object_category"], []).append(name)
    for cat, names in by_cat.items():
        scored = sorted(names, key=lambda n: np.linalg.norm(cases[n]["y_net"], axis=1).mean())
        picks[cat] = scored[len(scored) // 2]
    print(f"picks for Fig 1: {picks}")

    force_over_time_figure(cases, scalers_w, ridge_w, unified_w, per_cat_w,
                            os.path.join(args.fig_dir, "force_over_time.png"), picks)
    r2_grouped_bar(m_cross, m_within, os.path.join(args.fig_dir, "r2_by_model_split.png"))
    error_box_plot(cases, scalers_w, unified_w, per_cat_w,
                    os.path.join(args.fig_dir, "error_distribution.png"))
    feature_correlation(cases, os.path.join(args.fig_dir, "feature_correlation.png"))


if __name__ == "__main__":
    main()
