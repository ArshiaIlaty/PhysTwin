"""make_figures_v2.py — figures for the v2 (extended-feature) + multi-seed run.

Inputs:
  dataset_v2/*.npz                     31-feature dataset
  models_v2_within_noclip/             multi-seed (no force clip) within-case run
  models_v2_within/                    multi-seed (99%-clip) within-case run
  models_v2_cross/                     multi-seed (99%-clip) cross-case run
  models_v2_cross_noclip/              multi-seed (no clip) cross-case run
  (also reads original models_cross_case + models_within_case from v1)

Outputs (overwrites previous figures):
  figures/r2_multi_seed.png            multi-seed R² bars with error bars
  figures/force_over_time_v2.png       GT vs Ridge vs MLP-unified vs MLP-per-type, 3 cases
  figures/feature_importance_v2.png    Pearson r of 31 features w/ ‖F‖
  figures/v1_v2_comparison.png         per-cat R² lift v1 → v2
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
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_cases(dataset_dir: str) -> dict:
    cases = {}
    for f in sorted(glob.glob(os.path.join(dataset_dir, "*.npz"))):
        if Path(f).name.startswith("_"):
            continue
        d = np.load(f, allow_pickle=True)
        cases[str(d["case_name"])] = {
            "X": d["X"].astype(np.float32),
            "y_net": d["y_net"].astype(np.float32),
            "material": str(d["material"]),
            "object_category": str(d["object_category"]),
            "feature_names": list(d["feature_names"]),
        }
    return cases


def load_seed_models(seed_dir: str, in_dim: int, out_dim: int):
    with open(os.path.join(seed_dir, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)
    with open(os.path.join(seed_dir, "ridge.pkl"), "rb") as f:
        ridge = pickle.load(f)
    unified = ForceMLP(in_dim, out_dim)
    unified.load_state_dict(torch.load(os.path.join(seed_dir, "mlp_unified.pt"), map_location="cpu", weights_only=True))
    unified.eval()
    per_cat = {}
    for ckpt in glob.glob(os.path.join(seed_dir, "mlp_*.pt")):
        name = Path(ckpt).stem.replace("mlp_", "")
        if name == "unified":
            continue
        m = ForceMLP(in_dim, out_dim)
        m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
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


def r2_multi_seed_chart(summaries: dict[str, dict], out_path: str):
    """summaries: {label: multi_seed_summary dict}"""
    labels = list(summaries.keys())
    cats = ["rope", "cloth", "sloth"]
    models = ["ridge", "mlp_unified", "mlp_per_cat"]
    model_short = {"ridge": "Ridge", "mlp_unified": "MLP-uni", "mlp_per_cat": "MLP-type"}

    fig, axes = plt.subplots(1, len(cats), figsize=(5 * len(cats), 5), sharey=False)
    for ax, cat in zip(axes, cats):
        x = np.arange(len(labels))
        width = 0.25
        for i, model in enumerate(models):
            means, stds = [], []
            for lbl in labels:
                s = summaries[lbl].get("summary", {})
                key = f"{model}/{cat}"
                means.append(s.get(key, {}).get("r2_mean", float("nan")))
                stds.append(s.get(key, {}).get("r2_std", 0.0))
            ax.bar(x + (i - 1) * width, means, width, yerr=stds,
                    capsize=4, label=model_short[model])
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(f"{cat} — R² (mean ± std over 5 seeds)")
        ax.set_ylabel("R²")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-2.0, 1.0)
        if cat == cats[0]:
            ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def force_over_time_figure_v2(
    cases: dict, scalers, ridge, unified, per_cat, picks: dict, out_path: str
):
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
        ax.plot(T, np.linalg.norm(y_gt, axis=1),     label="GT (PhysTwin)",  color="black", lw=2)
        ax.plot(T, np.linalg.norm(y_ridge, axis=1),  label="Ridge",           ls="--", lw=1.0)
        ax.plot(T, np.linalg.norm(y_mlp_u, axis=1),  label="MLP-unified",     ls="-",  lw=1.0)
        if y_mlp_t is not None:
            ax.plot(T, np.linalg.norm(y_mlp_t, axis=1), label=f"MLP-{cat}",   ls="-",  lw=1.5)
        ax.set_title(f"{case_name} ({cat})")
        ax.set_xlabel("frame")
        ax.set_ylabel("‖force‖")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def feature_importance_v2(cases: dict, out_path: str):
    feat_names = next(iter(cases.values()))["feature_names"]
    X_all, mag_all = [], []
    for info in cases.values():
        X_all.append(info["X"])
        mag_all.append(np.linalg.norm(info["y_net"], axis=1))
    X = np.concatenate(X_all, axis=0)
    mag = np.concatenate(mag_all, axis=0)
    corrs = []
    for i in range(X.shape[1]):
        col = X[:, i]
        if col.std() < 1e-9:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(col, mag)[0, 1]))
    order = np.argsort(np.abs(corrs))[::-1]
    new_feats = set([
        "ctrl_centroid_disp_x", "ctrl_centroid_disp_y", "ctrl_centroid_disp_z",
        "nearest_dist", "mean_contact_dist",
        "rel_motion_x", "rel_motion_y", "rel_motion_z",
        "mean_vel_mag", "max_vel_mag",
        "centroid_vel_x", "centroid_vel_y", "centroid_vel_z",
    ])
    colors = ["tab:orange" if feat_names[i] in new_feats else "tab:blue" for i in order]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh([feat_names[i] for i in order], [corrs[i] for i in order], color=colors)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Pearson r with ‖net force‖")
    ax.set_title("Feature importance: blue = v1 (18), orange = v2 additions (13)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def v1_v2_comparison(v1_metrics: dict, v2_summary: dict, out_path: str):
    cats = ["rope", "cloth", "sloth"]
    models = ["ridge", "mlp_unified", "mlp_per_cat"]
    model_short = {"ridge": "Ridge", "mlp_unified": "MLP-uni", "mlp_per_cat": "MLP-type"}

    fig, axes = plt.subplots(1, len(cats), figsize=(5 * len(cats), 4.5), sharey=False)
    for ax, cat in zip(axes, cats):
        v1_vals = [v1_metrics["results"].get(m, {}).get(cat, {}).get("r2", float("nan")) for m in models]
        v2_means = [v2_summary["summary"].get(f"{m}/{cat}", {}).get("r2_mean", float("nan")) for m in models]
        v2_stds = [v2_summary["summary"].get(f"{m}/{cat}", {}).get("r2_std", 0.0) for m in models]
        x = np.arange(len(models))
        ax.bar(x - 0.2, v1_vals, 0.4, label="v1 (18 feat, no clip, 1 seed)", color="tab:blue", alpha=0.85)
        ax.bar(x + 0.2, v2_means, 0.4, yerr=v2_stds, capsize=4,
                label="v2 (31 feat, no clip, 5 seeds)", color="tab:orange", alpha=0.85)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([model_short[m] for m in models])
        ax.set_title(f"{cat}")
        ax.set_ylabel("R²")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-2.0, 1.0)
        if cat == cats[0]:
            ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("v1 vs v2 R² (within-case split)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset_v2")
    parser.add_argument("--fig_dir", type=str, default="figures")
    parser.add_argument("--best_seed_dir", type=str, default="models_v2_within_noclip/seed_0",
                        help="Used for force-over-time figure.")
    args = parser.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    cases = load_cases(args.dataset_dir)
    in_dim = next(iter(cases.values()))["X"].shape[1]
    out_dim = next(iter(cases.values()))["y_net"].shape[1]
    print(f"in_dim={in_dim} out_dim={out_dim} cases={len(cases)}")

    # Multi-seed summaries
    summaries = {}
    for label, path in [
        ("within / no-clip", "models_v2_within_noclip/multi_seed_summary.json"),
        ("within / clip-99", "models_v2_within/multi_seed_summary.json"),
        ("cross / no-clip",  "models_v2_cross_noclip/multi_seed_summary.json"),
        ("cross / clip-99",  "models_v2_cross/multi_seed_summary.json"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                summaries[label] = json.load(f)
    r2_multi_seed_chart(summaries, os.path.join(args.fig_dir, "r2_multi_seed.png"))

    # v1 vs v2 within-case comparison
    if os.path.exists("models_within_case/metrics.json") and "within / no-clip" in summaries:
        with open("models_within_case/metrics.json") as f:
            v1m = json.load(f)
        v1_v2_comparison(v1m, summaries["within / no-clip"],
                          os.path.join(args.fig_dir, "v1_v2_comparison.png"))

    # Force-over-time and feature importance from best seed dir
    scalers, ridge, unified, per_cat = load_seed_models(args.best_seed_dir, in_dim, out_dim)
    picks = {}
    by_cat: dict[str, list[str]] = {}
    for name, info in cases.items():
        by_cat.setdefault(info["object_category"], []).append(name)
    for cat, names in by_cat.items():
        scored = sorted(names, key=lambda n: np.linalg.norm(cases[n]["y_net"], axis=1).mean())
        picks[cat] = scored[len(scored) // 2]
    print(f"picks for Fig 1: {picks}")
    force_over_time_figure_v2(cases, scalers, ridge, unified, per_cat, picks,
                                os.path.join(args.fig_dir, "force_over_time_v2.png"))
    feature_importance_v2(cases, os.path.join(args.fig_dir, "feature_importance_v2.png"))


if __name__ == "__main__":
    main()
