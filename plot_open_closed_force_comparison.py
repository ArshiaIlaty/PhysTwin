"""plot_open_closed_force_comparison.py

Side-by-side scatter for sharing with Malak:
  Left  — Open-loop forward ML:  Predicted |F| vs Actual |F|
  Right — Closed-loop policy:    Achieved |F| vs Target |F|

Open-loop panel is generated from your trained MLP.
Closed-loop panel needs Malak's rollout CSV (see --closed_loop_csv).

Example:
  python plot_open_closed_force_comparison.py

  python plot_open_closed_force_comparison.py \\
    --closed_loop_csv scenario_sweeps/closed_loop_forces.csv \\
    --out scenario_sweeps/forward_force_results/figures/open_vs_closed_force.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_forward_predictions import force_magnitude, load_cloth_test_split
from train_models import ForceMLP, concat, metrics, set_seed


def _scatter_panel(ax, x, y, xlabel, ylabel, title, color="#2563eb"):
    ax.scatter(x, y, alpha=0.55, s=20, c=color, edgecolors="none")
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect")
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2) if len(x) > 1 else float("nan")
    mae = float(np.mean(np.abs(x - y)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nR²={r2:.3f}  MAE={mae:.2f} N")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")


def open_loop_pred_actual(
    model_dir: str,
    dataset_dir: str,
    split_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path) as f:
        meta = json.load(f)
    mlp_cfg = meta.get("mlp_config", {})

    set_seed(split_seed)
    test_cases = load_cloth_test_split(
        dataset_dir,
        split_seed=split_seed,
        clip_percentile=99.0,
        exclude_cases=("single_push_sloth",),
        exclude_categories=("toy",),
        add_type_feature=True,
    )
    X_test, y_test = concat(test_cases)

    with open(os.path.join(model_dir, "per_cat_scalers.pkl"), "rb") as f:
        sc = pickle.load(f)["cloth"]
    Xs = sc["scaler_X"].transform(X_test).astype(np.float32)
    ys = sc["scaler_y"].transform(y_test).astype(np.float32)

    model = ForceMLP(
        input_dim=meta["input_dim"],
        output_dim=meta["output_dim"],
        hidden=mlp_cfg.get("hidden", 256),
        layers=mlp_cfg.get("layers", 2),
        dropout=mlp_cfg.get("dropout", 0.0),
    )
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "mlp_cloth.pt"), map_location="cpu", weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        y_pred_s = model(torch.from_numpy(Xs)).numpy()

    y_true = sc["scaler_y"].inverse_transform(ys)
    y_pred = sc["scaler_y"].inverse_transform(y_pred_s)
    actual = force_magnitude(y_true)
    pred = force_magnitude(y_pred)
    return actual, pred


def load_closed_loop_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """CSV columns: target_force, achieved_force (or goal_force / realized_force)."""
    aliases = {
        "target": {"target_force", "goal_force", "f_goal", "target", "force_goal"},
        "achieved": {"achieved_force", "realized_force", "f_achieved", "achieved", "force_achieved"},
    }
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = {k.lower(): k for k in reader.fieldnames or []}
        tcol = next((fields[a] for a in aliases["target"] if a in fields), None)
        acol = next((fields[a] for a in aliases["achieved"] if a in fields), None)
        if not tcol or not acol:
            raise ValueError(
                f"{path} needs target + achieved force columns. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            rows.append((float(row[tcol]), float(row[acol])))
    if not rows:
        raise ValueError(f"No rows in {path}")
    target, achieved = zip(*rows)
    return np.asarray(target, dtype=np.float64), np.asarray(achieved, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="scenario_sweeps/forward_force_results/models_improved/seed_0",
    )
    parser.add_argument("--dataset_dir", default="dataset_v2")
    parser.add_argument(
        "--closed_loop_csv",
        default=None,
        help="Malak policy rollout: target_force + achieved_force columns.",
    )
    parser.add_argument(
        "--out",
        default="scenario_sweeps/forward_force_results/figures/open_vs_closed_force.png",
    )
    args = parser.parse_args()

    actual, pred = open_loop_pred_actual(args.model_dir, args.dataset_dir)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    _scatter_panel(
        axes[0], actual, pred,
        xlabel="Actual |F| (N)  [from sim]",
        ylabel="Predicted |F| (N)  [forward MLP]",
        title="Open-loop (Arshia): deformation → force",
        color="#2563eb",
    )

    if args.closed_loop_csv and os.path.isfile(args.closed_loop_csv):
        target, achieved = load_closed_loop_csv(args.closed_loop_csv)
        _scatter_panel(
            axes[1], target, achieved,
            xlabel="Target |F| (N)  [force goal]",
            ylabel="Achieved |F| (N)  [after policy + sim]",
            title="Closed-loop (Malak): policy → motion → force",
            color="#dc2626",
        )
    else:
        axes[1].text(
            0.5, 0.55,
            "Closed-loop panel\n(waiting for Malak's rollout CSV)\n\n"
            "Columns: target_force, achieved_force\n"
            "(one row per sim step)",
            ha="center", va="center", fontsize=11, transform=axes[1].transAxes,
        )
        axes[1].set_title("Closed-loop (Malak): achieved vs target force")
        axes[1].set_xlabel("Target |F| (N)")
        axes[1].set_ylabel("Achieved |F| (N)")
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)

    fig.suptitle(
        "Same diagonal = perfect prediction / perfect tracking",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
