"""plot_forward_predictions.py — pred vs actual scatter for forward force MLP."""
from __future__ import annotations

import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from train_models import (
    ForceMLP,
    concat,
    load_dataset,
    metrics,
    random_block_split,
    set_seed,
)


def force_magnitude(y: np.ndarray) -> np.ndarray:
    """Net wrench magnitude for y shaped [N, 3]."""
    return np.linalg.norm(y.reshape(y.shape[0], -1, 3), axis=-1).sum(axis=-1)


def load_category_test_split(
    category: str,
    dataset_dir: str,
    split_seed: int,
    clip_percentile: float | None,
    exclude_cases: tuple[str, ...],
    exclude_categories: tuple[str, ...],
    add_type_feature: bool,
) -> list[dict]:
    """Return held-out test blocks for `category`, matching train_models split.

    Uses one shared RNG across categories (same order as load_dataset) so
    random_block splits align with metrics.json train_test_split.
    """
    by_cat = load_dataset(
        dataset_dir,
        target_key="net",
        clip_percentile=clip_percentile,
        exclude_cases=exclude_cases,
        add_type_feature=add_type_feature,
        exclude_categories=exclude_categories,
    )
    if category not in by_cat or not by_cat[category]:
        raise SystemExit(f"No {category!r} cases found in dataset")

    block_rng = np.random.RandomState(split_seed)
    for cat, cases in by_cat.items():
        _, test_cases = random_block_split(cases, test_ratio=0.2, rng=block_rng)
        if cat == category:
            return test_cases
    raise SystemExit(f"Category {category!r} not found in dataset")


def load_cloth_test_split(
    dataset_dir: str,
    split_seed: int,
    clip_percentile: float | None,
    exclude_cases: tuple[str, ...],
    exclude_categories: tuple[str, ...],
    add_type_feature: bool,
) -> list[dict]:
    return load_category_test_split(
        "cloth",
        dataset_dir,
        split_seed,
        clip_percentile,
        exclude_cases,
        exclude_categories,
        add_type_feature,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="scenario_sweeps/forward_force_results/models_improved/seed_0",
    )
    parser.add_argument("--dataset_dir", type=str, default="dataset_v2")
    parser.add_argument("--category", type=str, default="cloth")
    parser.add_argument(
        "--out",
        type=str,
        default="scenario_sweeps/forward_force_results/figures/pred_vs_actual_cloth.png",
    )
    parser.add_argument("--clip_percentile", type=float, default=99.0)
    parser.add_argument("--exclude_cases", type=str, default="single_push_sloth")
    parser.add_argument("--exclude_categories", type=str, default="toy")
    parser.add_argument("--add_type_feature", action="store_true", default=True)
    args = parser.parse_args()

    metrics_path = os.path.join(args.model_dir, "metrics.json")
    with open(metrics_path) as f:
        meta = json.load(f)
    mlp_cfg = meta.get("mlp_config", {})
    split_seed = meta.get("split_seed")
    if split_seed is None:
        # infer from path .../seed_N/
        base = os.path.basename(os.path.normpath(args.model_dir))
        if base.startswith("seed_"):
            split_seed = int(base.split("_", 1)[1])
        else:
            split_seed = 0

    exclude = tuple(c.strip() for c in args.exclude_cases.split(",") if c.strip())
    exclude_cats = tuple(c.strip() for c in args.exclude_categories.split(",") if c.strip())

    set_seed(split_seed)
    test_cases = load_category_test_split(
        args.category,
        args.dataset_dir,
        split_seed=split_seed,
        clip_percentile=args.clip_percentile,
        exclude_cases=exclude,
        exclude_categories=exclude_cats,
        add_type_feature=args.add_type_feature,
    )
    X_test, y_test = concat(test_cases)
    if len(X_test) == 0:
        raise SystemExit(f"Empty {args.category} test split")

    scalers_path = os.path.join(args.model_dir, "per_cat_scalers.pkl")
    with open(scalers_path, "rb") as f:
        per_cat_scalers = pickle.load(f)
    sc_X = per_cat_scalers[args.category]["scaler_X"]
    sc_y = per_cat_scalers[args.category]["scaler_y"]

    Xs = sc_X.transform(X_test).astype(np.float32)
    ys = sc_y.transform(y_test).astype(np.float32)

    ckpt = os.path.join(args.model_dir, f"mlp_{args.category}.pt")
    model = ForceMLP(
        input_dim=meta["input_dim"],
        output_dim=meta["output_dim"],
        hidden=mlp_cfg.get("hidden", 256),
        layers=mlp_cfg.get("layers", 2),
        dropout=mlp_cfg.get("dropout", 0.0),
    )
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        y_pred_s = model(torch.from_numpy(Xs)).numpy()

    m = metrics(ys, y_pred_s)
    y_true_phys = sc_y.inverse_transform(ys)
    y_pred_phys = sc_y.inverse_transform(y_pred_s)
    mag_true = force_magnitude(y_true_phys)
    mag_pred = force_magnitude(y_pred_phys)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    ax.scatter(mag_true, mag_pred, alpha=0.55, s=18, edgecolors="none")
    lo = min(mag_true.min(), mag_pred.min())
    hi = max(mag_true.max(), mag_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel("Actual |F| (N)")
    ax.set_ylabel("Predicted |F| (N)")
    ax.set_title(f"{args.category} — force magnitude (R²={m['r2']:.3f})")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    t = np.arange(len(mag_true))
    ax.plot(t, mag_true, label="actual", lw=1.5)
    ax.plot(t, mag_pred, label="predicted", lw=1.5, alpha=0.85)
    ax.set_xlabel("Test frame (concatenated)")
    ax.set_ylabel("|F| (N)")
    ax.set_title("Held-out block — time series")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Forward MLP ({args.category}) — random_block seed={split_seed}, "
        f"MAE={m['force_mag_mae']:.3f} N",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    print(f"Saved {args.out}")
    print(f"  test blocks ({args.category}):")
    for c in test_cases:
        print(f"    {c['case_name']}  ({len(c['X'])} frames)")
    print(f"  test frames: {len(mag_true)}")
    print(f"  R²={m['r2']:.6f}  MSE={m['mse']:.6f}  |F| MAE={m['force_mag_mae']:.4f} N")
    metrics_cat = meta["results"]["mlp_per_cat"].get(args.category, {})
    if metrics_cat:
        print(f"  metrics.json R²={metrics_cat.get('r2', float('nan')):.6f}")


if __name__ == "__main__":
    main()
