"""evaluate_models.py
Loads the trained models from models/ and the per-case datasets from dataset/,
then generates the demo figures:

  1. force-over-time per case (GT vs Ridge vs MLP unified)
  2. R^2 bar chart by model x category
  3. per-component error box plots
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ForceMLP(nn.Module):
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


def load_metrics(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "metrics.json")) as f:
        return json.load(f)


def load_models(out_dir: str, input_dim: int, output_dim: int, device: str):
    with open(os.path.join(out_dir, "ridge.pkl"), "rb") as f:
        ridge = pickle.load(f)
    with open(os.path.join(out_dir, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)
    device_t = torch.device(device)
    mlp_unified = ForceMLP(input_dim, output_dim).to(device_t)
    mlp_unified.load_state_dict(torch.load(os.path.join(out_dir, "mlp_unified.pt"), map_location=device_t))
    mlp_unified.eval()
    per_cat = {}
    for f in glob.glob(os.path.join(out_dir, "mlp_*.pt")):
        name = os.path.basename(f).replace("mlp_", "").replace(".pt", "")
        if name == "unified":
            continue
        m = ForceMLP(input_dim, output_dim).to(device_t)
        m.load_state_dict(torch.load(f, map_location=device_t))
        m.eval()
        per_cat[name] = m
    return {"ridge": ridge, "scalers": scalers, "mlp_unified": mlp_unified, "mlp_per_cat": per_cat}


def force_magnitude(y: np.ndarray) -> np.ndarray:
    if y.shape[1] == 3:
        return np.linalg.norm(y, axis=1)
    n_groups = y.shape[1] // 3
    return np.linalg.norm(y.reshape(y.shape[0], n_groups, 3), axis=-1).sum(axis=-1)


def figure_force_over_time(dataset_dir: str, models, metrics_meta, fig_dir: str, target_key: str):
    os.makedirs(fig_dir, exist_ok=True)
    scaler_X = models["scalers"]["scaler_X"]
    scaler_y = models["scalers"]["scaler_y"]
    device_t = next(models["mlp_unified"].parameters()).device

    rep_case_by_cat: dict[str, str] = {}
    for cat, cases in metrics_meta["train_test_split"].items():
        if cases["test_cases"]:
            rep_case_by_cat[cat] = cases["test_cases"][0]
        elif cases["train_cases"]:
            rep_case_by_cat[cat] = cases["train_cases"][0]

    fig, axes = plt.subplots(len(rep_case_by_cat), 1, figsize=(8, 3 * len(rep_case_by_cat)), squeeze=False)
    for ax, (cat, case) in zip(axes[:, 0], rep_case_by_cat.items()):
        path = os.path.join(dataset_dir, f"{case}.npz")
        if not os.path.exists(path):
            ax.set_title(f"{cat}: {case} (missing)")
            continue
        d = np.load(path, allow_pickle=True)
        X = d["X"].astype(np.float32)
        if target_key == "net":
            y = d["y_net"].astype(np.float32)
        else:
            yp = d["y_per_ctrl"]
            y = yp.reshape(yp.shape[0], -1).astype(np.float32)
        Xs = scaler_X.transform(X).astype(np.float32)
        ys = scaler_y.transform(y)

        # Predict (return to unscaled space)
        y_ridge_s = models["ridge"].predict(Xs)
        with torch.no_grad():
            y_mlp_s = models["mlp_unified"](torch.from_numpy(Xs).to(device_t)).cpu().numpy()
        y_ridge = scaler_y.inverse_transform(y_ridge_s)
        y_mlp = scaler_y.inverse_transform(y_mlp_s)

        t = np.arange(y.shape[0])
        ax.plot(t, force_magnitude(y), label="GT", color="black", linewidth=1.5)
        ax.plot(t, force_magnitude(y_ridge), label="Ridge", color="tab:orange", linestyle="--")
        ax.plot(t, force_magnitude(y_mlp), label="MLP (unified)", color="tab:blue")
        ax.set_title(f"{cat}: {case}")
        ax.set_xlabel("frame")
        ax.set_ylabel("|F|")
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = os.path.join(fig_dir, "force_over_time.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


def figure_r2_bars(metrics_meta: dict, fig_dir: str):
    os.makedirs(fig_dir, exist_ok=True)
    results = metrics_meta["results"]
    cats = sorted({c for m in results.values() for c in m.keys()})
    models_order = ["ridge", "mlp_unified", "mlp_per_cat"]
    width = 0.25
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, model in enumerate(models_order):
        vals = [results.get(model, {}).get(c, {}).get("r2", float("nan")) for c in cats]
        ax.bar(x + (i - 1) * width, vals, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel(r"$R^2$")
    ax.set_title("Test-set R^2 by model x material")
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(fig_dir, "r2_by_model.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


def figure_error_distribution(dataset_dir: str, models, metrics_meta, fig_dir: str, target_key: str):
    os.makedirs(fig_dir, exist_ok=True)
    scaler_X = models["scalers"]["scaler_X"]
    scaler_y = models["scalers"]["scaler_y"]
    device_t = next(models["mlp_unified"].parameters()).device

    errs_by_cat: dict[str, list[np.ndarray]] = defaultdict(list)
    for cat, cases in metrics_meta["train_test_split"].items():
        for case in cases["test_cases"]:
            path = os.path.join(dataset_dir, f"{case}.npz")
            if not os.path.exists(path):
                continue
            d = np.load(path, allow_pickle=True)
            X = d["X"].astype(np.float32)
            if target_key == "net":
                y = d["y_net"].astype(np.float32)
            else:
                yp = d["y_per_ctrl"]
                y = yp.reshape(yp.shape[0], -1).astype(np.float32)
            Xs = scaler_X.transform(X).astype(np.float32)
            with torch.no_grad():
                y_pred_s = models["mlp_unified"](torch.from_numpy(Xs).to(device_t)).cpu().numpy()
            y_pred = scaler_y.inverse_transform(y_pred_s)
            errs_by_cat[cat].append(np.abs(y - y_pred))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    for ax, axis_idx, axis_name in zip(axes[0], [0, 1, 2], ["Fx", "Fy", "Fz"]):
        boxes, labels = [], []
        for cat, arrs in errs_by_cat.items():
            if not arrs:
                continue
            comb = np.concatenate(arrs, axis=0)
            n_groups = comb.shape[1] // 3
            comp = comb.reshape(comb.shape[0], n_groups, 3)[:, :, axis_idx].reshape(-1)
            boxes.append(comp)
            labels.append(cat)
        if not boxes:
            continue
        ax.boxplot(boxes, labels=labels, showfliers=False)
        ax.set_title(f"|err {axis_name}| on test set (MLP unified)")
    fig.tight_layout()
    out = os.path.join(fig_dir, "error_distribution.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--fig_dir", type=str, default="figures")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    metrics_meta = load_metrics(args.models_dir)
    target_key = metrics_meta["target_key"]
    models = load_models(args.models_dir, metrics_meta["input_dim"], metrics_meta["output_dim"], device)
    figure_force_over_time(args.dataset_dir, models, metrics_meta, args.fig_dir, target_key)
    figure_r2_bars(metrics_meta, args.fig_dir)
    figure_error_distribution(args.dataset_dir, models, metrics_meta, args.fig_dir, target_key)


if __name__ == "__main__":
    main()
