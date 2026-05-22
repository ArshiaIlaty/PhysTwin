"""train_models.py
Loads dataset/*.npz, performs case-level train/test split, trains
- Ridge linear baseline
- Unified MLP across all materials
- Per-material MLPs (rope / cloth / sloth)
and writes metrics + saved checkpoints + the StandardScalers under models/.

Usage:
    python train_models.py [--dataset_dir dataset] [--out_dir models] [--target net|per_ctrl]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


TYPE_FEATURE_CATS = ("rope", "cloth", "sloth", "toy")


def _one_hot_type(cat: str, T: int) -> np.ndarray:
    """[T, len(TYPE_FEATURE_CATS)] one-hot tensor for category `cat`."""
    n = len(TYPE_FEATURE_CATS)
    vec = np.zeros((T, n), dtype=np.float32)
    if cat in TYPE_FEATURE_CATS:
        idx = TYPE_FEATURE_CATS.index(cat)
        vec[:, idx] = 1.0
    return vec


def load_dataset(
    dataset_dir: str,
    target_key: str,
    clip_percentile: float | None = None,
    exclude_cases: tuple[str, ...] = (),
    add_type_feature: bool = False,
    exclude_categories: tuple[str, ...] = (),
):
    """Return dict: category -> list of dicts with X, y, case_name.

    Args:
        clip_percentile: per-case ‖F‖ percentile clip (T1.3). e.g. 99.0.
        exclude_cases: case names to drop entirely (e.g. outlier "single_push_sloth").
        add_type_feature: if True, append 4 one-hot dims to X identifying the
            object category. Lets the unified MLP condition its predictions on
            material without needing a separate per-cat model.
        exclude_categories: drop entire object_category buckets. Use to drop
            "toy" (only 4 cases, 2 different objects, model can't learn).
    """
    excluded = set(exclude_cases)
    excluded_cats = set(exclude_categories)
    files = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        d = np.load(f, allow_pickle=True)
        case = str(d["case_name"])
        if case in excluded:
            print(f"  EXCLUDED {case} (per --exclude_cases)")
            continue
        cat = str(d["object_category"])
        if cat in excluded_cats:
            print(f"  EXCLUDED {case} (category {cat} in --exclude_categories)")
            continue
        y_per_ctrl = d["y_per_ctrl"]                   # [T, 2, 3]
        y_net = d["y_net"]                             # [T, 3]
        X = d["X"].astype(np.float32)                  # [T, F]
        if target_key == "net":
            y = y_net.astype(np.float32)               # [T, 3]
        elif target_key == "per_ctrl":
            y = y_per_ctrl.reshape(y_per_ctrl.shape[0], -1).astype(np.float32)  # [T, 6]
        else:
            raise ValueError(target_key)
        if clip_percentile is not None:
            y = clip_force_outliers(y, percentile=clip_percentile)
        if add_type_feature:
            X = np.concatenate([X, _one_hot_type(cat, X.shape[0])], axis=1)
        by_cat[cat].append({"X": X, "y": y, "case_name": case})
        print(f"loaded {case} ({cat}): X={X.shape} y={y.shape}")
    return by_cat


def clip_force_outliers(y: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Clamp force vectors to per-case ‖F‖ percentile. Preserves direction.

    Args:
        y: [T, D]   where D is a multiple of 3.
    Returns:
        y_clipped: same shape, with rows whose magnitude exceeds the
        percentile rescaled inwards so ‖y[t]‖ ≤ cap.
    """
    if y.shape[1] % 3 != 0:
        # Not a force vector — return unchanged.
        return y
    n_groups = y.shape[1] // 3
    out = y.copy().reshape(y.shape[0], n_groups, 3)
    # Per-control-part clipping (each gripper has its own scale)
    for g in range(n_groups):
        mag = np.linalg.norm(out[:, g, :], axis=-1)
        if mag.max() < 1e-6:
            continue
        cap = np.percentile(mag, percentile)
        if cap < 1e-6:
            continue
        scale = np.where(mag > cap, cap / np.maximum(mag, 1e-6), 1.0)
        out[:, g, :] = out[:, g, :] * scale[:, None]
    return out.reshape(y.shape)


def case_split(cases: list[dict], test_ratio: float = 0.3, rng: np.random.RandomState | None = None):
    """Cross-case split: hold out whole cases for test."""
    rng = rng or np.random.RandomState(0)
    n = len(cases)
    if n < 2:
        return cases, []
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_ratio)))
    test_idx = set(idx[:n_test].tolist())
    train = [cases[i] for i in range(n) if i not in test_idx]
    test = [cases[i] for i in range(n) if i in test_idx]
    return train, test


def within_case_split(cases: list[dict], test_ratio: float = 0.2):
    """Within-case split: take last `test_ratio` of each trajectory as test.

    Per-case temporal split tests interpolation ("given the start of this
    trajectory, can the MLP predict the rest?") rather than extrapolation to
    a new object. Forces from a single trajectory share scale, so this is
    the right split when force magnitudes vary wildly across cases of the
    same material (which they do in PhysTwin — ~30x within sloths).
    """
    train, test = [], []
    for c in cases:
        T = len(c["X"])
        if T < 5:
            train.append(c)
            continue
        n_test = max(1, int(round(T * test_ratio)))
        cut = T - n_test
        train.append({"X": c["X"][:cut], "y": c["y"][:cut], "case_name": f"{c['case_name']}__train"})
        test.append({"X": c["X"][cut:], "y": c["y"][cut:], "case_name": f"{c['case_name']}__test"})
    return train, test


def random_block_split(cases: list[dict], test_ratio: float = 0.2,
                         rng: np.random.RandomState | None = None,
                         min_T: int = 5):
    """Random contiguous block split: per trajectory, pick a random start s.t.
    a single contiguous block of `test_ratio` frames is held out, with the
    remaining frames (split across both sides) used for training.

    The clamp `start ∈ [0, T - block]` guarantees the block always fits;
    no end-of-trajectory underflow possible.

    For trajectories with T < min_T, the case is kept entirely in train
    (too small to meaningfully split).

    Reproducibility: `rng` controls which start position is picked. Pass a
    seeded RandomState per multi-seed iteration to get different splits
    that are still reproducible per seed.
    """
    rng = rng or np.random.RandomState(0)
    train, test = [], []
    for c in cases:
        T = len(c["X"])
        if T < min_T:
            train.append(c)
            continue
        block = max(1, int(round(T * test_ratio)))
        max_start = T - block            # inclusive upper bound of valid starts
        start = int(rng.randint(0, max_start + 1))
        end = start + block              # exclusive

        mask = np.ones(T, dtype=bool)
        mask[start:end] = False           # False = test, True = train
        train.append({
            "X": c["X"][mask],
            "y": c["y"][mask],
            "case_name": f"{c['case_name']}__train",
        })
        test.append({
            "X": c["X"][~mask],
            "y": c["y"][~mask],
            "case_name": f"{c['case_name']}__test[{start}:{end}]",
        })
    return train, test


def concat(cases):
    if not cases:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    X = np.concatenate([c["X"] for c in cases], axis=0)
    y = np.concatenate([c["y"] for c in cases], axis=0)
    return X, y


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


def train_mlp(
    X_train, y_train, X_val, y_val,
    input_dim: int, output_dim: int,
    epochs: int = 300, batch_size: int = 256, lr: float = 1e-3,
    device: str = "cpu", patience: int = 30,
) -> tuple[ForceMLP, list[float], list[float]]:
    device_t = torch.device(device)
    model = ForceMLP(input_dim, output_dim).to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_train_t = torch.from_numpy(X_train).to(device_t)
    y_train_t = torch.from_numpy(y_train).to(device_t)
    X_val_t = torch.from_numpy(X_val).to(device_t)
    y_val_t = torch.from_numpy(y_val).to(device_t)

    n = len(X_train_t)
    history_train, history_val = [], []
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device_t)
        total = 0.0
        for i in range(0, n, batch_size):
            sl = perm[i : i + batch_size]
            opt.zero_grad()
            pred = model(X_train_t[sl])
            loss = loss_fn(pred, y_train_t[sl])
            loss.backward()
            opt.step()
            total += loss.item() * len(sl)
        train_loss = total / max(n, 1)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item() if len(X_val_t) else float("nan")
        history_train.append(train_loss)
        history_val.append(val_loss)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history_train, history_val


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"mse": float("nan"), "r2": float("nan"), "force_mag_mae": float("nan")}
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    # Force-magnitude error: take vector norm over last 3 dims (for 3- and 6-D targets).
    def mag(arr):
        if arr.shape[1] % 3 != 0:
            return np.linalg.norm(arr, axis=1)
        n_groups = arr.shape[1] // 3
        a = arr.reshape(arr.shape[0], n_groups, 3)
        return np.linalg.norm(a, axis=-1).sum(axis=-1)
    err = np.abs(mag(y_true) - mag(y_pred))
    return {"mse": mse, "r2": r2, "force_mag_mae": float(err.mean())}


def run_pipeline(
    by_cat: dict[str, list[dict]],
    out_dir: str,
    target_key: str,
    device: str,
    split: str = "cross_case",
    scaler_mode: str = "pooled",
    split_seed: int = 0,
    extra_train_by_cat: dict[str, list[dict]] | None = None,
):
    """Train + evaluate the three model families.

    Args:
        scaler_mode: "pooled" (single global StandardScaler fit on the union
            of train-pool labels; original behavior) or "per_cat" (a separate
            scaler_y per category, used by the per-category MLP and Ridge so
            adding new categories doesn't change the per-category numerics).
            The unified MLP always uses the pooled scaler — that's the whole
            point of unified.
        split_seed: passed to `random_block_split`'s RNG so each seed picks
            its own contiguous test block per trajectory.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(0)  # for cross_case case selection
    block_rng = np.random.RandomState(split_seed)

    # Pool train/test split per category, then unify.
    train_cases_by_cat: dict[str, list[dict]] = {}
    test_cases_by_cat: dict[str, list[dict]] = {}
    for cat, cases in by_cat.items():
        if split == "cross_case":
            tr, te = case_split(cases, test_ratio=0.3, rng=rng)
        elif split == "within_case":
            tr, te = within_case_split(cases, test_ratio=0.2)
        elif split == "random_block":
            tr, te = random_block_split(cases, test_ratio=0.2, rng=block_rng)
        else:
            raise ValueError(split)
        train_cases_by_cat[cat] = tr
        test_cases_by_cat[cat] = te
        print(f"  category={cat}: train cases={[c['case_name'] for c in tr]}  test cases={[c['case_name'] for c in te]}")

    # Extra synth-only training data: appended to train pool, never seen at test.
    if extra_train_by_cat:
        for cat, extras in extra_train_by_cat.items():
            if extras:
                train_cases_by_cat.setdefault(cat, []).extend(extras)
                print(f"  category={cat}: +{len(extras)} synthetic train-only cases")

    # ---- Unified scaler fit on TRAIN ONLY ----
    train_pool = [c for cs in train_cases_by_cat.values() for c in cs]
    test_pool = [c for cs in test_cases_by_cat.values() for c in cs]
    X_train_all, y_train_all = concat(train_pool)
    X_test_all, y_test_all = concat(test_pool)
    scaler_X = StandardScaler().fit(X_train_all)
    scaler_y = StandardScaler().fit(y_train_all)
    with open(os.path.join(out_dir, "scalers.pkl"), "wb") as f:
        pickle.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, f)

    Xs_train = scaler_X.transform(X_train_all).astype(np.float32)
    ys_train = scaler_y.transform(y_train_all).astype(np.float32)
    Xs_test = scaler_X.transform(X_test_all).astype(np.float32)
    ys_test = scaler_y.transform(y_test_all).astype(np.float32)

    input_dim = Xs_train.shape[1]
    output_dim = ys_train.shape[1]

    results: dict[str, dict[str, dict]] = {}

    # ---- Model 1: Ridge baseline ----
    print("\n[Ridge] training")
    ridge = Ridge(alpha=1.0).fit(Xs_train, ys_train)
    with open(os.path.join(out_dir, "ridge.pkl"), "wb") as f:
        pickle.dump(ridge, f)
    results.setdefault("ridge", {})
    for cat, cases in test_cases_by_cat.items():
        if not cases:
            continue
        X_cat, y_cat = concat(cases)
        if len(X_cat) == 0:
            continue
        Xs = scaler_X.transform(X_cat)
        ys = scaler_y.transform(y_cat)
        y_pred_s = ridge.predict(Xs)
        m = metrics(ys, y_pred_s)
        results["ridge"][cat] = m
        print(f"  Ridge / {cat}: R2={m['r2']:.3f}  MSE={m['mse']:.4f}")

    # ---- Model 2: Unified MLP ----
    print("\n[MLP unified] training")
    mlp_unified, train_h, val_h = train_mlp(
        Xs_train, ys_train, Xs_test, ys_test,
        input_dim=input_dim, output_dim=output_dim, device=device,
    )
    torch.save(mlp_unified.state_dict(), os.path.join(out_dir, "mlp_unified.pt"))
    results.setdefault("mlp_unified", {})
    mlp_unified.eval()
    device_t = torch.device(device)
    for cat, cases in test_cases_by_cat.items():
        if not cases:
            continue
        X_cat, y_cat = concat(cases)
        if len(X_cat) == 0:
            continue
        Xs = scaler_X.transform(X_cat)
        ys = scaler_y.transform(y_cat)
        with torch.no_grad():
            y_pred_s = mlp_unified(torch.from_numpy(Xs.astype(np.float32)).to(device_t)).cpu().numpy()
        m = metrics(ys, y_pred_s)
        results["mlp_unified"][cat] = m
        print(f"  MLP-unified / {cat}: R2={m['r2']:.3f}  MSE={m['mse']:.4f}")

    # ---- Model 3: Per-category MLPs ----
    # When scaler_mode == "per_cat", each per-cat MLP has its own
    # scaler_X / scaler_y fit only on its own training data — this isolates
    # per-cat numerics from the rest of the materials, so adding/removing
    # categories doesn't shift the per-cat R² via the shared denominator.
    print(f"\n[MLP per-cat] training  (scaler_mode={scaler_mode})")
    results.setdefault("mlp_per_cat", {})
    per_cat_scalers: dict[str, dict] = {}
    for cat, train_cases in train_cases_by_cat.items():
        test_cases = test_cases_by_cat.get(cat, [])
        X_tr, y_tr = concat(train_cases)
        X_te, y_te = concat(test_cases)
        if len(X_tr) == 0:
            print(f"  {cat}: no training data, skipping")
            continue
        if scaler_mode == "per_cat":
            sc_X = StandardScaler().fit(X_tr)
            sc_y = StandardScaler().fit(y_tr)
        elif scaler_mode == "pooled":
            sc_X, sc_y = scaler_X, scaler_y
        else:
            raise ValueError(f"scaler_mode={scaler_mode}")
        per_cat_scalers[cat] = {"scaler_X": sc_X, "scaler_y": sc_y}

        Xs_tr = sc_X.transform(X_tr).astype(np.float32)
        ys_tr = sc_y.transform(y_tr).astype(np.float32)
        if len(X_te) == 0:
            print(f"  {cat}: no held-out cases — falling back to last-20%-of-trajectory val")
            n_val = max(1, int(0.2 * len(Xs_tr)))
            Xs_te, ys_te = Xs_tr[-n_val:], ys_tr[-n_val:]
            Xs_tr, ys_tr = Xs_tr[:-n_val], ys_tr[:-n_val]
        else:
            Xs_te = sc_X.transform(X_te).astype(np.float32)
            ys_te = sc_y.transform(y_te).astype(np.float32)
        model, _, _ = train_mlp(
            Xs_tr, ys_tr, Xs_te, ys_te,
            input_dim=input_dim, output_dim=output_dim, device=device,
        )
        torch.save(model.state_dict(), os.path.join(out_dir, f"mlp_{cat}.pt"))
        model.eval()
        with torch.no_grad():
            y_pred_s = model(torch.from_numpy(Xs_te).to(device_t)).cpu().numpy()
        m = metrics(ys_te, y_pred_s)
        results["mlp_per_cat"][cat] = m
        print(f"  MLP-{cat}: R2={m['r2']:.3f}  MSE={m['mse']:.4f}")
    if scaler_mode == "per_cat":
        with open(os.path.join(out_dir, "per_cat_scalers.pkl"), "wb") as f:
            pickle.dump(per_cat_scalers, f)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "target_key": target_key,
                "input_dim": int(input_dim),
                "output_dim": int(output_dim),
                "results": results,
                "train_test_split": {
                    cat: {
                        "train_cases": [c["case_name"] for c in train_cases_by_cat[cat]],
                        "test_cases": [c["case_name"] for c in test_cases_by_cat[cat]],
                    }
                    for cat in by_cat
                },
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {out_dir}/metrics.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--target", type=str, choices=["net", "per_ctrl"], default="per_ctrl")
    parser.add_argument("--split", type=str,
                        choices=["cross_case", "within_case", "random_block"],
                        default="cross_case",
                        help="cross_case: hold out whole cases per category. "
                             "within_case: hold out the last 20%% of each trajectory. "
                             "random_block: hold out a random contiguous 20%% block per trajectory (start position derived from --seed).")
    parser.add_argument("--scaler_mode", type=str, choices=["pooled", "per_cat"], default="pooled",
                        help="pooled (default) uses a single global StandardScaler for everything. "
                             "per_cat fits a separate scaler per material for the per-cat MLP only "
                             "(unified MLP stays pooled). Use per_cat when you're changing the category set.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Single-seed mode (kept for back-compat).")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-seed eval (T1.5). e.g. '0,1,2,3,4'. "
                             "If set, overrides --seed and reports mean ± std per (model, category).")
    parser.add_argument("--clip_percentile", type=float, default=None,
                        help="Per-case force-magnitude clip (T1.3). E.g. 99.0 clips top 1%% spikes.")
    parser.add_argument("--exclude_cases", type=str, default="",
                        help="Comma-separated case names to drop entirely (Path A: drop single_push_sloth).")
    parser.add_argument("--exclude_categories", type=str, default="",
                        help="Comma-separated category names to drop (e.g. 'toy' to remove the zebra/dinosaur bucket "
                             "that doesn't generalize).")
    parser.add_argument("--add_type_feature", action="store_true",
                        help="Append 4-dim one-hot (rope/cloth/sloth/toy) to X. "
                             "Lets the unified MLP condition on material — competes against per-cat MLPs.")
    parser.add_argument("--extra_train_dir", type=str, default=None,
                        help="Directory of additional .npz files (same schema as --dataset_dir) "
                             "to add to the TRAINING pool only. Test split never sees them. "
                             "Use this for synthetic trajectories.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    exclude = tuple(c.strip() for c in args.exclude_cases.split(",") if c.strip())
    exclude_cats = tuple(c.strip() for c in args.exclude_categories.split(",") if c.strip())

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        run_multi_seed(seeds, args, device, exclude_cases=exclude, exclude_categories=exclude_cats)
    else:
        set_seed(args.seed)
        by_cat = load_dataset(args.dataset_dir, target_key=args.target,
                              clip_percentile=args.clip_percentile, exclude_cases=exclude,
                              add_type_feature=args.add_type_feature,
                              exclude_categories=exclude_cats)
        extra = (load_dataset(args.extra_train_dir, target_key=args.target,
                              clip_percentile=args.clip_percentile, exclude_cases=(),
                              add_type_feature=args.add_type_feature,
                              exclude_categories=exclude_cats)
                 if args.extra_train_dir else None)
        run_pipeline(by_cat, args.out_dir, args.target, device, split=args.split,
                     scaler_mode=args.scaler_mode, split_seed=args.seed,
                     extra_train_by_cat=extra)


def run_multi_seed(seeds: list[int], args, device: str,
                    exclude_cases: tuple[str, ...] = (),
                    exclude_categories: tuple[str, ...] = ()) -> None:
    """Run the pipeline once per seed; aggregate (model, category) R²/MSE.

    Each seed writes its artifacts to `<out_dir>/seed_<s>/`. A combined
    summary lands at `<out_dir>/multi_seed_summary.json`.
    """
    os.makedirs(args.out_dir, exist_ok=True)
    all_rows: list[dict] = []
    for s in seeds:
        print(f"\n========== seed {s} ==========")
        set_seed(s)
        seed_dir = os.path.join(args.out_dir, f"seed_{s}")
        by_cat = load_dataset(args.dataset_dir, target_key=args.target,
                              clip_percentile=args.clip_percentile,
                              exclude_cases=exclude_cases,
                              add_type_feature=getattr(args, "add_type_feature", False),
                              exclude_categories=exclude_categories)
        extra = None
        if getattr(args, "extra_train_dir", None):
            extra = load_dataset(args.extra_train_dir, target_key=args.target,
                                 clip_percentile=args.clip_percentile,
                                 exclude_cases=(),
                                 add_type_feature=getattr(args, "add_type_feature", False),
                                 exclude_categories=exclude_categories)
        run_pipeline(by_cat, seed_dir, args.target, device, split=args.split,
                     scaler_mode=args.scaler_mode, split_seed=s,
                     extra_train_by_cat=extra)
        with open(os.path.join(seed_dir, "metrics.json")) as f:
            m = json.load(f)
        for model_name, by_cat_metrics in m["results"].items():
            for cat, vals in by_cat_metrics.items():
                all_rows.append({"seed": s, "model": model_name, "cat": cat,
                                 "r2": vals.get("r2"), "mse": vals.get("mse"),
                                 "force_mag_mae": vals.get("force_mag_mae")})
    # Aggregate
    agg: dict[tuple[str, str], dict] = {}
    for row in all_rows:
        key = (row["model"], row["cat"])
        agg.setdefault(key, {"r2": [], "mse": [], "force_mag_mae": []})
        for k in ("r2", "mse", "force_mag_mae"):
            if row[k] is not None and not (isinstance(row[k], float) and np.isnan(row[k])):
                agg[key][k].append(row[k])
    summary = {}
    print("\n========== multi-seed summary (mean ± std over seeds) ==========")
    print(f"{'model':15s} {'cat':8s} {'R² mean':>9s} {'R² std':>9s} {'MSE mean':>10s}")
    for (model, cat), v in sorted(agg.items()):
        r2_arr = np.array(v["r2"]) if v["r2"] else np.array([float("nan")])
        mse_arr = np.array(v["mse"]) if v["mse"] else np.array([float("nan")])
        summary[f"{model}/{cat}"] = {
            "seeds": seeds,
            "r2_mean": float(np.nanmean(r2_arr)), "r2_std": float(np.nanstd(r2_arr)),
            "mse_mean": float(np.nanmean(mse_arr)), "mse_std": float(np.nanstd(mse_arr)),
        }
        print(f"{model:15s} {cat:8s} {np.nanmean(r2_arr):>9.3f} {np.nanstd(r2_arr):>9.3f} {np.nanmean(mse_arr):>10.4g}")
    with open(os.path.join(args.out_dir, "multi_seed_summary.json"), "w") as f:
        json.dump({"args": vars(args), "summary": summary, "rows": all_rows}, f, indent=2)


if __name__ == "__main__":
    main()
