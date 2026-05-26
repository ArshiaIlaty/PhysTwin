#!/usr/bin/env python3
"""
train_policy.py

Train a frame-independent MLP policy that maps
  (state[31], force_now[2,3], force_goal[2,3])  ->  Δctrl[2,3]
using the policy dataset built by build_policy_dataset.py.

Modes (CLI flags):
  --check_only      G1: load, split, scale, no training
  --overfit_tiny    G2: overfit 100 rows × N epochs
  --inspect_only    G4: load a saved model and print sample predictions
  (default)         G3 single seed (--seed) or G5 multi (--seeds 0,1,2,3,4)

Plan:    my_work/docs/closed_loop_control/step2_plan.md
Inputs:  my_work/results/policy_dataset.npz
Outputs: my_work/results/models_policy/seed_*/{policy.pt, *.pkl, *.json}
         my_work/results/models_policy/summary.json
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("train_policy")

SCRIPT_DIR = Path(__file__).resolve().parent
MY_WORK = SCRIPT_DIR.parent
RESULTS = MY_WORK / "results"
DEFAULT_DATA = RESULTS / "policy_dataset.npz"
DEFAULT_OUT = RESULTS / "models_policy"


# ----------------------------- data ---------------------------------------

def load_dataset(npz_path: Path, exclude_materials):
    d = np.load(npz_path, allow_pickle=True)
    R = d["state"].shape[0]
    # Cast string arrays from <U* to plain Python strings for safe equality.
    material = np.array([str(m) for m in d["material"]])
    case_name = np.array([str(c) for c in d["case_name"]])
    motion_type = np.array([str(m) for m in d["motion_type"]])
    source = np.array([str(s) for s in d["source"]])

    keep = np.ones(R, dtype=bool)
    for m in exclude_materials:
        keep &= material != m
    n_dropped = int((~keep).sum())
    logger.info("dropped %d rows for excluded materials %s",
                n_dropped, list(exclude_materials))

    out = {
        "state":        d["state"][keep].astype(np.float32),
        "force_now":    d["force_now"][keep].astype(np.float32),
        "force_goal":   d["force_goal"][keep].astype(np.float32),
        "action":       d["action"][keep].astype(np.float32),
        "action_mask":  d["action_mask"][keep].astype(np.float32),
        "n_ctrl_parts": d["n_ctrl_parts"][keep].astype(np.int8),
        "material":     material[keep],
        "case_name":    case_name[keep],
        "motion_type":  motion_type[keep],
        "source":       source[keep],
    }
    feature_parts = [
        out["state"],
        out["force_now"].reshape(-1, 6),
        out["force_goal"].reshape(-1, 6),
    ]
    # Fix E/F: optional material descriptor (1 or 2 dim per row)
    if "material_descriptor" in d.files:
        md = d["material_descriptor"][keep].astype(np.float32)
        if md.ndim == 1:
            md = md.reshape(-1, 1)
        out["material_descriptor"] = md
        feature_parts.append(md)
    out["features"] = np.concatenate(feature_parts, axis=1).astype(np.float32)
    # Allow 43 (no descriptor), 44 (Fix E), 45 (Fix F)
    assert out["features"].shape[1] in (43, 44, 45), out["features"].shape
    return out


def random_block_split(case_names: np.ndarray, val_frac: float, seed: int):
    """For each unique case, pick a contiguous block of val_frac rows for val."""
    rng = np.random.RandomState(seed)
    N = len(case_names)
    val_mask = np.zeros(N, dtype=bool)
    for case in np.unique(case_names):
        idx = np.where(case_names == case)[0]
        n = len(idx)
        n_val = max(1, int(round(n * val_frac)))
        if n_val >= n:
            n_val = max(1, n - 1)
        start_max = n - n_val
        start = rng.randint(0, start_max + 1) if start_max > 0 else 0
        val_mask[idx[start:start + n_val]] = True
    return val_mask


def assert_split_sanity(data, val_mask, val_frac):
    N = len(val_mask)
    n_val = int(val_mask.sum())
    n_train = N - n_val
    train_frac = n_train / N
    val_frac_actual = n_val / N
    logger.info("split: train=%d (%.1f%%), val=%d (%.1f%%)",
                n_train, 100 * train_frac, n_val, 100 * val_frac_actual)
    assert abs(val_frac_actual - val_frac) < 0.02, (
        f"val fraction off: {val_frac_actual:.3f} vs target {val_frac:.3f}")
    train_mats = set(data["material"][~val_mask])
    val_mats = set(data["material"][val_mask])
    missing = train_mats ^ val_mats
    assert not missing, f"materials in only one of train/val: {missing}"


# ----------------------------- scalers ------------------------------------

def fit_feature_scaler(features: np.ndarray):
    return {
        "mean": features.mean(axis=0).astype(np.float32),
        "std": (features.std(axis=0) + 1e-6).astype(np.float32),
    }


def fit_target_scalers(actions, materials, mask):
    """Per-material per-group scaler. actions [N,2,3], mask [N,2].

    For each material, fit a scaler with 6 params (3 per group). Group axes
    are fit only on rows where that group is active (mask[g] == 1), so
    single-ctrl rows don't pollute group-2 statistics with zeros.
    """
    scalers = {}
    for m in sorted(np.unique(materials)):
        m_rows = materials == m
        actions_m = actions[m_rows]
        mask_m = mask[m_rows]
        mean = np.zeros(6, dtype=np.float32)
        std = np.ones(6, dtype=np.float32)
        n_active = [0, 0]
        for g in range(2):
            active = mask_m[:, g] > 0.5
            n_active[g] = int(active.sum())
            if n_active[g] == 0:
                continue
            vals = actions_m[active, g, :]
            mean[g * 3:(g + 1) * 3] = vals.mean(axis=0)
            std[g * 3:(g + 1) * 3] = vals.std(axis=0) + 1e-6
        scalers[m] = {"mean": mean, "std": std, "n_active": n_active}
    return scalers


def scale_features(features, scaler):
    return (features - scaler["mean"]) / scaler["std"]


def scale_targets(actions, materials, scalers):
    flat = actions.reshape(-1, 6)
    scaled = np.zeros_like(flat)
    for m, sc in scalers.items():
        m_rows = materials == m
        scaled[m_rows] = (flat[m_rows] - sc["mean"]) / sc["std"]
    return scaled  # [N, 6]


def unscale_predictions(pred_flat, materials, scalers):
    unscaled = np.zeros_like(pred_flat)
    for m, sc in scalers.items():
        m_rows = materials == m
        unscaled[m_rows] = pred_flat[m_rows] * sc["std"] + sc["mean"]
    return unscaled.reshape(-1, 2, 3)


# ----------------------------- model --------------------------------------

class PolicyMLP(nn.Module):
    def __init__(self, in_dim=None, hidden=256, out_dim=6):
        if in_dim is None:
            in_dim = 43
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def masked_mse(pred, target_scaled, mask):
    """pred, target_scaled: [B,6]; mask: [B,2]."""
    pred3 = pred.view(-1, 2, 3)
    tgt3 = target_scaled.view(-1, 2, 3)
    m3 = mask.unsqueeze(-1).expand_as(pred3)
    sq = (pred3 - tgt3) ** 2 * m3
    return sq.sum() / m3.sum().clamp(min=1)


# ----------------------------- training -----------------------------------

def train_loop(model, train_t, val_t, args, seed):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_left = args.patience
    log = []

    Xt, Yt, Mt = train_t
    Xv, Yv, Mv = val_t
    N = Xt.shape[0]
    rng = np.random.RandomState(seed)

    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(N)
        train_losses = []
        for i in range(0, N, args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = masked_mse(pred, Yt[idx], Mt[idx])
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            pred_v = model(Xv)
            val_loss = masked_mse(pred_v, Yv, Mv).item()
        train_loss = float(np.mean(train_losses))
        log.append({"epoch": epoch, "train": train_loss, "val": val_loss})

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                logger.info("early stop epoch %d  best_val=%.4f", epoch, best_val)
                break

    model.load_state_dict(best_state)
    return model, log, best_val


# ----------------------------- evaluation ---------------------------------

def r2_vec(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-12))


def r2_scalar(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-12))


def evaluate_unscaled(model, val_data, target_scalers, feat_scaler):
    """Returns per-material per-group metrics dict."""
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(scale_features(val_data["features"], feat_scaler)).float()
        pred_scaled = model(X).numpy()
    pred_unscaled = unscale_predictions(pred_scaled, val_data["material"], target_scalers)

    out = {}
    for m in sorted(set(val_data["material"].tolist())):
        m_rows = val_data["material"] == m
        out[m] = {}
        for g in range(2):
            active = val_data["action_mask"][m_rows, g] > 0.5
            n = int(active.sum())
            if n < 5:
                continue
            true_g = val_data["action"][m_rows][active, g, :]
            pred_g = pred_unscaled[m_rows][active, g, :]
            true_mag = np.linalg.norm(true_g, axis=1)
            pred_mag = np.linalg.norm(pred_g, axis=1)
            nz = true_mag > 1e-6
            dir_cos = float(
                ((pred_g[nz] * true_g[nz]).sum(axis=1)
                 / (pred_mag[nz] * true_mag[nz] + 1e-12)).mean()
            ) if nz.any() else float("nan")
            out[m][f"group_{g}"] = {
                "n_rows": n,
                "mse_per_axis": ((pred_g - true_g) ** 2).mean(axis=0).tolist(),
                "mag_r2": r2_scalar(true_mag, pred_mag),
                "dir_cosine": dir_cos,
                "vec_r2": r2_vec(true_g, pred_g),
            }
    return out


# ----------------------------- IO -----------------------------------------

def save_seed(out_dir: Path, model, feat_scaler, target_scalers, metrics, train_log,
              best_val, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "policy.pt")
    with open(out_dir / "feat_scaler.pkl", "wb") as f:
        pickle.dump(feat_scaler, f)
    with open(out_dir / "target_scalers.pkl", "wb") as f:
        pickle.dump(target_scalers, f)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"metrics": metrics, "best_val_loss": float(best_val),
                   "args": {k: str(v) if isinstance(v, Path) else v
                            for k, v in vars(args).items()}}, f, indent=2)
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)


def aggregate(out_root: Path):
    summary = {}
    seed_dirs = sorted(out_root.glob("seed_*"))
    if not seed_dirs:
        return summary
    per_seed = []
    for sd in seed_dirs:
        with open(sd / "metrics.json") as f:
            per_seed.append(json.load(f)["metrics"])
    # Find common (material, group) pairs across seeds
    mats = set()
    for s in per_seed:
        mats |= set(s.keys())
    for m in sorted(mats):
        summary[m] = {}
        groups = set()
        for s in per_seed:
            if m in s:
                groups |= set(s[m].keys())
        for g in sorted(groups):
            metric_keys = ["mag_r2", "dir_cosine", "vec_r2"]
            for mk in metric_keys:
                vals = [s[m][g][mk] for s in per_seed
                        if m in s and g in s[m] and mk in s[m][g]
                        and s[m][g][mk] == s[m][g][mk]]  # filter NaN
                if vals:
                    arr = np.array(vals)
                    summary[m].setdefault(g, {})[f"{mk}_mean"] = float(arr.mean())
                    summary[m].setdefault(g, {})[f"{mk}_std"] = float(arr.std())
                    summary[m].setdefault(g, {})[f"{mk}_seeds"] = len(vals)
    return summary


# ----------------------------- gate runners -------------------------------

def prepare_split_and_scale(data, val_frac, seed):
    val_mask = random_block_split(data["case_name"], val_frac, seed)
    assert_split_sanity(data, val_mask, val_frac)
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    train = {k: v[train_idx] for k, v in data.items()}
    val = {k: v[val_idx] for k, v in data.items()}
    feat_scaler = fit_feature_scaler(train["features"])
    target_scalers = fit_target_scalers(train["action"], train["material"],
                                        train["action_mask"])
    return train, val, feat_scaler, target_scalers


def to_tensors(split, feat_scaler, target_scalers):
    Xs = scale_features(split["features"], feat_scaler)
    Ys = scale_targets(split["action"], split["material"], target_scalers)
    return (torch.from_numpy(Xs).float(),
            torch.from_numpy(Ys).float(),
            torch.from_numpy(split["action_mask"]).float())


def run_check(args):
    data = load_dataset(args.data, args.exclude_materials)
    logger.info("loaded %d rows after exclusions; features=%s",
                len(data["state"]), data["features"].shape)
    train, val, fs, ts = prepare_split_and_scale(data, args.val_frac, args.seed)
    logger.info("train materials: %s", sorted(set(train["material"].tolist())))
    logger.info("val materials:   %s", sorted(set(val["material"].tolist())))
    Xs = scale_features(train["features"], fs)
    logger.info("scaled feature mean abs (should ≈ 0): %.4f", np.abs(Xs.mean(axis=0)).max())
    logger.info("scaled feature std mean (should ≈ 1): %.4f", Xs.std(axis=0).mean())
    for m, sc in ts.items():
        logger.info("target scaler [%s]: std min=%.6f max=%.6f  n_active=%s",
                    m, sc["std"].min(), sc["std"].max(), sc["n_active"])
        if sc["std"].min() < 1e-6:
            logger.warning("scaler [%s] has near-zero std on some axis", m)
    Ys = scale_targets(train["action"], train["material"], ts)
    # NaN check
    for name, arr in [("Xs", Xs), ("Ys", Ys)]:
        if not np.isfinite(arr).all():
            raise AssertionError(f"NaN/Inf in scaled {name}")
    logger.info("G1 PASS: no NaN, materials match, scalers sane")
    print("\n=== G1 dataloader smoke: PASS ===")


def run_overfit(args):
    data = load_dataset(args.data, args.exclude_materials)
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(data["state"]), size=100, replace=False)
    tiny = {k: v[idx] for k, v in data.items()}
    fs = fit_feature_scaler(tiny["features"])
    ts = fit_target_scalers(tiny["action"], tiny["material"], tiny["action_mask"])
    Xs = scale_features(tiny["features"], fs)
    Ys = scale_targets(tiny["action"], tiny["material"], ts)
    Xt = torch.from_numpy(Xs).float()
    Yt = torch.from_numpy(Ys).float()
    Mt = torch.from_numpy(tiny["action_mask"]).float()

    torch.manual_seed(args.seed)
    in_dim = tiny["features"].shape[1]
    model = PolicyMLP(in_dim=in_dim, hidden=args.hidden, out_dim=6)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    final_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        pred = model(Xt)
        loss = masked_mse(pred, Yt, Mt)
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            logger.info("G2 epoch %d  loss=%.6f", epoch, loss.item())
        final_loss = loss.item()
    passed = final_loss < 1e-3
    logger.info("G2 final loss=%.6f  threshold=1e-3  %s",
                final_loss, "PASS" if passed else "FAIL")
    print(f"\n=== G2 tiny overfit: {'PASS' if passed else 'FAIL'} "
          f"(final loss={final_loss:.6f}) ===")
    if not passed:
        with torch.no_grad():
            pred = model(Xt[:5]).numpy()
        print("first 5 (pred, target) pairs:")
        for i in range(5):
            print(f"  row {i}  pred={pred[i]}  target={Ys[i]}")
        sys.exit(1)


def run_inspect(args):
    if args.model_dir is None:
        raise SystemExit("--inspect_only requires --model_dir")
    md = Path(args.model_dir)
    # Infer in_dim from feat_scaler shape (saves us from sniffing the state_dict)
    in_dim = int(fs["mean"].shape[0])
    model = PolicyMLP(in_dim=in_dim, hidden=args.hidden, out_dim=6)
    model.load_state_dict(torch.load(md / "policy.pt", map_location="cpu",
                                     weights_only=True))
    with open(md / "feat_scaler.pkl", "rb") as f:
        fs = pickle.load(f)
    with open(md / "target_scalers.pkl", "rb") as f:
        ts = pickle.load(f)
    data = load_dataset(args.data, args.exclude_materials)
    # Use the same split as training (read seed from metrics.json if possible).
    seed = args.seed
    try:
        with open(md / "metrics.json") as f:
            metrics_blob = json.load(f)
        seed = int(metrics_blob["args"].get("seed", seed))
    except Exception:
        pass
    val_mask = random_block_split(data["case_name"], args.val_frac, seed)
    val = {k: v[val_mask] for k, v in data.items()}
    rng = np.random.RandomState(0)
    idx = rng.choice(len(val["state"]), size=min(args.n_examples, len(val["state"])),
                     replace=False)
    sample = {k: v[idx] for k, v in val.items()}
    with torch.no_grad():
        Xs = torch.from_numpy(scale_features(sample["features"], fs)).float()
        pred_scaled = model(Xs).numpy()
    pred = unscale_predictions(pred_scaled, sample["material"], ts)
    print(f"\n=== G4 inspect ({args.n_examples} samples from val) ===")
    print(f"{'i':>2}  {'mat':>6} {'n_ctrl':>6}  "
          f"{'pred_g0 (xyz, mm)':>30}  {'true_g0 (xyz, mm)':>30}  "
          f"{'|err_g0| mm':>12}  {'mag(true_g0) mm':>15}")
    for i in range(len(idx)):
        m = str(sample["material"][i])
        nc = int(sample["n_ctrl_parts"][i])
        p0 = pred[i, 0] * 1000.0   # m -> mm
        t0 = sample["action"][i, 0] * 1000.0
        err = float(np.linalg.norm(p0 - t0))
        mag = float(np.linalg.norm(t0))
        print(f"{i:>2}  {m:>6} {nc:>6}  "
              f"({p0[0]:+7.2f},{p0[1]:+7.2f},{p0[2]:+7.2f})  "
              f"({t0[0]:+7.2f},{t0[1]:+7.2f},{t0[2]:+7.2f})  "
              f"{err:>12.3f}  {mag:>15.3f}")
        if nc == 2:
            p1 = pred[i, 1] * 1000.0
            t1 = sample["action"][i, 1] * 1000.0
            print(f"      group1:  pred=({p1[0]:+7.2f},{p1[1]:+7.2f},{p1[2]:+7.2f})  "
                  f"true=({t1[0]:+7.2f},{t1[1]:+7.2f},{t1[2]:+7.2f})")
    # Also report mean group-1 prediction magnitude for single-ctrl rows.
    single = sample["n_ctrl_parts"] == 1
    if single.any():
        g1_mag = np.linalg.norm(pred[single, 1, :], axis=1)
        print(f"\nsingle-ctrl rows: mean ‖pred_g1‖ = {g1_mag.mean() * 1000:.3f} mm "
              f"(should be near 0)")


def run_training(args, seeds):
    data = load_dataset(args.data, args.exclude_materials)
    in_dim = data["features"].shape[1]
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        logger.info("=== seed %d ===  (in_dim=%d)", seed, in_dim)
        torch.manual_seed(seed)
        np.random.seed(seed)
        train, val, fs, ts = prepare_split_and_scale(data, args.val_frac, seed)
        Xt, Yt, Mt = to_tensors(train, fs, ts)
        Xv, Yv, Mv = to_tensors(val, fs, ts)
        model = PolicyMLP(in_dim=in_dim, hidden=args.hidden, out_dim=6)
        model, log, best_val = train_loop(model, (Xt, Yt, Mt), (Xv, Yv, Mv),
                                          args, seed)
        metrics = evaluate_unscaled(model, val, ts, fs)
        sd_dir = args.out / f"seed_{seed}"
        save_seed(sd_dir, model, fs, ts, metrics, log, best_val, args)
        logger.info("seed %d saved to %s", seed, sd_dir)
        # Brief per-seed headline
        line = f"seed {seed}: "
        for m, gs in metrics.items():
            for g, mk in gs.items():
                line += f"{m}/{g} vec_r2={mk['vec_r2']:+.3f}  "
        logger.info(line)

    if len(seeds) > 1:
        summary = aggregate(args.out)
        with open(args.out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\n=== Multi-seed summary (mean ± std over seeds) ===")
        for m, gs in summary.items():
            for g, mk in gs.items():
                if "vec_r2_mean" in mk:
                    print(f"  {m:6s} {g:>8s}  "
                          f"vec_r2 = {mk['vec_r2_mean']:+.3f} ± {mk['vec_r2_std']:.3f}  "
                          f"mag_r2 = {mk['mag_r2_mean']:+.3f} ± {mk['mag_r2_std']:.3f}  "
                          f"dir_cos = {mk['dir_cosine_mean']:+.3f} ± {mk['dir_cosine_std']:.3f}  "
                          f"(n_seeds={mk['vec_r2_seeds']})")


# ----------------------------- CLI ----------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--exclude_materials", nargs="*", default=["toy"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated seeds for multi-seed run")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--check_only", action="store_true", help="G1")
    p.add_argument("--overfit_tiny", action="store_true", help="G2")
    p.add_argument("--inspect_only", action="store_true", help="G4")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--n_examples", type=int, default=10)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.check_only:
        run_check(args)
    elif args.overfit_tiny:
        run_overfit(args)
    elif args.inspect_only:
        run_inspect(args)
    else:
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
        run_training(args, seeds)


if __name__ == "__main__":
    main()
