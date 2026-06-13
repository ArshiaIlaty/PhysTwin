"""
stiffness_conditioned_policy.py
────────────────────────────────────────────────────────────────────────────
Adds PhysTwin spring-stiffness conditioning to Malak's existing
behavioural cloning (BC) policy.

Three drop-in components:

  1. StiffnessFeatureExtractor   — converts raw stiffness arrays to a
                                    fixed-dim embedding
  2. StiffnessConditionedMLP     — Malak's movement predictor with
                                    stiffness embedding concatenated
  3. StiffnessConditionedDataset — wraps Malak's existing episode CSV
                                    and joins in the stiffness features

Usage
─────
    from stiffness_conditioned_policy import (
        StiffnessConditionedMLP,
        StiffnessConditionedDataset,
        load_stiffness_cache,
    )

    stiffness_cache = load_stiffness_cache("stiffness_per_case.csv")

    dataset = StiffnessConditionedDataset(
        episodes_csv   = "malak_episodes.csv",
        stiffness_cache= stiffness_cache,
    )

    model = StiffnessConditionedMLP(
        obs_dim      = 12,   # Malak's observation dimension
        action_dim   = 6,    # Malak's action dimension
        stiff_dim    = 4,    # k_mean, k_std, k_median, log(k_mean)
        hidden_dims  = [256, 256],
    )
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 1. Stiffness feature vector (per-case, computed once)
# ─────────────────────────────────────────────────────────────────────────────

class StiffnessFeatureExtractor:
    """
    Converts a raw stiffness array k (shape: n_springs,) into a compact
    fixed-dimensional feature vector suitable for policy conditioning.

    Features (4 dims by default):
        [k_mean, k_std, k_median, log(k_mean + 1)]

    All features are z-score normalised using statistics computed from
    the training set so the policy doesn't see raw N/m values.
    """

    FEATURE_NAMES = ["k_mean", "k_std", "k_median", "log_k_mean"]

    def __init__(self):
        self._fit_mean = None
        self._fit_std  = None

    def extract(self, k_vals: np.ndarray) -> np.ndarray:
        """
        k_vals : 1-D array of per-spring stiffness values (N/m)
        Returns: feature vector of shape (4,), dtype float32
        """
        k = k_vals.astype(np.float32)
        feats = np.array([
            np.mean(k),
            np.std(k),
            np.median(k),
            np.log(np.mean(k) + 1.0),
        ], dtype=np.float32)
        return feats

    def fit(self, feature_matrix: np.ndarray):
        """
        Fit normalisation on training set.
        feature_matrix : shape (n_cases, 4)
        """
        self._fit_mean = feature_matrix.mean(axis=0)
        self._fit_std  = feature_matrix.std(axis=0) + 1e-8

    def transform(self, feats: np.ndarray) -> np.ndarray:
        """
        feats : shape (4,) or (n, 4)
        Returns z-scored features.
        """
        if self._fit_mean is None:
            return feats
        return (feats - self._fit_mean) / self._fit_std

    def fit_transform(self, feature_matrix: np.ndarray) -> np.ndarray:
        self.fit(feature_matrix)
        return self.transform(feature_matrix)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"mean": self._fit_mean, "std": self._fit_std}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self._fit_mean = d["mean"]
        self._fit_std  = d["std"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load stiffness cache (output of stiffness_force_analysis.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_stiffness_cache(csv_path: str) -> Dict[str, np.ndarray]:
    """
    Load the stiffness_per_case.csv produced by stiffness_force_analysis.py.
    Returns a dict: {case_name: feature_vector (shape 4,)}
    """
    df = pd.read_csv(csv_path)
    extractor = StiffnessFeatureExtractor()

    # Build raw feature matrix for normalisation
    raw = np.stack([
        np.array([
            row["k_mean"],
            row["k_std"],
            row["k_median"],
            np.log(row["k_mean"] + 1.0),
        ], dtype=np.float32)
        for _, row in df.iterrows()
    ])
    normed = extractor.fit_transform(raw)

    cache = {}
    for i, (_, row) in enumerate(df.iterrows()):
        cache[row["case_name"]] = normed[i]

    print(f"  Loaded stiffness features for {len(cache)} cases")
    return cache, extractor


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class StiffnessConditionedDataset(Dataset):
    """
    Wraps Malak's episode CSV and joins in stiffness features.

    Expected CSV columns (adjust EPISODE_COLS if different):
        case_name   : str
        observation : comma-sep floats  OR  obs_0, obs_1, ..., obs_N
        action      : comma-sep floats  OR  act_0, act_1, ..., act_M
        gt_force    : float (optional, for comparison)

    If observation/action are multi-column (e.g. obs_0 … obs_11),
    set obs_prefix / act_prefix accordingly.
    """

    def __init__(
        self,
        episodes_csv:    str,
        stiffness_cache: Dict[str, np.ndarray],
        obs_prefix:      str = "obs",
        act_prefix:      str = "act",
        obs_col:         str = "observation",   # alt: single-column format
        act_col:         str = "action",
        case_col:        str = "case_name",
        stiff_case_col:  str = "stiffness_case",
        gt_force_col:    str = "gt_force",
        stiff_dim:       int = 4,
    ):
        super().__init__()
        self.stiffness_cache = stiffness_cache
        self.stiff_dim       = stiff_dim
        self.case_col        = case_col
        self.gt_force_col    = gt_force_col

        df = pd.read_csv(episodes_csv)

        # ── parse observations ────────────────────────────────────────────
        obs_multi = [c for c in df.columns if c.startswith(obs_prefix + "_")]
        act_multi = [c for c in df.columns if c.startswith(act_prefix + "_")]

        if obs_multi:
            self.obs  = df[obs_multi].values.astype(np.float32)
        elif obs_col in df.columns:
            # single-column format: "0.1,0.2,0.3"
            self.obs = np.stack([
                np.fromstring(s, sep=",").astype(np.float32)
                for s in df[obs_col]
            ])
        else:
            raise ValueError(f"Cannot find observation columns in {episodes_csv}. "
                             f"Available: {list(df.columns)}")

        if act_multi:
            self.acts = df[act_multi].values.astype(np.float32)
        elif act_col in df.columns:
            self.acts = np.stack([
                np.fromstring(s, sep=",").astype(np.float32)
                for s in df[act_col]
            ])
        else:
            raise ValueError(f"Cannot find action columns.")

        self.cases = df[case_col].tolist()
        if stiff_case_col in df.columns:
            self.stiff_cases = df[stiff_case_col].tolist()
        else:
            self.stiff_cases = [
                c.split("__synth_")[0] if "__synth_" in c else c
                for c in self.cases
            ]
        self.gt_forces = (df[gt_force_col].values.astype(np.float32)
                          if gt_force_col in df.columns else None)

        # ── build stiffness feature matrix ────────────────────────────────
        missing = [c for c in self.stiff_cases if c not in stiffness_cache]
        if missing:
            print(f"  [warn] {len(set(missing))} stiffness keys not in cache "
                  f"(affecting {len(missing)} rows) — zero features used.")
        zero = np.zeros(stiff_dim, dtype=np.float32)
        self.stiff_feats = np.stack([
            stiffness_cache.get(c, zero) for c in self.stiff_cases
        ])

        assert len(self.obs) == len(self.acts) == len(self.stiff_feats)
        print(f"  Dataset: {len(self.obs)} steps  |  "
              f"obs_dim={self.obs.shape[1]}  "
              f"act_dim={self.acts.shape[1]}  "
              f"stiff_dim={stiff_dim}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs    = torch.tensor(self.obs[idx])
        act    = torch.tensor(self.acts[idx])
        stiff  = torch.tensor(self.stiff_feats[idx])
        sample = {"obs": obs, "action": act, "stiffness": stiff}
        if self.gt_forces is not None:
            sample["gt_force"] = torch.tensor(
                self.gt_forces[idx], dtype=torch.float32)
        return sample


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model: stiffness-conditioned MLP policy
# ─────────────────────────────────────────────────────────────────────────────

class StiffnessConditionedMLP(nn.Module):
    """
    Malak's movement-predictor MLP with stiffness embedding concatenated
    to the observation.

    Architecture:
        [obs (obs_dim) || stiffness_embedding (stiff_embed_dim)]
        → MLP(hidden_dims)
        → action (act_dim)

    The stiffness is passed through a small learned embedding network
    before concatenation, so the policy can learn a nonlinear
    transformation of the physical features.
    """

    def __init__(
        self,
        obs_dim:         int,
        action_dim:      int,
        stiff_dim:       int   = 4,
        stiff_embed_dim: int   = 16,
        hidden_dims:     List[int] = (256, 256),
        dropout:         float = 0.1,
    ):
        super().__init__()

        # Stiffness embedding network
        self.stiff_encoder = nn.Sequential(
            nn.Linear(stiff_dim, stiff_embed_dim),
            nn.LayerNorm(stiff_embed_dim),
            nn.ReLU(),
            nn.Linear(stiff_embed_dim, stiff_embed_dim),
            nn.ReLU(),
        )

        # Main policy MLP
        in_dim  = obs_dim + stiff_embed_dim
        layers  = []
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        obs:      torch.Tensor,   # (B, obs_dim)
        stiffness: torch.Tensor,  # (B, stiff_dim)
    ) -> torch.Tensor:            # (B, action_dim)
        stiff_emb = self.stiff_encoder(stiffness)
        x = torch.cat([obs, stiff_emb], dim=-1)
        return self.mlp(x)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Training loop (BC)
# ─────────────────────────────────────────────────────────────────────────────

def train_bc(
    model:       nn.Module,
    dataset:     StiffnessConditionedDataset,
    n_epochs:    int   = 100,
    lr:          float = 1e-3,
    batch_size:  int   = 256,
    val_frac:    float = 0.15,
    device:      str   = "cuda" if torch.cuda.is_available() else "cpu",
    save_path:   str   = "bc_stiff_conditioned.pt",
    patience:    int   = 15,
) -> dict:
    """
    Standard BC training loop with:
    - Train/val split
    - Early stopping
    - LR scheduler
    - Loss logged per epoch
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience // 2, factor=0.5, verbose=True)
    criterion = nn.MSELoss()

    n_val  = int(len(dataset) * val_frac)
    n_tr   = len(dataset) - n_val
    tr_set, val_set = random_split(
        dataset, [n_tr, n_val],
        generator=torch.Generator().manual_seed(42))

    tr_loader  = DataLoader(tr_set,  batch_size=batch_size,
                            shuffle=True,  drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False)

    history = {"train_loss": [], "val_loss": []}
    best_val, best_epoch, best_state = float("inf"), 0, None

    for epoch in range(1, n_epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for batch in tr_loader:
            obs    = batch["obs"].to(device)
            action = batch["action"].to(device)
            stiff  = batch["stiffness"].to(device)

            pred = model(obs, stiff)
            loss = criterion(pred, action)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()

        tr_loss /= len(tr_loader)

        # ── validate ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                obs    = batch["obs"].to(device)
                action = batch["action"].to(device)
                stiff  = batch["stiffness"].to(device)
                pred   = model(obs, stiff)
                val_loss += criterion(pred, action).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:4d}/{n_epochs}  "
                  f"train={tr_loss:.5f}  val={val_loss:.5f}  "
                  f"best_val={best_val:.5f} @ ep {best_epoch}")

        # Early stopping
        if epoch - best_epoch >= patience:
            print(f"  Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    obs_dim = dataset.obs.shape[1]
    act_dim = dataset.acts.shape[1]
    torch.save({"model_state": best_state,
                "history": history,
                "config": {"obs_dim": obs_dim, "action_dim": act_dim},
                }, save_path)
    print(f"\n  Best model saved → {save_path}  (val_loss={best_val:.5f})")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 6. Ablation: unconditioned baseline (for comparison with Malak)
# ─────────────────────────────────────────────────────────────────────────────

class BaselineMLP(nn.Module):
    """
    Identical architecture to StiffnessConditionedMLP but WITHOUT
    stiffness conditioning.  Use this to prove that stiffness helps.
    """

    def __init__(
        self,
        obs_dim:     int,
        action_dim:  int,
        hidden_dims: List[int] = (256, 256),
        dropout:     float = 0.1,
    ):
        super().__init__()
        in_dim = obs_dim
        layers = []
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, stiffness=None) -> torch.Tensor:
        return self.mlp(obs)   # stiffness ignored — same API for easy swapping


# ─────────────────────────────────────────────────────────────────────────────
# 7. Evaluation: compare conditioned vs unconditioned per material
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_per_material(
    model:       nn.Module,
    dataset:     StiffnessConditionedDataset,
    device:      str = "cpu",
) -> pd.DataFrame:
    """
    Runs the trained model on the full dataset and returns a DataFrame
    with per-case MAE — so you can group by obj_type and compare.
    """
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    all_preds, all_acts, all_stiffs, step = [], [], [], 0
    with torch.no_grad():
        for batch in loader:
            obs   = batch["obs"].to(device)
            stiff = batch["stiffness"].to(device)
            pred  = model(obs, stiff).cpu().numpy()
            all_preds.append(pred)
            all_acts.append(batch["action"].numpy())

    preds = np.concatenate(all_preds)
    acts  = np.concatenate(all_acts)
    mae_per_step = np.abs(preds - acts).mean(axis=1)

    df = pd.DataFrame({
        "case_name": dataset.cases,
        "obj_type":  [_infer_type(c) for c in dataset.cases],
        "mae":       mae_per_step,
        "k_mean":    dataset.stiff_feats[:, 0],   # normalised
    })
    return df


def _infer_type(case_name: str) -> str:
    """Quick object-type inference from case name."""
    n = case_name.lower()
    if any(k in n for k in ["rope"]):                    return "rope"
    if any(k in n for k in ["cloth"]):                   return "cloth"
    if any(k in n for k in ["sloth","stuffed","bunny"]): return "stuffed"
    if any(k in n for k in ["package","box","cardboard"]): return "package"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Quick demo / smoke test (no real data required)
# ─────────────────────────────────────────────────────────────────────────────

def _demo():
    """
    Run a small smoke test with synthetic data.
    Verifies model forward pass and training loop.
    """
    print("\n── Smoke test (synthetic data) ──")
    torch.manual_seed(0)

    obs_dim, act_dim, B = 12, 6, 128

    # Synthetic stiffness: high-k episodes should need smaller actions
    k_vals = torch.rand(B, 4)       # normalised stiffness features
    obs    = torch.randn(B, obs_dim)
    # GT action = inverse of stiffness (what we want the policy to learn)
    act    = obs[:, :act_dim] / (k_vals[:, :1] + 0.5)

    model   = StiffnessConditionedMLP(obs_dim, act_dim)
    pred    = model(obs, k_vals)
    loss    = nn.MSELoss()(pred, act)

    baseline = BaselineMLP(obs_dim, act_dim)
    pred_b   = baseline(obs)

    print(f"  StiffnessConditionedMLP  output shape: {pred.shape}  "
          f"loss (untrained): {loss.item():.4f}")
    print(f"  BaselineMLP              output shape: {pred_b.shape}")
    print(f"  ✓ Forward pass OK for both models")

    # Tiny train loop
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(50):
        p = model(obs, k_vals)
        l = nn.MSELoss()(p, act)
        opt.zero_grad(); l.backward(); opt.step()
    print(f"  After 50 grad steps: loss = {l.item():.4f}")
    print("  ✓ Training loop OK\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Stiffness-conditioned BC policy for PhysTwin")
    parser.add_argument("--demo", action="store_true",
                        help="Run smoke test with synthetic data")
    parser.add_argument("--train", action="store_true",
                        help="Train on exported episodes CSV")
    parser.add_argument("--episodes_csv", default="scenario_sweeps/episodes.csv")
    parser.add_argument("--stiffness_csv",
                        default="scenario_sweeps/analysis_results/stiffness_per_case.csv")
    parser.add_argument("--out_dir", default="scenario_sweeps/policy_models")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--baseline", action="store_true",
                        help="Also train unconditioned BaselineMLP for comparison")
    args = parser.parse_args()

    if args.demo:
        _demo()
    elif args.train:
        os.makedirs(args.out_dir, exist_ok=True)
        if not os.path.isfile(args.episodes_csv):
            raise SystemExit(
                f"Missing {args.episodes_csv}. Run:\n"
                f"  bash scenario_sweeps/run_policy_data_pipeline.sh")
        if not os.path.isfile(args.stiffness_csv):
            raise SystemExit(
                f"Missing {args.stiffness_csv}. Run stiffness extraction first.")

        print("\n── Loading stiffness cache ──")
        stiff_cache, _ = load_stiffness_cache(args.stiffness_csv)

        print("\n── Loading episodes ──")
        dataset = StiffnessConditionedDataset(
            episodes_csv=args.episodes_csv,
            stiffness_cache=stiff_cache,
        )
        obs_dim = dataset.obs.shape[1]
        act_dim = dataset.acts.shape[1]

        print(f"\n── Training StiffnessConditionedMLP "
              f"(obs={obs_dim}, act={act_dim}) ──")
        model = StiffnessConditionedMLP(obs_dim, act_dim)
        train_bc(
            model, dataset,
            n_epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            save_path=os.path.join(args.out_dir, "bc_stiff_conditioned.pt"),
        )
        eval_df = evaluate_per_material(model, dataset)
        eval_path = os.path.join(args.out_dir, "eval_stiff_conditioned.csv")
        eval_df.to_csv(eval_path, index=False)
        print(f"  Per-material eval → {eval_path}")
        print(eval_df.groupby("obj_type")["mae"].mean())

        if args.baseline:
            print("\n── Training BaselineMLP (no stiffness) ──")
            baseline = BaselineMLP(obs_dim, act_dim)
            train_bc(
                baseline, dataset,
                n_epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                save_path=os.path.join(args.out_dir, "bc_baseline.pt"),
            )
            base_eval = evaluate_per_material(baseline, dataset)
            base_eval.to_csv(
                os.path.join(args.out_dir, "eval_baseline.csv"), index=False)
    else:
        print("Import this module, or run with --demo or --train.")
        print("Full pipeline: bash scenario_sweeps/run_policy_data_pipeline.sh")
