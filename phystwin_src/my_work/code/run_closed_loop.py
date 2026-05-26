#!/usr/bin/env python3
"""
run_closed_loop.py

Drive PhysTwin with a learned (or trivial) policy via the new run_policy()
method on InvPhyTrainerWarp. One script, five profiles selected via --profile.

Profiles (G1..G5 from step3_plan.md):
  random                  G1   random Δ per frame; no model loaded
  zero                    G2   Δ = 0 every frame; no model loaded
  replay_action           G3   Δ = recorded per-group centroid Δ; no model loaded
  policy_recorded_goal    G4   trained policy; F_goal = recorded y_per_ctrl
  policy_ramp             G5   trained policy; F_goal = linear ramp on group 0

Plan: my_work/docs/closed_loop_control/step3_plan.md

Usage:
  python my_work/code/run_closed_loop.py \\
      --case_name single_push_rope_1 \\
      --profile replay_action \\
      --out_dir my_work/results/closed_loop_rollouts
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

# Local imports
SCRIPT_DIR = Path(__file__).resolve().parent
MY_WORK = SCRIPT_DIR.parent
RESULTS = MY_WORK / "results"
REPO_ROOT = MY_WORK.parent   # = /share/.../PhysTwin/phystwin_src

# IMPORTANT: there are two `qqtt/` trees in this repo:
#   /share/.../PhysTwin/qqtt/                    (top-level copy)
#   /share/.../PhysTwin/phystwin_src/qqtt/       (this one — my edits, incl run_policy)
# We MUST pick the phystwin_src copy. Prepend REPO_ROOT so Python finds it first.
sys.path.insert(0, str(REPO_ROOT))
# Also ensure my_work/code is on sys.path so we can import features.
sys.path.insert(0, str(SCRIPT_DIR))

from features import (  # noqa: E402
    compute_31d_features, get_log_spring_Y_mean, get_material_descriptor,
)

from qqtt import InvPhyTrainerWarp  # noqa: E402
from qqtt.utils import cfg, logger as qqtt_logger  # noqa: E402

# Sanity: assert we got the qqtt that has run_policy.
import qqtt as _qqtt_mod  # noqa: E402
_qqtt_path = _qqtt_mod.__file__
if str(REPO_ROOT) not in _qqtt_path:
    raise RuntimeError(
        f"Imported qqtt from {_qqtt_path} but expected the copy under {REPO_ROOT}. "
        "The closed-loop driver needs the run_policy() edit in phystwin_src/qqtt/."
    )
if not hasattr(InvPhyTrainerWarp, "run_policy"):
    raise RuntimeError(
        f"InvPhyTrainerWarp at {_qqtt_path} has no run_policy attribute. "
        "Check the upstream edit in qqtt/engine/trainer_warp.py."
    )

logger = logging.getLogger("run_closed_loop")


DEFAULT_OUT_DIR = RESULTS / "closed_loop_rollouts"
DEFAULT_POLICY_DIR = RESULTS / "models_policy" / "seed_1"
DEFAULT_DATASET_V2 = RESULTS / "dataset_v2"


# ----------------------------- helpers ------------------------------------

def infer_n_ctrl_parts_and_cfg_type(case_name: str) -> tuple[int, str]:
    """Mirror extract_dataset.py's heuristic + overrides."""
    if case_name == "rope_double_hand":
        n = 2
    elif case_name.startswith("double_"):
        n = 2
    elif case_name.startswith("single_"):
        n = 1
    else:
        n = 1  # fallback
    cfg_type = "cloth" if "cloth" in case_name else "real"
    return n, cfg_type


def setup_trainer(case_name: str, n_ctrl_parts: int, cfg_type: str,
                   base_path: str = "./data/different_types"):
    """Mirror extract_dataset.py:extract_case setup. Returns (trainer, best_model_path)."""
    cfg_file = "configs/cloth.yaml" if cfg_type == "cloth" else "configs/real.yaml"
    cfg.load_from_yaml(cfg_file)

    optimal_path = f"./experiments_optimization/{case_name}/optimal_params.pkl"
    with open(optimal_path, "rb") as f:
        cfg.set_optimal_params(pickle.load(f))

    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array([np.linalg.inv(c) for c in c2ws])

    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        meta = json.load(f)
    cfg.intrinsics = np.array(meta["intrinsics"])
    cfg.WH = meta["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"

    best_models = glob.glob(f"experiments/{case_name}/train/best_*.pth")
    assert best_models, f"No best_*.pth for {case_name}"

    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=f"./experiments/{case_name}",
        pure_inference_mode=True,
    )
    return trainer, best_models[0]


# ----------------------------- policy loader ------------------------------

class PolicyMLP(nn.Module):
    def __init__(self, in_dim=43, hidden=256, out_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_policy_artifacts(policy_dir: Path):
    """Load model + scalers from a trained seed directory.

    Both `in_dim` and `hidden` are inferred from the saved state_dict so
    we can load any (43/44/45 input)×(256/512/…) variant without
    hardcoded args.
    """
    with open(policy_dir / "feat_scaler.pkl", "rb") as f:
        feat_scaler = pickle.load(f)
    with open(policy_dir / "target_scalers.pkl", "rb") as f:
        target_scalers = pickle.load(f)
    sd = torch.load(policy_dir / "policy.pt", map_location="cpu", weights_only=True)
    # Sniff dims from the first layer weight shape: [hidden, in_dim]
    first_layer = sd["net.0.weight"]
    hidden, in_dim = int(first_layer.shape[0]), int(first_layer.shape[1])
    assert in_dim == int(feat_scaler["mean"].shape[0]), \
        f"state_dict in_dim={in_dim} mismatches feat_scaler {feat_scaler['mean'].shape[0]}"
    model = PolicyMLP(in_dim=in_dim, hidden=hidden, out_dim=6)
    model.load_state_dict(sd)
    model.eval()
    return model, feat_scaler, target_scalers


def make_policy_callable(model, feat_scaler, target_scalers, material: str,
                          F_user_target=None, goal_shaping: str = "direct",
                          max_step_force: float = 500.0,
                          log_shaped_goals: list | None = None,
                          material_descriptor=None):
    # material_descriptor may be a scalar (Fix E, 1 dim) or 2-vector (Fix F, 2 dim)
    """Closure: (state31, force_now_pad, force_goal_pad, frame_idx) -> Δ [2,3].

    goal_shaping:
      'direct'      — pass force_goal_pad (the user target at this frame) into
                      the model unchanged. Original Step 3 behavior.
      'incremental' — IGNORE force_goal_pad and feed the model a shaped goal
                      that's at most `max_step_force` Newtons away from
                      force_now_pad (per element), heading toward
                      F_user_target[frame_idx]. Keeps the policy's input
                      in-distribution.

    If log_shaped_goals is a list, the shaped goal at each frame is appended.
    """
    if material not in target_scalers:
        raise KeyError(f"material '{material}' not in target_scalers "
                       f"(have {list(target_scalers.keys())})")
    if goal_shaping == "incremental" and F_user_target is None:
        raise ValueError("incremental shaping requires F_user_target")

    ts = target_scalers[material]
    mean = ts["mean"].astype(np.float32)
    std = ts["std"].astype(np.float32)
    f_mean = feat_scaler["mean"].astype(np.float32)
    f_std = feat_scaler["std"].astype(np.float32)
    max_step = float(max_step_force)
    expected_md_dim = f_mean.shape[0] - 43  # 0 if no descriptor, 1 (Fix E), 2 (Fix F)
    md = None
    if expected_md_dim > 0:
        if material_descriptor is None:
            raise ValueError(
                f"Policy expects {f_mean.shape[0]}-dim input (material descriptor "
                f"of {expected_md_dim} dims) but none was passed.")
        md = np.atleast_1d(np.asarray(material_descriptor, dtype=np.float32)).flatten()
        if md.shape[0] != expected_md_dim:
            raise ValueError(
                f"material_descriptor dim mismatch: got {md.shape[0]}, "
                f"policy expects {expected_md_dim}")

    def policy_fn(state31, force_now_pad, force_goal_pad, frame_idx):
        if goal_shaping == "incremental":
            user_target_t = np.asarray(F_user_target[frame_idx], dtype=np.float32)
            delta_target = np.clip(
                user_target_t - force_now_pad, -max_step, +max_step
            )
            shaped_goal = force_now_pad + delta_target
        else:
            shaped_goal = np.asarray(force_goal_pad, dtype=np.float32)

        if log_shaped_goals is not None:
            log_shaped_goals.append(shaped_goal.copy())

        parts = [state31, force_now_pad.flatten(), shaped_goal.flatten()]
        if md is not None:
            parts.append(md)
        feats = np.concatenate(parts).astype(np.float32)
        x = (feats - f_mean) / f_std
        with torch.no_grad():
            y = model(torch.from_numpy(x).unsqueeze(0)).numpy()[0]
        delta = (y * std + mean).reshape(2, 3).astype(np.float32)
        return delta

    return policy_fn


# ----------------------------- profile builders ---------------------------

def build_profile(profile: str, case_name: str, n_ctrl_parts: int, material: str,
                  args, recorded_v2: dict | None):
    """Return (policy_fn, F_goal, info_dict)."""
    T = recorded_v2["controller_pos"].shape[0] if recorded_v2 is not None else args.max_frames

    if profile == "random":
        rng = np.random.RandomState(args.seed)
        deltas = rng.uniform(-1e-3, 1e-3, size=(T, 2, 3)).astype(np.float32)

        def policy_fn(s, fn, fg, t):
            return deltas[t]

        F_goal = np.zeros((T, 2, 3), dtype=np.float32)
        return policy_fn, F_goal, {"deltas_source": "random"}

    if profile == "zero":
        def policy_fn(s, fn, fg, t):
            return np.zeros((2, 3), dtype=np.float32)

        F_goal = np.zeros((T, 2, 3), dtype=np.float32)
        return policy_fn, F_goal, {"deltas_source": "zero"}

    if profile == "replay_action":
        assert recorded_v2 is not None, "replay_action needs recorded data"
        ctrl = recorded_v2["controller_pos"].astype(np.float32)  # [T, K, 3]
        # KMeans on frame 0 with same seed as run_policy
        K = ctrl.shape[0] if ctrl.ndim == 2 else ctrl.shape[1]
        ctrl0 = ctrl[0]
        if n_ctrl_parts == 1:
            group_ids = np.zeros(ctrl0.shape[0], dtype=np.int64)
        else:
            km = KMeans(n_clusters=n_ctrl_parts, random_state=0, n_init=10)
            group_ids = km.fit_predict(ctrl0).astype(np.int64)
        # Per-group centroid per frame
        centroids = np.zeros((ctrl.shape[0], 2, 3), dtype=np.float32)
        for g in range(n_ctrl_parts):
            centroids[:, g] = ctrl[:, group_ids == g].mean(axis=1)
        deltas = np.zeros_like(centroids)
        deltas[1:] = centroids[1:] - centroids[:-1]  # deltas[t] applied at step t

        def policy_fn(s, fn, fg, t):
            return deltas[t]

        F_goal = recorded_v2["y_per_ctrl"].astype(np.float32)  # not used; populated
        return policy_fn, F_goal, {"deltas_source": "recorded_centroid_delta"}

    if profile == "policy_recorded_goal":
        assert recorded_v2 is not None
        model, fs, ts = load_policy_artifacts(args.policy_dir)
        F_user = recorded_v2["y_per_ctrl"].astype(np.float32)
        shaped_log = [] if args.goal_shaping == "incremental" else None
        # Try Fix F (2-vec) first; fall back to Fix E (scalar) if dataset_v2 missing.
        md = None
        try:
            md = get_material_descriptor(args.case_name)
        except Exception:
            try:
                md = get_log_spring_Y_mean(args.case_name)
            except Exception:
                pass
        pfn = make_policy_callable(
            model, fs, ts, material,
            F_user_target=F_user, goal_shaping=args.goal_shaping,
            max_step_force=args.max_step_force, log_shaped_goals=shaped_log,
            material_descriptor=md,
        )
        return pfn, F_user, {
            "deltas_source": "policy", "policy_dir": str(args.policy_dir),
            "goal_shaping": args.goal_shaping,
            "max_step_force": args.max_step_force,
            "_shaped_log": shaped_log,
        }

    if profile == "policy_hierarchical":
        # Hierarchical Model B (sub-goal planner) + Model A (controller).
        # Goal trajectory comes from recorded y_per_ctrl (replay-style test).
        # Model B looks H_long frames ahead, predicts a sub-goal; Model A acts
        # on that sub-goal.
        assert recorded_v2 is not None
        model_a, fs_a, ts_a = load_policy_artifacts(args.policy_dir)
        if args.policy_b_dir is None:
            raise SystemExit("--policy_b_dir required for policy_hierarchical")
        model_b, fs_b, ts_b = load_policy_artifacts(args.policy_b_dir)
        F_user = recorded_v2["y_per_ctrl"].astype(np.float32)
        T_user = F_user.shape[0]

        if material not in ts_a:
            raise KeyError(f"material '{material}' not in Model A scalers")
        if material not in ts_b:
            raise KeyError(f"material '{material}' not in Model B scalers")
        ts_a_m = ts_a[material]; ts_b_m = ts_b[material]
        a_mean = ts_a_m["mean"].astype(np.float32); a_std = ts_a_m["std"].astype(np.float32)
        b_mean = ts_b_m["mean"].astype(np.float32); b_std = ts_b_m["std"].astype(np.float32)
        fa_mean = fs_a["mean"].astype(np.float32); fa_std = fs_a["std"].astype(np.float32)
        fb_mean = fs_b["mean"].astype(np.float32); fb_std = fs_b["std"].astype(np.float32)
        H_long = int(args.h_long)

        def policy_fn(state31, force_now_pad, force_goal_pad, frame_idx):
            # Long-term goal: H_long frames ahead, clamped to end of trajectory
            idx = min(frame_idx + H_long, T_user - 1)
            long_term_goal = F_user[idx]                   # [2, 3]

            # Model B forward: (state, force_now, long_term_goal) -> sub_goal
            feats_b = np.concatenate(
                [state31, force_now_pad.flatten(), long_term_goal.flatten()]
            ).astype(np.float32)
            xb = (feats_b - fb_mean) / fb_std
            with torch.no_grad():
                yb = model_b(torch.from_numpy(xb).unsqueeze(0)).numpy()[0]
            sub_goal = (yb * b_std + b_mean).reshape(2, 3).astype(np.float32)

            # Model A forward: (state, force_now, sub_goal) -> action
            feats_a = np.concatenate(
                [state31, force_now_pad.flatten(), sub_goal.flatten()]
            ).astype(np.float32)
            xa = (feats_a - fa_mean) / fa_std
            with torch.no_grad():
                ya = model_a(torch.from_numpy(xa).unsqueeze(0)).numpy()[0]
            delta = (ya * a_std + a_mean).reshape(2, 3).astype(np.float32)
            return delta

        return policy_fn, F_user, {
            "deltas_source": "policy_hierarchical",
            "policy_dir":   str(args.policy_dir),
            "policy_b_dir": str(args.policy_b_dir),
            "h_long": H_long,
        }

    if profile == "policy_hierarchical_ramp":
        # Hierarchical Model B + Model A with a synthetic ramp F_goal.
        # This is the G5 release test.
        assert recorded_v2 is not None
        model_a, fs_a, ts_a = load_policy_artifacts(args.policy_dir)
        if args.policy_b_dir is None:
            raise SystemExit("--policy_b_dir required for policy_hierarchical_ramp")
        model_b, fs_b, ts_b = load_policy_artifacts(args.policy_b_dir)

        recorded_F = recorded_v2["y_per_ctrl"].astype(np.float32)
        mag = np.linalg.norm(recorded_F, axis=-1)
        peak_mag = float(np.percentile(mag[mag > 0], 50)) if (mag > 0).any() else 1.0
        peak_mag *= args.ramp_scale
        T = recorded_F.shape[0]
        half = T // 2
        ramp_up = np.linspace(0.0, peak_mag, half, dtype=np.float32)
        ramp_down = np.linspace(peak_mag, 0.0, T - half, dtype=np.float32)
        ramp = np.concatenate([ramp_up, ramp_down])
        F_goal = np.zeros((T, 2, 3), dtype=np.float32)
        F_goal[:, 0, 0] = ramp

        if material not in ts_a or material not in ts_b:
            raise KeyError(f"material '{material}' missing from one of the scalers")
        ts_a_m = ts_a[material]; ts_b_m = ts_b[material]
        a_mean = ts_a_m["mean"].astype(np.float32); a_std = ts_a_m["std"].astype(np.float32)
        b_mean = ts_b_m["mean"].astype(np.float32); b_std = ts_b_m["std"].astype(np.float32)
        fa_mean = fs_a["mean"].astype(np.float32); fa_std = fs_a["std"].astype(np.float32)
        fb_mean = fs_b["mean"].astype(np.float32); fb_std = fs_b["std"].astype(np.float32)
        H_long = int(args.h_long)
        T_full = F_goal.shape[0]

        def policy_fn(state31, force_now_pad, force_goal_pad, frame_idx):
            idx = min(frame_idx + H_long, T_full - 1)
            long_term_goal = F_goal[idx]
            feats_b = np.concatenate(
                [state31, force_now_pad.flatten(), long_term_goal.flatten()]
            ).astype(np.float32)
            xb = (feats_b - fb_mean) / fb_std
            with torch.no_grad():
                yb = model_b(torch.from_numpy(xb).unsqueeze(0)).numpy()[0]
            sub_goal = (yb * b_std + b_mean).reshape(2, 3).astype(np.float32)
            feats_a = np.concatenate(
                [state31, force_now_pad.flatten(), sub_goal.flatten()]
            ).astype(np.float32)
            xa = (feats_a - fa_mean) / fa_std
            with torch.no_grad():
                ya = model_a(torch.from_numpy(xa).unsqueeze(0)).numpy()[0]
            return (ya * a_std + a_mean).reshape(2, 3).astype(np.float32)

        return policy_fn, F_goal, {
            "deltas_source": "policy_hierarchical_ramp",
            "policy_dir": str(args.policy_dir),
            "policy_b_dir": str(args.policy_b_dir),
            "h_long": H_long,
            "ramp_peak_N": peak_mag,
        }

    if profile == "policy_ramp":
        assert recorded_v2 is not None
        model, fs, ts = load_policy_artifacts(args.policy_dir)
        # Ramp: 0 → peak → 0 over T frames, applied per group with chosen direction.
        # Peak chosen relative to recorded case force range.
        recorded_F = recorded_v2["y_per_ctrl"].astype(np.float32)
        n_ctrl_real = int(recorded_v2.get("n_ctrl_parts", n_ctrl_parts))
        mag = np.linalg.norm(recorded_F, axis=-1)  # [T, 2]
        peak_mag = float(np.percentile(mag[mag > 0], 50)) if (mag > 0).any() else 1.0
        peak_mag *= args.ramp_scale
        T = recorded_F.shape[0]
        half = T // 2
        ramp_up = np.linspace(0.0, peak_mag, half, dtype=np.float32)
        ramp_down = np.linspace(peak_mag, 0.0, T - half, dtype=np.float32)
        ramp = np.concatenate([ramp_up, ramp_down])  # [T]
        F_goal = np.zeros((T, 2, 3), dtype=np.float32)

        # Determine direction per group.
        # 'x_axis' (default, backward-compat): +x axis (arbitrary, may not be
        #   achievable depending on rope/cloth geometry).
        # 'recorded_mean': mean direction of the recorded force per group.
        #   Physically achievable — the simulator naturally produces forces in
        #   roughly this direction when the gripper moves.
        if args.ramp_direction == "recorded_mean":
            for g in range(n_ctrl_parts):
                F_g = recorded_F[:, g, :]                          # [T, 3]
                # Weight by per-frame magnitude so noisy low-force frames don't
                # dominate the direction estimate.
                w = np.linalg.norm(F_g, axis=-1, keepdims=True)
                num = (F_g * w).sum(axis=0)
                direction = num / (np.linalg.norm(num) + 1e-9)
                F_goal[:, g, :] = ramp[:, None] * direction[None, :]
            ramp_direction_info = "recorded_mean"
        else:  # "x_axis"
            F_goal[:, 0, 0] = ramp
            ramp_direction_info = "x_axis"
        shaped_log = [] if args.goal_shaping == "incremental" else None
        # Try Fix F (2-vec) first; fall back to Fix E (scalar) if dataset_v2 missing.
        md = None
        try:
            md = get_material_descriptor(args.case_name)
        except Exception:
            try:
                md = get_log_spring_Y_mean(args.case_name)
            except Exception:
                pass
        pfn = make_policy_callable(
            model, fs, ts, material,
            F_user_target=F_goal, goal_shaping=args.goal_shaping,
            max_step_force=args.max_step_force, log_shaped_goals=shaped_log,
            material_descriptor=md,
        )
        return pfn, F_goal, {
            "deltas_source": "policy",
            "policy_dir": str(args.policy_dir),
            "ramp_peak_N": peak_mag,
            "goal_shaping": args.goal_shaping,
            "max_step_force": args.max_step_force,
            "ramp_direction": ramp_direction_info,
            "_shaped_log": shaped_log,
        }

    raise ValueError(f"unknown profile: {profile}")


# ----------------------------- main ---------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case_name", required=True)
    ap.add_argument("--profile", required=True,
                    choices=["random", "zero", "replay_action",
                             "policy_recorded_goal", "policy_ramp",
                             "policy_hierarchical", "policy_hierarchical_ramp"])
    ap.add_argument("--policy_dir", type=Path, default=DEFAULT_POLICY_DIR)
    ap.add_argument("--policy_b_dir", type=Path, default=None,
                    help="Path to Model B (sub-goal planner) seed dir; "
                         "required for hierarchical profiles")
    ap.add_argument("--h_long", type=int, default=20,
                    help="Long-term horizon for Model B in hierarchical profile")
    ap.add_argument("--dataset_v2_dir", type=Path, default=DEFAULT_DATASET_V2)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ramp_scale", type=float, default=1.0,
                    help="Multiplier on peak ramp force for policy_ramp")
    ap.add_argument("--ramp_direction", choices=["x_axis", "recorded_mean"],
                    default="x_axis",
                    help="Direction the synthetic ramp force points. 'x_axis' "
                         "(default, backward-compat) = arbitrary +x. "
                         "'recorded_mean' = mean direction of the recorded "
                         "force trajectory (physically achievable, recommended "
                         "for demo videos).")
    ap.add_argument("--goal_shaping", choices=["direct", "incremental"],
                    default="direct",
                    help="How the user's F_goal is fed to the policy. "
                         "'direct' = unchanged; 'incremental' = clip per-frame "
                         "goal step to ±max_step_force Newtons.")
    ap.add_argument("--max_step_force", type=float, default=500.0,
                    help="Per-frame goal step limit (N) for incremental shaping")
    ap.add_argument("--base_path", default="./data/different_types")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    n_ctrl_parts, cfg_type = infer_n_ctrl_parts_and_cfg_type(args.case_name)
    logger.info("case=%s n_ctrl_parts=%d cfg_type=%s",
                args.case_name, n_ctrl_parts, cfg_type)

    # Load recorded dataset_v2 case for material + recorded forces / ctrl
    v2_path = args.dataset_v2_dir / f"{args.case_name}.npz"
    recorded_v2 = None
    material = None
    if v2_path.exists():
        d = np.load(v2_path, allow_pickle=True)
        material = str(d["object_category"])
        recorded_v2 = {
            "y_per_ctrl": d["y_per_ctrl"],
            "controller_pos": d["controller_pos"],
            "positions": d["positions"],
        }
        logger.info("loaded recorded data from %s (material=%s, T=%d)",
                    v2_path.name, material, d["y_per_ctrl"].shape[0])
    else:
        logger.warning("no dataset_v2 file at %s — replay_action and "
                       "policy_* profiles will fail.", v2_path)

    # Build trainer (heavy)
    trainer, best_model_path = setup_trainer(
        args.case_name, n_ctrl_parts, cfg_type, args.base_path
    )

    # Build the chosen profile
    policy_fn, F_goal, info = build_profile(
        args.profile, args.case_name, n_ctrl_parts, material, args, recorded_v2
    )
    shaped_log = info.pop("_shaped_log", None)  # mutated by policy_fn during rollout
    info_log = dict(info)  # cleaned for printing/save
    logger.info("profile=%s info=%s F_goal.shape=%s",
                args.profile, info_log, F_goal.shape)

    # Run the rollout
    positions, forces, ctrl_pos, meta = trainer.run_policy(
        best_model_path,
        n_ctrl_parts,
        policy_fn,
        F_goal,
        compute_31d_features,
        max_frames=args.max_frames,
    )
    logger.info("rollout finished: T=%d, positions=%s, forces=%s",
                meta["frame_len"], positions.shape, forces.shape)

    # Save
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shaping_tag = "" if args.goal_shaping == "direct" else f"__shaped{int(args.max_step_force)}"
    out_path = args.out_dir / f"{args.case_name}__{args.profile}{shaping_tag}.npz"
    save_dict = dict(
        positions=positions,
        forces=forces,
        controller_pos=ctrl_pos,
        F_goal=F_goal[:meta["frame_len"]].astype(np.float32),
        case_name=args.case_name,
        material=material if material else "unknown",
        n_ctrl_parts=np.int8(n_ctrl_parts),
        profile=args.profile,
        group_ids=np.array(meta["group_ids"], dtype=np.int8),
        info=json.dumps(info_log),
        goal_shaping=args.goal_shaping,
        max_step_force=np.float32(args.max_step_force),
    )
    if shaped_log is not None and len(shaped_log) > 0:
        # Logged at policy-call time, so length = T-1 (no entry for frame 0)
        shaped_arr = np.zeros((meta["frame_len"], 2, 3), dtype=np.float32)
        shaped_arr[1:1 + len(shaped_log)] = np.stack(shaped_log)
        save_dict["F_goal_shaped"] = shaped_arr
    if recorded_v2 is not None:
        save_dict["recorded_positions"]      = recorded_v2["positions"][:meta["frame_len"]]
        save_dict["recorded_controller_pos"] = recorded_v2["controller_pos"][:meta["frame_len"]]
        save_dict["recorded_y_per_ctrl"]     = recorded_v2["y_per_ctrl"][:meta["frame_len"]]
    np.savez(out_path, **save_dict)
    logger.info("saved %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    # --- per-profile quick diagnostics (helpful for G-gate triage) ----------
    print("\n=== Rollout summary ===")
    print(f"profile:           {args.profile}")
    print(f"case:              {args.case_name}  (material={material})")
    print(f"T:                 {meta['frame_len']}")
    print(f"final positions:   ‖Δ from initial‖_inf = "
          f"{float(np.abs(positions[-1] - positions[0]).max()):.4f} m")
    print(f"final ctrl drift:  ‖ctrl[-1] - ctrl[0]‖_inf = "
          f"{float(np.abs(ctrl_pos[-1] - ctrl_pos[0]).max()):.4f} m")
    print(f"any NaN:           "
          f"pos={bool(np.isnan(positions).any())}  "
          f"force={bool(np.isnan(forces).any())}  "
          f"ctrl={bool(np.isnan(ctrl_pos).any())}")
    if recorded_v2 is not None:
        T_eval = meta["frame_len"]
        rec_pos = recorded_v2["positions"][:T_eval]
        pos_err = float(np.abs(positions - rec_pos).max())
        rec_F = recorded_v2["y_per_ctrl"][:T_eval, :n_ctrl_parts]  # [T, n, 3]
        force_err = float(np.linalg.norm(forces - rec_F, axis=-1).mean())
        rec_F_norm = float(np.linalg.norm(rec_F, axis=-1).mean())
        print(f"vs recorded:       max |pos err| = {pos_err:.5f} m, "
              f"mean |force err| = {force_err:.2f} N (recorded mean ‖F‖ = "
              f"{rec_F_norm:.2f} N, ratio = {force_err / max(rec_F_norm, 1e-6):.3f})")


if __name__ == "__main__":
    main()
