"""features.py — shared 18 + 13 = 31-D feature computation.

The 31-dim feature vector is the policy's input state representation.
Originally implemented inline in two places:

  - 18-D base summary features: extract_dataset.py:summary_features()
  - 13-D controller + velocity features: augment_dataset.py:augment_case()
    (inline, no standalone function)

To make sure the closed-loop driver (Step 3) sees the SAME features the
policy was trained on, we factor both pieces into one module and call it
from anything that needs features.

Run as a script to verify against a stored dataset_v2 case:

    python my_work/code/features.py
"""

from __future__ import annotations

import numpy as np


# -- 18-D base summary features (mirror extract_dataset.py:summary_features) --

def summary_features(positions: np.ndarray) -> np.ndarray:
    """positions: [T, N, 3]; returns [T, 18] float32."""
    T = positions.shape[0]
    rest = positions[0:1]
    disp = positions - rest
    abs_disp = np.abs(disp)
    disp_mag = np.linalg.norm(disp, axis=-1)

    pos_min = positions.min(axis=1)
    pos_max = positions.max(axis=1)
    rest_dim = (pos_max[0:1] - pos_min[0:1])
    bbox_dim = (pos_max - pos_min) - rest_dim

    centroid_disp = disp.mean(axis=1)
    max_abs = abs_disp.max(axis=1)
    mean_abs = abs_disp.mean(axis=1)
    std_disp = disp.std(axis=1)
    ke_proxy = (disp ** 2).sum(axis=(1, 2)) / max(positions.shape[1], 1)
    mean_mag = disp_mag.mean(axis=1)
    max_mag = disp_mag.max(axis=1)

    feats = np.concatenate(
        [centroid_disp, bbox_dim, max_abs, mean_abs, std_disp,
         ke_proxy[:, None], mean_mag[:, None], max_mag[:, None]],
        axis=1,
    )
    assert feats.shape == (T, 18), feats.shape
    return feats.astype(np.float32)


SUMMARY_FEATURE_NAMES = (
    [f"centroid_disp_{a}"   for a in "xyz"]
    + [f"bbox_stretch_{a}"  for a in "xyz"]
    + [f"max_abs_{a}"       for a in "xyz"]
    + [f"mean_abs_{a}"      for a in "xyz"]
    + [f"std_disp_{a}"      for a in "xyz"]
    + ["ke_proxy", "mean_disp_mag", "max_disp_mag"]
)


# -- 13-D controller + velocity features (mirror augment_dataset.py inline) --

def controller_velocity_features(positions: np.ndarray,
                                  ctrl: np.ndarray) -> np.ndarray:
    """positions: [T, N, 3]; ctrl: [T, K, 3]; returns [T, 13] float32."""
    T, N, _ = positions.shape

    ctrl_centroid = ctrl.mean(axis=1)
    ctrl_centroid_disp = ctrl_centroid - ctrl_centroid[0:1]
    obj_centroid = positions.mean(axis=1)
    rel_motion = ctrl_centroid - obj_centroid

    diffs = positions - ctrl_centroid[:, None, :]
    obj_to_ctrl = np.linalg.norm(diffs, axis=-1)
    nearest_dist = obj_to_ctrl.min(axis=1, keepdims=True)
    mean_contact_dist = obj_to_ctrl.mean(axis=1, keepdims=True)

    vel = np.zeros_like(positions)
    vel[1:] = positions[1:] - positions[:-1]
    vel_mag = np.linalg.norm(vel, axis=-1)
    mean_vel_mag = vel_mag.mean(axis=1, keepdims=True)
    max_vel_mag = vel_mag.max(axis=1, keepdims=True)
    centroid_vel = vel.mean(axis=1)

    extra = np.concatenate(
        [ctrl_centroid_disp, nearest_dist, mean_contact_dist, rel_motion,
         mean_vel_mag, max_vel_mag, centroid_vel],
        axis=1,
    ).astype(np.float32)
    assert extra.shape == (T, 13), extra.shape
    return extra


NEW_FEATURE_NAMES = [
    "ctrl_centroid_disp_x", "ctrl_centroid_disp_y", "ctrl_centroid_disp_z",
    "nearest_dist", "mean_contact_dist",
    "rel_motion_x", "rel_motion_y", "rel_motion_z",
    "mean_vel_mag", "max_vel_mag",
    "centroid_vel_x", "centroid_vel_y", "centroid_vel_z",
]
ALL_FEATURE_NAMES = SUMMARY_FEATURE_NAMES + NEW_FEATURE_NAMES


def compute_31d_features(positions: np.ndarray,
                          ctrl: np.ndarray) -> np.ndarray:
    """positions: [T, N, 3]; ctrl: [T, K, 3]; returns [T, 31] float32."""
    base = summary_features(positions)
    extra = controller_velocity_features(positions, ctrl)
    out = np.concatenate([base, extra], axis=1).astype(np.float32)
    assert out.shape[1] == 31, out.shape
    return out


# ----- material descriptor (Fix E experiment) ---------------------------

def get_log_spring_Y_mean(case_name: str, base_dir: str = ".") -> float:
    """Return mean log spring stiffness for a case (per-case constant)."""
    import torch
    from pathlib import Path
    paths = sorted(Path(base_dir, "experiments", case_name, "train").glob("best_*.pth"))
    if not paths:
        raise FileNotFoundError(f"No best_*.pth for {case_name}")
    ckpt = torch.load(paths[0], map_location="cpu", weights_only=True)
    spring_Y = ckpt["spring_Y"]
    return float(torch.log(spring_Y).mean().item())


def get_log_mean_force(case_name: str, dataset_v2_dir=None) -> float:
    """Return log of mean force magnitude across active groups in the case's
    recorded data. Captures 'typical force scale' for this case.

    Distinguishes cases like single_push_rope_4 (gentle, peak ~2.5kN) from
    typical rope cases (~15kN) — useful where the material descriptor alone
    can't differentiate.
    """
    from pathlib import Path
    if dataset_v2_dir is None:
        dataset_v2_dir = Path(__file__).resolve().parent.parent / "results" / "dataset_v2"
    p = Path(dataset_v2_dir) / f"{case_name}.npz"
    if not p.exists():
        raise FileNotFoundError(f"No dataset_v2 npz for {case_name}")
    d = np.load(p, allow_pickle=True)
    y_per_ctrl = d["y_per_ctrl"]
    n_ctrl = int(d["n_ctrl_parts"])
    mag = np.linalg.norm(y_per_ctrl[:, :n_ctrl], axis=-1)
    return float(np.log(float(mag.mean()) + 1e-6))


def get_material_descriptor(case_name: str, dataset_v2_dir=None) -> np.ndarray:
    """2-vec per-case descriptor: [log_mean_spring_Y, log_mean_force].

    Fix F: extends Fix E by adding the per-case "typical force" alongside
    "material stiffness," so the policy can distinguish gentle cases from
    aggressive ones independently of material stiffness.
    """
    return np.array([
        get_log_spring_Y_mean(case_name),
        get_log_mean_force(case_name, dataset_v2_dir),
    ], dtype=np.float32)


# ----- learned material encoder (Fix H) ----------------------------------

# Quantile probabilities for the stiffness-distribution summary. The mean +
# std + these three quantiles give a 5-D leakage-free descriptor of the
# per-spring log(spring_Y) distribution (vs Fix F's single .mean()).
_STIFFNESS_QUANTILES = (0.1, 0.5, 0.9)
STIFFNESS_STATS_DIM = 5            # [mean, std, q10, q50, q90]
STIFFNESS_STATS_DIM_WITH_FORCE = 6  # + log_mean_force (leakage ablation only)


def get_stiffness_stats(case_name: str,
                        base_dir: str = ".",
                        include_force_scale: bool = False,
                        dataset_v2_dir=None) -> np.ndarray:
    """Per-case summary of the per-spring log(spring_Y) distribution.

    Returns [mean, std, q10, q50, q90] (float32) — the input to the Fix H
    material encoder. Unlike Fix F's `get_material_descriptor`, this keeps the
    *distributional shape* of stiffness (not just the mean) and is **stiffness
    only**: no demo-force term, so nothing here leaks the replay goal.

    PhysTwin fits a separate twin (separate spring_Y field) per case, so this
    varies per case, but it is a per-case constant — identical for every frame
    and every synthetic trajectory derived from the case. It is calibration
    data available at deployment, analogous to a material's Young's modulus.

    include_force_scale=True appends `get_log_mean_force` for the explicit
    leakage ablation only — NOT for the headline encoder.
    """
    import torch
    from pathlib import Path
    paths = sorted(Path(base_dir, "experiments", case_name, "train").glob("best_*.pth"))
    if not paths:
        raise FileNotFoundError(f"No best_*.pth for {case_name}")
    ckpt = torch.load(paths[0], map_location="cpu", weights_only=True)
    log_Y = torch.log(ckpt["spring_Y"]).flatten().numpy().astype(np.float64)
    quantiles = np.quantile(log_Y, _STIFFNESS_QUANTILES)
    stats = [float(log_Y.mean()), float(log_Y.std())] + [float(q) for q in quantiles]
    if include_force_scale:
        stats.append(get_log_mean_force(case_name, dataset_v2_dir))
    return np.asarray(stats, dtype=np.float32)


MATERIAL_STATS_DIM = 11  # see get_material_stats


def get_material_stats(case_name: str, base_dir: str = ".") -> np.ndarray:
    """Fix K — enriched per-case material description (11-D, leakage-free).

    Extends get_stiffness_stats with everything else PhysTwin's twin
    calibration recovers, all per-case constants:

      [0:5]  OBJECT-spring log(spring_Y): mean, std, q10, q50, q90
      [5:7]  CONTROL-spring log(spring_Y): mean, std
             (the control springs connect gripper to object — their stiffness
             IS the gripper-displacement→force gain, i.e. the inverse-dynamics
             gain the policy must implicitly know; pooling them with object
             springs, as get_stiffness_stats does, washes this out)
      [7:11] collide_elas, collide_fric, collide_object_elas,
             collide_object_fric (calibrated ground/object collision physics)
    """
    import torch
    from pathlib import Path
    paths = sorted(Path(base_dir, "experiments", case_name, "train").glob("best_*.pth"))
    if not paths:
        raise FileNotFoundError(f"No best_*.pth for {case_name}")
    ckpt = torch.load(paths[0], map_location="cpu", weights_only=True)
    log_Y = torch.log(ckpt["spring_Y"]).flatten().numpy().astype(np.float64)
    n_obj = int(ckpt["num_object_springs"])
    obj, ctl = log_Y[:n_obj], log_Y[n_obj:]
    q = np.quantile(obj, _STIFFNESS_QUANTILES)
    stats = [float(obj.mean()), float(obj.std()),
             float(q[0]), float(q[1]), float(q[2]),
             float(ctl.mean()) if len(ctl) else float(obj.mean()),
             float(ctl.std()) if len(ctl) else 0.0,
             float(ckpt["collide_elas"].item()),
             float(ckpt["collide_fric"].item()),
             float(ckpt["collide_object_elas"].item()),
             float(ckpt["collide_object_fric"].item())]
    out = np.asarray(stats, dtype=np.float32)
    assert out.shape[0] == MATERIAL_STATS_DIM
    return out


# -------------------- self-test ------------------------------------------

def _self_test():
    import glob
    import sys
    from pathlib import Path

    results = Path(__file__).resolve().parent.parent / "results" / "dataset_v2"
    files = sorted(glob.glob(str(results / "*.npz")))
    if not files:
        print("self-test SKIP: no dataset_v2 files at", results)
        return
    ok = True
    for path in files[:3]:
        d = np.load(path, allow_pickle=True)
        if "positions" not in d.files:
            continue
        positions = d["positions"]
        ctrl = d["controller_pos"]
        X_stored = d["X"]
        X_computed = compute_31d_features(positions, ctrl)
        if X_stored.shape != X_computed.shape:
            print(f"FAIL {Path(path).name}: shape {X_stored.shape} vs {X_computed.shape}")
            ok = False
            continue
        diff = float(np.abs(X_stored - X_computed).max())
        status = "PASS" if diff < 1e-5 else "FAIL"
        print(f"{status} {Path(path).name}: max |diff| = {diff:.2e}")
        if diff >= 1e-5:
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _self_test()
