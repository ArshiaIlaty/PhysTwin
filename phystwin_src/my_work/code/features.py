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
