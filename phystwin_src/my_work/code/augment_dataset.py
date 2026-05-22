"""augment_dataset.py — Tier 1.1 + Tier 1.2 feature extension.

Reads `dataset/*.npz` (originals: 18 features), computes 13 new features per
frame from `positions` and `controller_pos`, and writes `dataset_v2/*.npz`.

New features (per frame, all relative to rest = frame 0):
  Controller-aware (7):
    ctrl_centroid_disp_x,y,z   — controller centroid translation
    nearest_dist               — min distance from controller centroid to any
                                 object particle (contact proxy)
    mean_contact_dist          — mean distance ctrl→object particles
    rel_motion_x,y,z           — controller centroid − object centroid
  Velocity (6):
    mean_vel_mag               — mean ‖vel‖ over particles, per-frame
    max_vel_mag                — max ‖vel‖ over particles
    centroid_vel_x,y,z         — velocity of object centroid

So `X` goes from [T, 18] → [T, 31]. Targets and everything else are unchanged.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np


_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SRC_DIR = str(_RESULTS_DIR / "dataset")
DST_DIR = str(_RESULTS_DIR / "dataset_v2")


NEW_FEATURE_NAMES = [
    "ctrl_centroid_disp_x", "ctrl_centroid_disp_y", "ctrl_centroid_disp_z",
    "nearest_dist", "mean_contact_dist",
    "rel_motion_x", "rel_motion_y", "rel_motion_z",
    "mean_vel_mag", "max_vel_mag",
    "centroid_vel_x", "centroid_vel_y", "centroid_vel_z",
]


def augment_case(npz_path: str, out_path: str) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    X_old = d["X"].astype(np.float32)              # [T, 18]
    positions = d["positions"].astype(np.float32)  # [T, N, 3]
    ctrl = d["controller_pos"].astype(np.float32)  # [T, K, 3]

    T, N, _ = positions.shape

    # --- controller features ---
    ctrl_centroid = ctrl.mean(axis=1)              # [T, 3]
    ctrl_centroid_disp = ctrl_centroid - ctrl_centroid[0:1]   # [T, 3]
    obj_centroid = positions.mean(axis=1)          # [T, 3]
    rel_motion = ctrl_centroid - obj_centroid      # [T, 3]
    # Distance ctrl_centroid → object particles
    diffs = positions - ctrl_centroid[:, None, :]  # [T, N, 3]
    obj_to_ctrl = np.linalg.norm(diffs, axis=-1)   # [T, N]
    nearest_dist = obj_to_ctrl.min(axis=1, keepdims=True)         # [T, 1]
    mean_contact_dist = obj_to_ctrl.mean(axis=1, keepdims=True)   # [T, 1]

    # --- velocity features (forward diff; frame 0 → zero velocity) ---
    vel = np.zeros_like(positions)
    vel[1:] = positions[1:] - positions[:-1]       # [T, N, 3]
    vel_mag = np.linalg.norm(vel, axis=-1)         # [T, N]
    mean_vel_mag = vel_mag.mean(axis=1, keepdims=True)    # [T, 1]
    max_vel_mag = vel_mag.max(axis=1, keepdims=True)      # [T, 1]
    centroid_vel = vel.mean(axis=1)                # [T, 3]

    extra = np.concatenate([
        ctrl_centroid_disp,                        # 3
        nearest_dist, mean_contact_dist,           # 2
        rel_motion,                                # 3
        mean_vel_mag, max_vel_mag,                 # 2
        centroid_vel,                              # 3
    ], axis=1).astype(np.float32)                  # [T, 13]
    assert extra.shape == (T, 13), extra.shape

    X_new = np.concatenate([X_old, extra], axis=1) # [T, 31]
    feat_names = list(d["feature_names"]) + NEW_FEATURE_NAMES
    assert len(feat_names) == 31

    np.savez(
        out_path,
        X=X_new,
        y_per_ctrl=d["y_per_ctrl"],
        y_net=d["y_net"],
        positions=d["positions"],
        controller_pos=d["controller_pos"],
        case_name=d["case_name"],
        material=d["material"],
        object_category=d["object_category"],
        n_ctrl_parts=d["n_ctrl_parts"],
        feature_names=np.array(feat_names),
    )
    return {"case_name": str(d["case_name"]), "T": T, "N": N, "X_shape": X_new.shape}


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC_DIR, "*.npz")))
    files = [f for f in files if not Path(f).name.startswith("_")]
    print(f"Augmenting {len(files)} cases: {SRC_DIR}/ -> {DST_DIR}/ (X: 18 -> 31 features)")
    for f in files:
        out = os.path.join(DST_DIR, Path(f).name)
        info = augment_case(f, out)
        print(f"  {info['case_name']:30s} T={info['T']:>4d}  X={info['X_shape']}")


if __name__ == "__main__":
    main()
