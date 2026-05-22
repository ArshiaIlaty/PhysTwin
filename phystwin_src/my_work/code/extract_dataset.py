"""extract_dataset.py
Drives PhysTwin's force/position inference across a list of cases and saves
per-case (summary_features, control_point_force) datasets to ./dataset/*.npz.

Run as:
    python extract_dataset.py                  # all default cases
    python extract_dataset.py --case_name foo  # one case
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg, logger

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# Cases we want to extract. The 3rd column ("real" vs "cloth") picks the cfg yaml.
# n_ctrl_parts is inferred from the case_name prefix (single_ -> 1, double_ -> 2;
# overridden here when the heuristic doesn't match).
DEFAULT_CASES: list[tuple[str, int, str]] = [
    # rope / twine — all use n_ctrl_parts = 1 except rope_double_hand
    ("single_push_rope_1", 1, "real"),
    ("single_push_rope",   1, "real"),
    ("single_push_rope_4", 1, "real"),
    ("single_lift_rope",   1, "real"),
    ("rope_double_hand",   2, "real"),
    # cloth
    ("single_clift_cloth_1", 1, "cloth"),
    ("single_clift_cloth_3", 1, "cloth"),
    ("single_lift_cloth",    1, "cloth"),
    ("single_lift_cloth_1",  1, "cloth"),
    ("single_lift_cloth_3",  1, "cloth"),
    ("single_lift_cloth_4",  1, "cloth"),
    ("double_lift_cloth_1",  2, "cloth"),
    ("double_lift_cloth_3",  2, "cloth"),
    # sloth (stuffed animal)
    ("double_stretch_sloth", 2, "real"),
    ("double_lift_sloth",    2, "real"),
    ("single_lift_sloth",    1, "real"),
    ("single_push_sloth",    1, "real"),
    # toy (zebra + dinosaur) — uses real.yaml
    ("double_lift_zebra",    2, "real"),
    ("double_stretch_zebra", 2, "real"),
    ("single_lift_zebra",    1, "real"),
    ("single_lift_dinosor",  1, "real"),
    # NOTE: weird_package (~39 frames, lone "package" case) is intentionally excluded.
    # It's a singleton with no peer trajectories for material-level evaluation.
]


def summary_features(positions: np.ndarray) -> np.ndarray:
    """Compute per-timestep fixed-dim summary features from particle positions.

    Args:
        positions: [T, N, 3] float array.

    Returns:
        features: [T, D] float array, D = 18.
    """
    T = positions.shape[0]
    rest = positions[0:1]                              # [1, N, 3]
    disp = positions - rest                            # [T, N, 3]
    abs_disp = np.abs(disp)
    disp_mag = np.linalg.norm(disp, axis=-1)           # [T, N]

    pos_min = positions.min(axis=1)                    # [T, 3]
    pos_max = positions.max(axis=1)                    # [T, 3]
    rest_dim = (pos_max[0:1] - pos_min[0:1])           # [1, 3]
    bbox_dim = (pos_max - pos_min) - rest_dim          # [T, 3]

    centroid_disp = disp.mean(axis=1)                  # [T, 3]
    max_abs = abs_disp.max(axis=1)                     # [T, 3]
    mean_abs = abs_disp.mean(axis=1)                   # [T, 3]
    std_disp = disp.std(axis=1)                        # [T, 3]
    ke_proxy = (disp ** 2).sum(axis=(1, 2)) / max(positions.shape[1], 1)  # [T]
    mean_mag = disp_mag.mean(axis=1)                   # [T]
    max_mag = disp_mag.max(axis=1)                     # [T]

    feats = np.concatenate(
        [
            centroid_disp,                              # 3
            bbox_dim,                                   # 3
            max_abs,                                    # 3
            mean_abs,                                   # 3
            std_disp,                                   # 3
            ke_proxy[:, None],                          # 1
            mean_mag[:, None],                          # 1
            max_mag[:, None],                           # 1
        ],
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


def pad_forces(forces: np.ndarray, max_ctrl_parts: int = 2) -> np.ndarray:
    """Zero-pad force tensor along the control-part axis.

    forces: [T, n_ctrl, 3]  -> [T, max_ctrl_parts, 3]
    """
    T, n, _ = forces.shape
    padded = np.zeros((T, max_ctrl_parts, 3), dtype=forces.dtype)
    padded[:, :n] = forces
    return padded


def extract_case(
    case_name: str,
    n_ctrl_parts: int,
    cfg_type: str,
    base_path: str,
    out_dir: str,
) -> dict | None:
    """Run extraction for one case. Returns metadata dict (or None on failure)."""
    cfg_file = (
        "configs/cloth.yaml" if cfg_type == "cloth" else "configs/real.yaml"
    )
    cfg.load_from_yaml(cfg_file)
    base_dir = f"./experiments/{case_name}"

    optimal_path = f"./experiments_optimization/{case_name}/optimal_params.pkl"
    if not os.path.exists(optimal_path):
        logger.warning(f"[{case_name}] missing optimal_params.pkl, skipping.")
        return None
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

    calib_path = f"{base_path}/{case_name}/calibrate.pkl"
    if not os.path.exists(calib_path):
        logger.warning(f"[{case_name}] missing calibrate.pkl, skipping.")
        return None
    with open(calib_path, "rb") as f:
        c2ws = pickle.load(f)
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array([np.linalg.inv(c) for c in c2ws])

    meta_path = f"{base_path}/{case_name}/metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    cfg.intrinsics = np.array(meta["intrinsics"])
    cfg.WH = meta["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"

    best_models = glob.glob(f"experiments/{case_name}/train/best_*.pth")
    if not best_models:
        logger.warning(f"[{case_name}] no trained model found, skipping.")
        return None
    best_model_path = best_models[0]

    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )

    positions, forces, controller_pos, sim_meta = trainer.extract_force_data(
        best_model_path, n_ctrl_parts=n_ctrl_parts
    )

    X = summary_features(positions)             # [T, 18]
    y_per_ctrl = pad_forces(forces, max_ctrl_parts=2)  # [T, 2, 3]
    y_net = forces.sum(axis=1)                  # [T, 3]

    out_path = Path(out_dir) / f"{case_name}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        X=X,
        y_per_ctrl=y_per_ctrl,
        y_net=y_net,
        positions=positions,                    # raw, useful for re-feature later
        controller_pos=controller_pos,
        case_name=case_name,
        material=cfg_type,                      # "real" or "cloth"
        object_category=_object_category(case_name),
        n_ctrl_parts=n_ctrl_parts,
        feature_names=np.array(SUMMARY_FEATURE_NAMES),
    )
    logger.info(
        f"[{case_name}] saved X={X.shape} y_net={y_net.shape} y_per_ctrl={y_per_ctrl.shape}"
        f" frame_len={sim_meta['frame_len']} -> {out_path}"
    )
    return {
        "case_name": case_name,
        "frame_len": sim_meta["frame_len"],
        "num_all_points": sim_meta["num_all_points"],
        "n_ctrl_parts": n_ctrl_parts,
        "out_path": str(out_path),
    }


def _object_category(case_name: str) -> str:
    cn = case_name.lower()
    if "rope" in cn or "twine" in cn:
        return "rope"
    if "cloth" in cn or "package" in cn:
        return "cloth"
    if "sloth" in cn:
        return "sloth"
    if "zebra" in cn or "dinosor" in cn or "toy" in cn:
        return "toy"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--out_dir", type=str, default=str(_RESULTS_DIR / "dataset"))
    parser.add_argument(
        "--case_name",
        type=str,
        default=None,
        help="Run on a single case (must also pass --n_ctrl_parts and --cfg_type).",
    )
    parser.add_argument("--n_ctrl_parts", type=int, default=None)
    parser.add_argument("--cfg_type", type=str, choices=["real", "cloth"], default=None)
    args = parser.parse_args()

    if args.case_name is not None:
        if args.n_ctrl_parts is None or args.cfg_type is None:
            raise SystemExit(
                "When --case_name is given you must also pass --n_ctrl_parts and --cfg_type"
            )
        cases = [(args.case_name, args.n_ctrl_parts, args.cfg_type)]
    else:
        cases = DEFAULT_CASES

    summary = []
    for case_name, n_ctrl_parts, cfg_type in cases:
        try:
            m = extract_case(case_name, n_ctrl_parts, cfg_type, args.base_path, args.out_dir)
            if m is not None:
                summary.append(m)
        except Exception as e:
            logger.exception(f"[{case_name}] FAILED: {e}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open(Path(args.out_dir) / "_extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"DONE. {len(summary)} cases extracted -> {args.out_dir}")


if __name__ == "__main__":
    main()
