#!/usr/bin/env python3
"""
build_policy_dataset.py

Convert per-trajectory npz files (real + synthetic) into a single flat
(state, force_now, force_goal, action) table for closed-loop policy training.

Reads:  my_work/results/dataset_v2/*.npz  (real)
        my_work/results/dataset_synth_raw/*.npz  (synthetic)
Writes: my_work/results/policy_dataset.npz
        my_work/results/policy_dataset_summary.json

Plan: my_work/docs/closed_loop_control/step1_plan.md
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger("build_policy_dataset")


SCRIPT_DIR = Path(__file__).resolve().parent
MY_WORK = SCRIPT_DIR.parent
RESULTS = MY_WORK / "results"

DEFAULT_DATASET_V2 = RESULTS / "dataset_v2"
DEFAULT_SYNTH = RESULTS / "dataset_synth_raw"
DEFAULT_OUT = RESULTS / "policy_dataset.npz"
DEFAULT_SUMMARY = RESULTS / "policy_dataset_summary.json"
DEFAULT_EXCLUDE = ["single_push_sloth"]


def load_trajectory(npz_path: Path, source: str):
    """Load one trajectory npz. Returns dict, or None if unusable."""
    d = np.load(npz_path, allow_pickle=True)
    needed = {"X", "y_per_ctrl", "controller_pos", "n_ctrl_parts"}
    missing = needed - set(d.files)
    if missing:
        logger.warning("skipping %s: missing fields %s", npz_path.name, missing)
        return None
    T = d["X"].shape[0]
    if T < 2:
        logger.warning("skipping %s: T=%d (need >=2)", npz_path.name, T)
        return None
    motion_type = str(d["motion_type"]) if "motion_type" in d.files else "real"
    # `material` in the npz is actually cfg_type ("real" or "cloth"); the
    # real material taxonomy (rope/cloth/sloth/toy) lives in object_category.
    material = (str(d["object_category"]) if "object_category" in d.files
                else (str(d["material"]) if "material" in d.files else "unknown"))
    case_name = str(d["case_name"]) if "case_name" in d.files else npz_path.stem
    return {
        "X": d["X"].astype(np.float32),
        "y_per_ctrl": d["y_per_ctrl"].astype(np.float32),
        "controller_pos": d["controller_pos"].astype(np.float32),
        "n_ctrl_parts": int(d["n_ctrl_parts"]),
        "material": material,
        "case_name": case_name,
        "motion_type": motion_type,
        "source": source,
    }


def compute_group_ids(controller_pos_t0: np.ndarray, n_ctrl_parts: int) -> np.ndarray:
    """KMeans on frame 0 controller points; returns [K] int8."""
    K = controller_pos_t0.shape[0]
    if n_ctrl_parts == 1:
        return np.zeros(K, dtype=np.int8)
    km = KMeans(n_clusters=n_ctrl_parts, n_init=10, random_state=0).fit(controller_pos_t0)
    return km.labels_.astype(np.int8)


def _pad_y_per_ctrl(y: np.ndarray, T: int) -> np.ndarray:
    """Defensive: ensure y_per_ctrl is [T, 2, 3]."""
    if y.ndim == 2:
        y = y.reshape(T, 1, 3)
    if y.shape[1] < 2:
        pad = np.zeros((T, 2 - y.shape[1], 3), dtype=y.dtype)
        y = np.concatenate([y, pad], axis=1)
    return y


def trajectory_to_rows(traj: dict):
    """Convert one trajectory dict into per-frame-pair row arrays.

    Returns (rows_dict, group_ids, group_separation_or_None) or None on failure.
    """
    X = traj["X"]
    ctrl = traj["controller_pos"]
    n_ctrl_parts = traj["n_ctrl_parts"]
    T, K, _ = ctrl.shape
    y = _pad_y_per_ctrl(traj["y_per_ctrl"], T)

    group_ids = compute_group_ids(ctrl[0], n_ctrl_parts)

    centroids = np.zeros((T, 2, 3), dtype=np.float32)
    for g in range(n_ctrl_parts):
        mask = group_ids == g
        if mask.sum() == 0:
            logger.warning(
                "case=%s: group %d has 0 controller points (degenerate KMeans)",
                traj["case_name"], g,
            )
            return None
        centroids[:, g] = ctrl[:, mask].mean(axis=1)

    group_sep = None
    if n_ctrl_parts == 2:
        group_sep = float(np.linalg.norm(centroids[0, 0] - centroids[0, 1]))

    T_rows = T - 1
    rows = {
        "state":       X[:T_rows].astype(np.float32),
        "force_now":   y[:T_rows].astype(np.float32),
        "force_goal":  y[1:].astype(np.float32),
        "action":      (centroids[1:] - centroids[:T_rows]).astype(np.float32),
        "action_mask": np.array(
            [[1.0] * n_ctrl_parts + [0.0] * (2 - n_ctrl_parts)] * T_rows,
            dtype=np.float32,
        ),
        "material":     np.array([traj["material"]]    * T_rows, dtype="<U16"),
        "case_name":    np.array([traj["case_name"]]   * T_rows, dtype="<U64"),
        "n_ctrl_parts": np.full(T_rows, n_ctrl_parts, dtype=np.int8),
        "source":       np.array([traj["source"]]      * T_rows, dtype="<U10"),
        "motion_type":  np.array([traj["motion_type"]] * T_rows, dtype="<U16"),
    }
    return rows, group_ids, group_sep


def percentile_stats(x, ps=(5, 50, 95, 99, 99.9)):
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}


def validate(combined: dict, per_traj_meta: list) -> dict:
    """Run validation checks. Hard checks raise; soft checks report + warn."""
    summary = {}
    warnings_list = []
    R = combined["state"].shape[0]

    # 1. row count
    expected_R = sum(m["T_rows"] for m in per_traj_meta)
    summary["row_count"] = R
    summary["expected_row_count"] = expected_R
    summary["num_trajectories"] = len(per_traj_meta)
    if R != expected_R:
        raise AssertionError(f"row count: got {R}, expected {expected_R}")

    # 2. per-material rows
    mat = combined["material"]
    summary["per_material_rows"] = {
        m: int((mat == m).sum()) for m in sorted(np.unique(mat).tolist())
    }

    # 3. action magnitude percentiles per material (warn if p99 > 0.1)
    action_mag = np.linalg.norm(combined["action"], axis=-1)  # [R, 2]
    mask = combined["action_mask"]
    summary["action_mag_per_material"] = {}
    for m in sorted(np.unique(mat).tolist()):
        rows_m = mat == m
        vals = action_mag[rows_m][mask[rows_m] > 0]
        if len(vals):
            stats = percentile_stats(vals)
            summary["action_mag_per_material"][m] = stats
            if stats["p99"] > 0.1:
                msg = (f"material={m}: action p99 = {stats['p99']:.4f} m/frame "
                       "exceeds 0.1 m sanity bound")
                logger.warning(msg)
                warnings_list.append(msg)

    # 4. force residual percentiles per material
    fres = np.linalg.norm(combined["force_goal"] - combined["force_now"], axis=-1)
    summary["force_residual_per_material"] = {}
    for m in sorted(np.unique(mat).tolist()):
        rows_m = mat == m
        vals = fres[rows_m][mask[rows_m] > 0]
        if len(vals):
            summary["force_residual_per_material"][m] = percentile_stats(
                vals, (50, 95, 99)
            )

    # 5. group separation for double-control cases
    double_meta = [m for m in per_traj_meta if m["n_ctrl_parts"] == 2
                   and m["group_separation"] is not None]
    if double_meta:
        seps = {m["case_name"]: float(m["group_separation"]) for m in double_meta}
        summary["group_separation_double_ctrl"] = seps
        min_sep = min(seps.values())
        summary["min_group_separation"] = min_sep
        if min_sep < 0.01:
            msg = f"min double-control group separation = {min_sep:.4f} m"
            logger.warning(msg)
            warnings_list.append(msg)

    # 6. no NaN / Inf (hard)
    for field in ("state", "force_now", "force_goal", "action"):
        arr = combined[field]
        n_bad = int(np.sum(~np.isfinite(arr)))
        if n_bad:
            raise AssertionError(f"NaN/Inf in {field}: {n_bad} entries")
    summary["nan_inf_check"] = "passed"

    # 7. mask coverage exact match (hard)
    rows_with_g1 = int((combined["action_mask"][:, 1] == 1.0).sum())
    rows_without_g1 = R - rows_with_g1
    expected_with_g1 = sum(m["T_rows"] for m in per_traj_meta if m["n_ctrl_parts"] == 2)
    expected_without_g1 = sum(m["T_rows"] for m in per_traj_meta if m["n_ctrl_parts"] == 1)
    if rows_with_g1 != expected_with_g1:
        raise AssertionError(
            f"mask coverage (double-ctrl): got {rows_with_g1}, expected {expected_with_g1}"
        )
    if rows_without_g1 != expected_without_g1:
        raise AssertionError(
            f"mask coverage (single-ctrl): got {rows_without_g1}, expected {expected_without_g1}"
        )
    summary["mask_coverage"] = {
        "double_ctrl_rows": rows_with_g1,
        "single_ctrl_rows": rows_without_g1,
    }

    # 8. motion-type coverage
    mt = combined["motion_type"]
    summary["per_motion_type_rows"] = {
        t: int((mt == t).sum()) for t in sorted(np.unique(mt).tolist())
    }

    summary["warnings"] = warnings_list
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_v2_dir", type=Path, default=DEFAULT_DATASET_V2)
    parser.add_argument("--synth_dir", type=Path, default=DEFAULT_SYNTH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE),
                        help="case_name prefixes to exclude (startswith match)")
    parser.add_argument("--no_synth", action="store_true",
                        help="skip dataset_synth_raw")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    real_files = sorted(args.dataset_v2_dir.glob("*.npz")) if args.dataset_v2_dir.exists() else []
    synth_files = (sorted(args.synth_dir.glob("*.npz"))
                   if (not args.no_synth and args.synth_dir.exists()) else [])
    logger.info("found %d real, %d synthetic npz files", len(real_files), len(synth_files))

    all_rows = []
    per_traj_meta = []
    group_ids_per_case = []
    case_to_idx = {}

    def process(files, source):
        for path in files:
            traj = load_trajectory(path, source)
            if traj is None:
                continue
            if any(traj["case_name"].startswith(p) for p in args.exclude):
                logger.info("excluding %s", traj["case_name"])
                continue
            result = trajectory_to_rows(traj)
            if result is None:
                continue
            rows, gids, gsep = result
            T_rows = rows["state"].shape[0]
            case_key = traj["case_name"]
            if case_key not in case_to_idx:
                case_to_idx[case_key] = len(group_ids_per_case)
                group_ids_per_case.append(gids)
            rows["case_idx"] = np.full(T_rows, case_to_idx[case_key], dtype=np.int32)
            all_rows.append(rows)
            per_traj_meta.append({
                "case_name": case_key,
                "n_ctrl_parts": traj["n_ctrl_parts"],
                "T_rows": T_rows,
                "group_separation": gsep,
                "source": source,
            })

    process(real_files, "real")
    process(synth_files, "synth")

    if not all_rows:
        logger.error("no trajectories processed; aborting")
        sys.exit(1)

    keys = list(all_rows[0].keys())
    combined = {k: np.concatenate([r[k] for r in all_rows], axis=0) for k in keys}
    logger.info("combined %d trajectories → %d rows", len(all_rows), combined["state"].shape[0])

    summary = validate(combined, per_traj_meta)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    group_ids_array = np.empty(len(group_ids_per_case), dtype=object)
    for i, gids in enumerate(group_ids_per_case):
        group_ids_array[i] = gids
    np.savez(args.out, **combined, group_ids=group_ids_array)
    logger.info("saved %s (%.1f MB)", args.out, args.out.stat().st_size / 1e6)

    summary["sources"] = {
        "dataset_v2_dir": str(args.dataset_v2_dir),
        "synth_dir": str(args.synth_dir) if not args.no_synth else None,
        "excluded_prefixes": args.exclude,
        "num_trajectories_processed": len(per_traj_meta),
        "num_real_processed":  sum(1 for m in per_traj_meta if m["source"] == "real"),
        "num_synth_processed": sum(1 for m in per_traj_meta if m["source"] == "synth"),
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("saved summary %s", args.summary)

    print("\n=== Step 1 summary ===")
    src = summary["sources"]
    print(f"trajectories: {src['num_trajectories_processed']}  "
          f"(real={src['num_real_processed']}, synth={src['num_synth_processed']})")
    print(f"total rows:   {summary['row_count']}")
    print(f"per material: {summary['per_material_rows']}")
    print(f"per motion:   {summary['per_motion_type_rows']}")
    print(f"mask cov:     {summary['mask_coverage']}")
    if summary.get("warnings"):
        print(f"warnings ({len(summary['warnings'])}):")
        for w in summary["warnings"]:
            print(f"  - {w}")
    else:
        print("warnings:     none")
    print()


if __name__ == "__main__":
    main()
