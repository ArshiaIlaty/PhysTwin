#!/usr/bin/env python3
"""
build_subgoal_dataset.py — Hierarchical policy, Model B's training data

For Model B (the sub-goal planner) we want:
  input  = state[t] (31) + force_now=y[t] (6) + long_term_goal=y[t+H_long] (6)
  target = sub_goal = y[t + H_short] (6)

We write the dataset in the same npz schema as policy_dataset.npz so
train_policy.py can train Model B with zero code changes — we just
remap the field meanings:
   state       = X[t]                              (same)
   force_now   = y_per_ctrl[t]                     (same)
   force_goal  = y_per_ctrl[t + H_long]            (was: y[t+1] for Model A)
   action      = y_per_ctrl[t + H_short]           (was: ctrl Δ for Model A)
   action_mask = mask from n_ctrl_parts            (same)
   ...

Why we don't reuse build_policy_dataset.py: build_policy emits the
controller Δ as the action; here we emit the intermediate force.
Different label semantics, so we keep the script separate to avoid
field-name confusion.

Plan: my_work/docs/closed_loop_control/fix_c_plan.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger("build_subgoal_dataset")

SCRIPT_DIR = Path(__file__).resolve().parent
MY_WORK = SCRIPT_DIR.parent
RESULTS = MY_WORK / "results"

DEFAULT_DATASET_V2 = RESULTS / "dataset_v2"
DEFAULT_SYNTH = RESULTS / "dataset_synth_raw"
DEFAULT_OUT = RESULTS / "subgoal_dataset.npz"
DEFAULT_SUMMARY = RESULTS / "subgoal_dataset_summary.json"
DEFAULT_EXCLUDE = ["single_push_sloth"]


def _pad_y(y: np.ndarray, T: int) -> np.ndarray:
    if y.ndim == 2:
        y = y.reshape(T, 1, 3)
    if y.shape[1] < 2:
        pad = np.zeros((T, 2 - y.shape[1], 3), dtype=y.dtype)
        y = np.concatenate([y, pad], axis=1)
    return y.astype(np.float32)


def load_trajectory(npz_path: Path, source: str):
    d = np.load(npz_path, allow_pickle=True)
    needed = {"X", "y_per_ctrl", "controller_pos", "n_ctrl_parts"}
    if needed - set(d.files):
        return None
    T = d["X"].shape[0]
    if T < 2:
        return None
    return {
        "X": d["X"].astype(np.float32),
        "y_per_ctrl": d["y_per_ctrl"].astype(np.float32),
        "controller_pos": d["controller_pos"].astype(np.float32),
        "n_ctrl_parts": int(d["n_ctrl_parts"]),
        "material": (str(d["object_category"]) if "object_category" in d.files
                     else str(d.get("material", "unknown"))),
        "case_name": str(d["case_name"]),
        "motion_type": str(d["motion_type"]) if "motion_type" in d.files else "real",
        "source": source,
    }


def trajectory_to_rows(traj: dict, h_short: int, h_long: int):
    X = traj["X"]
    n_ctrl_parts = traj["n_ctrl_parts"]
    T = X.shape[0]
    y = _pad_y(traj["y_per_ctrl"], T)

    # We need t + h_long < T; output T_rows = T - h_long rows.
    T_rows = T - h_long
    if T_rows <= 0:
        return None

    state = X[:T_rows]                              # [T_rows, 31]
    force_now = y[:T_rows]                          # [T_rows, 2, 3]
    long_term_goal = y[h_long:h_long + T_rows]      # [T_rows, 2, 3]
    sub_goal = y[h_short:h_short + T_rows]          # [T_rows, 2, 3]

    rows = {
        "state":         state.astype(np.float32),
        "force_now":     force_now.astype(np.float32),
        "force_goal":    long_term_goal.astype(np.float32),  # ← long-term goal
        "action":        sub_goal.astype(np.float32),         # ← sub-goal force
        "action_mask":   np.array(
            [[1.0] * n_ctrl_parts + [0.0] * (2 - n_ctrl_parts)] * T_rows,
            dtype=np.float32,
        ),
        "material":      np.array([traj["material"]] * T_rows, dtype="<U16"),
        "case_name":     np.array([traj["case_name"]] * T_rows, dtype="<U64"),
        "n_ctrl_parts":  np.full(T_rows, n_ctrl_parts, dtype=np.int8),
        "source":        np.array([traj["source"]] * T_rows, dtype="<U10"),
        "motion_type":   np.array([traj["motion_type"]] * T_rows, dtype="<U16"),
    }
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_v2_dir", type=Path, default=DEFAULT_DATASET_V2)
    p.add_argument("--synth_dir", type=Path, default=DEFAULT_SYNTH)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE))
    p.add_argument("--no_synth", action="store_true")
    p.add_argument("--h_short", type=int, default=5,
                   help="sub-goal horizon (Model B predicts force at t+h_short)")
    p.add_argument("--h_long", type=int, default=20,
                   help="long-term horizon (Model B sees goal at t+h_long)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    assert args.h_short < args.h_long, "h_short must be < h_long"

    real = sorted(args.dataset_v2_dir.glob("*.npz")) if args.dataset_v2_dir.exists() else []
    synth = (sorted(args.synth_dir.glob("*.npz"))
             if (not args.no_synth and args.synth_dir.exists()) else [])
    logger.info("found %d real, %d synth", len(real), len(synth))
    logger.info("h_short=%d, h_long=%d", args.h_short, args.h_long)

    all_rows = []
    skipped = {"too_short": 0, "excluded": 0}
    n_trajs = 0
    for path, src in [(p, "real") for p in real] + [(p, "synth") for p in synth]:
        traj = load_trajectory(path, src)
        if traj is None:
            skipped["too_short"] += 1
            continue
        if any(traj["case_name"].startswith(x) for x in args.exclude):
            skipped["excluded"] += 1
            continue
        result = trajectory_to_rows(traj, args.h_short, args.h_long)
        if result is None:
            skipped["too_short"] += 1
            continue
        all_rows.append(result)
        n_trajs += 1

    if not all_rows:
        logger.error("no trajectories processed; aborting")
        sys.exit(1)

    keys = list(all_rows[0].keys())
    combined = {k: np.concatenate([r[k] for r in all_rows], axis=0) for k in keys}
    R = combined["state"].shape[0]
    logger.info("combined %d trajectories → %d rows", n_trajs, R)

    # Validation: no NaN/Inf
    for f in ("state", "force_now", "force_goal", "action"):
        if not np.isfinite(combined[f]).all():
            raise AssertionError(f"NaN/Inf in {f}")

    # Sanity stats
    mat = combined["material"]
    materials = sorted(np.unique(mat).tolist())
    summary = {
        "row_count": R,
        "n_trajectories": n_trajs,
        "skipped": skipped,
        "h_short": args.h_short,
        "h_long": args.h_long,
        "per_material_rows": {m: int((mat == m).sum()) for m in materials},
        "nan_inf_check": "passed",
    }

    # Sub-goal force magnitudes per material (sanity: should be in physical range)
    summary["subgoal_force_mag_N"] = {}
    sub_mag = np.linalg.norm(combined["action"], axis=-1)  # [R, 2]
    mask = combined["action_mask"]
    for m in materials:
        rm = mat == m
        vals = sub_mag[rm][mask[rm] > 0]
        if len(vals):
            summary["subgoal_force_mag_N"][m] = {
                "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)),
                "p99": float(np.percentile(vals, 99)),
                "max": float(vals.max()),
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **combined)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("saved %s (%.1f MB) + %s", args.out, args.out.stat().st_size / 1e6,
                args.summary)

    print()
    print("=== G1 subgoal dataset summary ===")
    print(f"trajectories: {n_trajs}  rows: {R}")
    print(f"per material: {summary['per_material_rows']}")
    print("subgoal force mag (N, per material):")
    for m in materials:
        s = summary['subgoal_force_mag_N'].get(m, {})
        if s:
            print(f"  {m:6s}  p50={s['p50']:>6.0f}  p95={s['p95']:>7.0f}  p99={s['p99']:>7.0f}  max={s['max']:>7.0f}")
    print(f"skipped: {skipped}")
    print()


if __name__ == "__main__":
    main()
