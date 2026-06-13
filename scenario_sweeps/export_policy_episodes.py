#!/usr/bin/env python3
"""Export policy_dataset.npz rows to episodes CSV for BC / stiffness policy training.

Reads Malak's build_policy_dataset.py output and writes wide-format CSV:
  case_name, stiffness_case, obs_0..obs_N, act_0..act_M, gt_force, step, material, motion_type

Synthetic trajectories map stiffness via `stiffness_case` (donor case prefix before __synth_).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def stiffness_case_name(case_name: str) -> str:
    if "__synth_" in case_name:
        return case_name.split("__synth_")[0]
    return case_name


def force_magnitude(force_goal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scalar target force per row: sum of per-group ‖F‖ with action mask."""
    per_group = np.linalg.norm(force_goal, axis=-1)  # [R, 2]
    return (per_group * mask).sum(axis=1)


def export_episodes(
    policy_npz: Path,
    out_csv: Path,
    max_rows: int | None = None,
) -> dict:
    d = np.load(policy_npz, allow_pickle=True)
    state = d["state"].astype(np.float32)
    action = d["action"].reshape(len(d["action"]), -1).astype(np.float32)
    mask = d["action_mask"].astype(np.float32)
    force_goal = d["force_goal"].astype(np.float32)
    case_names = d["case_name"].astype(str)
    materials = d["material"].astype(str)
    motion_types = d["motion_type"].astype(str)

    n = state.shape[0]
    if max_rows is not None:
        n = min(n, max_rows)

    gt_force = force_magnitude(force_goal[:n], mask[:n])
    stiff_cases = [stiffness_case_name(c) for c in case_names[:n]]

    data = {
        "case_name": case_names[:n],
        "stiffness_case": stiff_cases,
        "gt_force": gt_force,
        "step": np.arange(n, dtype=np.int32),
        "material": materials[:n],
        "motion_type": motion_types[:n],
    }
    for i in range(state.shape[1]):
        data[f"obs_{i}"] = state[:n, i]
    for i in range(action.shape[1]):
        data[f"act_{i}"] = action[:n, i]

    df = pd.DataFrame(data)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = {
        "source_npz": str(policy_npz),
        "out_csv": str(out_csv),
        "rows": int(len(df)),
        "obs_dim": int(state.shape[1]),
        "act_dim": int(action.shape[1]),
        "unique_cases": int(df["case_name"].nunique()),
        "unique_stiffness_cases": int(df["stiffness_case"].nunique()),
        "per_material": df.groupby("material").size().to_dict(),
        "per_motion": df.groupby("motion_type").size().to_dict(),
    }
    summary_path = out_csv.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Exported {len(df)} rows → {out_csv}")
    print(f"  obs_dim={summary['obs_dim']}  act_dim={summary['act_dim']}")
    print(f"  cases={summary['unique_cases']}  stiffness_cases={summary['unique_stiffness_cases']}")
    print(f"  per material: {summary['per_material']}")
    print(f"  summary → {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy_npz",
        type=Path,
        default=Path("phystwin_src/my_work/results/policy_dataset.npz"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("scenario_sweeps/episodes.csv"),
    )
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()
    if not args.policy_npz.exists():
        raise SystemExit(f"Missing {args.policy_npz}. Run build_policy_dataset.py first.")
    export_episodes(args.policy_npz, args.out, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
