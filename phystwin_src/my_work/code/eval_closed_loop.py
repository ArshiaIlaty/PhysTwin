#!/usr/bin/env python3
"""
eval_closed_loop.py — Step 4 systematic evaluation

Runs the trained policy across a curated set of (case, profile) pairs via
run_closed_loop.py (subprocess), collects per-rollout metrics, and aggregates
per material.

Defaults to the Step 4 plan's case/profile set (11 replay + 3 ramp = 14
rollouts). Pass --cases to override for smoke tests.

Plan:    my_work/docs/closed_loop_control/step4_plan.md
Outputs: my_work/results/eval_closed_loop/{case}__{profile}.npz
         my_work/results/eval_closed_loop/summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("eval_closed_loop")

SCRIPT_DIR = Path(__file__).resolve().parent
MY_WORK = SCRIPT_DIR.parent
REPO_ROOT = MY_WORK.parent
RESULTS = MY_WORK / "results"

DEFAULT_OUT = RESULTS / "eval_closed_loop"
DEFAULT_POLICY_DIR = RESULTS / "models_policy" / "seed_1"
DEFAULT_DATASET_V2 = RESULTS / "dataset_v2"

# Per step4_plan.md case selection
DEFAULT_CASES_REPLAY = [
    "single_push_rope_1", "single_push_rope", "single_push_rope_4", "single_lift_rope",
    "double_lift_cloth_3", "double_lift_cloth_1", "single_clift_cloth_1", "single_lift_cloth",
    "double_stretch_sloth", "double_lift_sloth", "single_lift_sloth",
]
DEFAULT_CASES_RAMP = [
    "single_push_rope_1", "double_lift_cloth_3", "double_stretch_sloth",
]


def preflight(case: str, dataset_v2_dir: Path, repo_root: Path) -> str | None:
    """Return error message if case can't run, None if OK."""
    models = list((repo_root / "experiments" / case / "train").glob("best_*.pth"))
    if not models:
        return f"missing experiments/{case}/train/best_*.pth"
    v2 = dataset_v2_dir / f"{case}.npz"
    if not v2.exists():
        return f"missing dataset_v2/{case}.npz"
    return None


def run_one(case: str, profile: str, out_dir: Path, policy_dir: Path,
             repo_root: Path, verbose: bool = False) -> bool:
    cmd = [
        sys.executable,
        str(MY_WORK / "code" / "run_closed_loop.py"),
        "--case_name", case,
        "--profile", profile,
        "--policy_dir", str(policy_dir),
        "--out_dir", str(out_dir),
    ]
    if profile == "policy_ramp":
        cmd += ["--ramp_scale", "1.0"]
    logger.info("running: %s", " ".join(cmd[-7:]))
    proc = subprocess.run(
        cmd, cwd=str(repo_root),
        capture_output=not verbose, text=True,
    )
    if proc.returncode != 0:
        logger.error("rollout failed for %s/%s (rc=%d)", case, profile, proc.returncode)
        if proc.stderr:
            logger.error("stderr tail:\n%s", proc.stderr[-1500:])
        return False
    return True


def compute_metrics(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    forces = d["forces"]                   # [T, n_ctrl, 3]
    F_goal = d["F_goal"]                   # [T, 2, 3]
    positions = d["positions"]
    ctrl_pos = d["controller_pos"]
    n = int(d["n_ctrl_parts"])
    T = int(forces.shape[0])

    F_g_active = F_goal[:, :n]
    err = np.linalg.norm(forces - F_g_active, axis=-1)
    mag_a = np.linalg.norm(forces, axis=-1)
    mag_g = np.linalg.norm(F_g_active, axis=-1)

    any_nan = bool(np.isnan(forces).any()
                   or np.isnan(positions).any()
                   or np.isnan(ctrl_pos).any())

    if any_nan:
        mean_err = p95_err = peak_a = final_force = final_drift = float("nan")
        err_ratio = overshoot = float("nan")
    else:
        mean_err = float(np.mean(err))
        p95_err = float(np.percentile(err, 95))
        peak_a = float(np.max(mag_a))
        final_force = float(np.mean(mag_a[-1]))
        final_drift = float(np.max(np.linalg.norm(ctrl_pos[-1] - ctrl_pos[0], axis=-1)))
    mean_goal = float(np.mean(mag_g))
    peak_g = float(np.max(mag_g))
    if not any_nan and mean_goal > 1e-6:
        err_ratio = mean_err / mean_goal
    if not any_nan and peak_g > 1e-6:
        overshoot = peak_a / peak_g

    return {
        "case_name": str(d["case_name"]),
        "material": str(d["material"]),
        "profile": str(d["profile"]),
        "n_ctrl_parts": n,
        "T": T,
        "mean_force_err_N": mean_err,
        "p95_force_err_N": p95_err,
        "mean_goal_mag_N": mean_goal,
        "force_err_ratio": err_ratio,
        "peak_overshoot_ratio": overshoot,
        "final_force_N": final_force,
        "final_drift_m": final_drift,
        "any_nan": any_nan,
    }


def aggregate(per_rollout: dict) -> dict:
    grouped: dict = {}
    for m in per_rollout.values():
        grouped.setdefault((m["material"], m["profile"]), []).append(m)

    out: dict = {}
    for (mat, prof), items in grouped.items():
        out.setdefault(mat, {})
        ratios = [it["force_err_ratio"] for it in items if not np.isnan(it["force_err_ratio"])]
        overs = [it["peak_overshoot_ratio"] for it in items if not np.isnan(it["peak_overshoot_ratio"])]
        nans = sum(1 for it in items if it["any_nan"])
        uncontrolled = sum(
            1 for it in items
            if (not it["any_nan"]) and it["force_err_ratio"] > 1.0
        )
        out[mat][prof] = {
            "n_cases": len(items),
            "n_nan_rollouts": nans,
            "n_uncontrolled": uncontrolled,
            "mean_force_err_ratio": float(np.mean(ratios)) if ratios else float("nan"),
            "std_force_err_ratio": float(np.std(ratios)) if len(ratios) > 1 else 0.0,
            "mean_peak_overshoot_ratio": float(np.mean(overs)) if overs else float("nan"),
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases_replay", nargs="*", default=DEFAULT_CASES_REPLAY)
    p.add_argument("--cases_ramp", nargs="*", default=DEFAULT_CASES_RAMP)
    p.add_argument("--cases", nargs="*", default=None,
                   help="Override: use these cases for ALL profiles in --profiles")
    p.add_argument("--profiles", nargs="*",
                   default=["policy_recorded_goal", "policy_ramp"])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--policy_dir", type=Path, default=DEFAULT_POLICY_DIR)
    p.add_argument("--dataset_v2_dir", type=Path, default=DEFAULT_DATASET_V2)
    p.add_argument("--summary_only", action="store_true",
                   help="Skip rollouts; aggregate npzs already in --out")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.cases is not None:
        case_profile_pairs = [(c, p) for c in args.cases for p in args.profiles]
    else:
        case_profile_pairs = []
        if "policy_recorded_goal" in args.profiles:
            case_profile_pairs += [(c, "policy_recorded_goal") for c in args.cases_replay]
        if "policy_ramp" in args.profiles:
            case_profile_pairs += [(c, "policy_ramp") for c in args.cases_ramp]

    logger.info("plan: %d rollouts", len(case_profile_pairs))
    args.out.mkdir(parents=True, exist_ok=True)

    # Preflight
    unique_cases = sorted(set(c for c, _ in case_profile_pairs))
    failed_preflight: dict = {}
    for c in unique_cases:
        err = preflight(c, args.dataset_v2_dir, REPO_ROOT)
        if err:
            failed_preflight[c] = err
            logger.warning("preflight skip: %s — %s", c, err)
    case_profile_pairs = [
        (c, p) for c, p in case_profile_pairs if c not in failed_preflight
    ]

    if not args.summary_only:
        n_ok = 0
        for i, (case, profile) in enumerate(case_profile_pairs, 1):
            logger.info("[%d/%d] case=%s profile=%s",
                        i, len(case_profile_pairs), case, profile)
            if run_one(case, profile, args.out, args.policy_dir, REPO_ROOT,
                        verbose=args.verbose):
                n_ok += 1
        logger.info("rollouts: %d/%d ok", n_ok, len(case_profile_pairs))

    # Aggregate
    per_rollout: dict = {}
    for case, profile in case_profile_pairs:
        npz = args.out / f"{case}__{profile}.npz"
        if not npz.exists():
            logger.warning("missing npz: %s", npz.name)
            continue
        try:
            per_rollout[f"{case}__{profile}"] = compute_metrics(npz)
        except Exception as e:
            logger.error("metric compute failed for %s: %s", npz.name, e)

    per_material = aggregate(per_rollout)

    summary = {
        "per_rollout": per_rollout,
        "per_material": per_material,
        "policy_dir": str(args.policy_dir),
        "failed_preflight": failed_preflight,
        "n_rollouts_attempted": len(case_profile_pairs),
        "n_rollouts_with_metrics": len(per_rollout),
    }
    summary_path = args.out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("wrote %s", summary_path)

    # Headline table
    print()
    print("=== Per-material aggregates ===")
    for mat, profs in sorted(per_material.items()):
        for prof, agg in profs.items():
            print(f"  {mat:6s} {prof:24s}  "
                  f"n={agg['n_cases']:>2d}  "
                  f"err_ratio={agg['mean_force_err_ratio']:.3f}±{agg['std_force_err_ratio']:.3f}  "
                  f"overshoot={agg['mean_peak_overshoot_ratio']:.2f}  "
                  f"nan={agg['n_nan_rollouts']}/{agg['n_cases']}  "
                  f"uncontrolled={agg['n_uncontrolled']}/{agg['n_cases']}")
    print()


if __name__ == "__main__":
    main()
