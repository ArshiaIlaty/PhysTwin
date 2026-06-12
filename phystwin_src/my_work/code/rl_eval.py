#!/usr/bin/env python3
"""
rl_eval.py — Evaluate a trained RL actor across the standard 14-case sweep.

Bridges the RL checkpoint (which stores actor+critic+log_std) into the
supervised-policy CLI by writing a stripped policy.pt + scalers into a
compat dir, then invoking eval_closed_loop.py.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
MY_WORK = SCRIPT_DIR.parent
REPO_ROOT = MY_WORK.parent


def extract_supervised_compatible_dir(rl_ckpt: Path, out_dir: Path,
                                       scaler_src_dir: Path):
    """Make the RL actor look like a normal supervised policy_dir for eval.

    Policy-v2 runs (scaler_src_dir has arch.json) store the transformer under
    actor."model.*"; strip that prefix and copy the v2 artifacts so
    run_closed_loop's arch.json loader picks it up.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(rl_ckpt, map_location="cpu", weights_only=True)
    actor_sd = ckpt["actor"]
    is_v2 = (scaler_src_dir / "arch.json").exists()
    clean_sd = {}
    for k, v in actor_sd.items():
        if k.startswith("log_std"):
            continue
        if is_v2 and k.startswith("model."):
            clean_sd[k[len("model."):]] = v
        elif not is_v2 and k.startswith("net."):
            clean_sd[k] = v
    torch.save(clean_sd, out_dir / "policy.pt")
    files = (["scalers_v2.pkl", "arch.json", "target_scalers.pkl"] if is_v2
             else ["feat_scaler.pkl", "target_scalers.pkl"])
    for f in files:
        shutil.copy(scaler_src_dir / f, out_dir / f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rl_run_dir", type=Path, required=True)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    ckpt = args.rl_run_dir / args.checkpoint
    if not ckpt.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt}")

    compat_dir = args.out / "policy_dir"
    extract_supervised_compatible_dir(ckpt, compat_dir, args.rl_run_dir)
    print(f"compat dir at {compat_dir}")

    cmd = [
        sys.executable,
        str(MY_WORK / "code" / "eval_closed_loop.py"),
        "--policy_dir", str(compat_dir),
        "--out", str(args.out),
    ]
    print("running:", " ".join(cmd[-4:]))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    main()
