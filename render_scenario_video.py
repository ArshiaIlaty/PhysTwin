"""render_scenario_video.py — render force-visualization videos for scenario sweeps.

Replays the same synthetic motion + physics overrides used in
generate_scenario_sweeps.py (no policy / BC code).

Examples:
  # Single scenario tag
  xvfb-run -a python render_scenario_video.py \\
    --tag single_clift_cloth_3__hold_release__k0.5__concrete

  # Side-by-side stiffness comparison (one per material)
  xvfb-run -a python render_scenario_video.py --compare_defaults

Run sweeps on CPU when GPU is busy:
  python generate_scenario_sweeps.py --device cpu ...
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import cv2
import numpy as np
import torch

from generate_scenario_sweeps import (
    DEFAULT_DONORS,
    MOTION_AMP_BY_MATERIAL,
    MOTION_BY_MATERIAL,
    SEED_BY_MATERIAL,
    SURFACE_PRESETS,
)
from generate_synthetic import _setup_trainer, make_motion


TAG_RE = re.compile(r"^(.+)__(.+)__k([\d.]+)__(.+)$")

DEFAULT_COMPARE = {
    "cloth": {"donor": "single_clift_cloth_3", "n_ctrl": 1, "cfg": "cloth",
              "motion": "hold_release", "surface": "concrete", "k_low": 0.5, "k_high": 2.0},
    "rope": {"donor": "single_push_rope_1", "n_ctrl": 1, "cfg": "real",
             "motion": "linear_push", "surface": "concrete", "k_low": 0.5, "k_high": 2.0},
    "sloth": {"donor": "double_stretch_sloth", "n_ctrl": 2, "cfg": "real",
              "motion": "hold_release", "surface": "concrete", "k_low": 0.5, "k_high": 2.0},
}


def parse_tag(tag: str) -> dict:
    m = TAG_RE.match(tag)
    if not m:
        raise ValueError(f"Bad scenario tag: {tag!r}")
    donor, motion, k_str, surface = m.groups()
    if surface not in SURFACE_PRESETS:
        raise ValueError(f"Unknown surface preset: {surface}")
    return {
        "donor": donor,
        "motion": motion,
        "stiffness_scale": float(k_str),
        "surface": surface,
    }


def gaussian_path(case_name: str, gaussian_root: str) -> str:
    exp = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    path = (
        f"{gaussian_root}/{case_name}/{exp}/point_cloud/iteration_10000/point_cloud.ply"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Gaussian ply not found: {path}")
    return path


def prepare_trainer(
    donor: str,
    n_ctrl_parts: int,
    cfg_type: str,
    material: str,
    motion: str,
    T: int,
    seed: int,
    device: str,
):
    from qqtt.utils import cfg
    import json
    import pickle

    trainer, best_path = _setup_trainer(donor, n_ctrl_parts, cfg_type, device=device)
    base_path = "./data/different_types"
    cfg.overlay_path = f"{base_path}/{donor}/color"

    orig_ctrl = trainer.simulator.controller_points.detach().cpu().numpy()
    ctrl_rest = orig_ctrl[0]
    donor_ctrl_extent = float(
        np.linalg.norm(orig_ctrl - orig_ctrl[0:1], axis=-1).max()
    ) + 1e-6
    amp = MOTION_AMP_BY_MATERIAL.get(material, (0.15, 0.35))
    rng = np.random.RandomState(seed)
    ctrl_synth = make_motion(
        motion, T, ctrl_rest.shape[0], ctrl_rest, donor_ctrl_extent, rng,
        amp_min=amp[0], amp_max=amp[1],
    )
    ctrl_tensor = torch.tensor(
        ctrl_synth,
        device=trainer.simulator.controller_points.device,
        dtype=trainer.simulator.controller_points.dtype,
    )
    trainer.simulator.controller_points = ctrl_tensor
    trainer.controller_points = ctrl_tensor
    trainer.dataset.frame_len = T
    return trainer, best_path


def render_one(
    donor: str,
    n_ctrl_parts: int,
    cfg_type: str,
    material: str,
    motion: str,
    stiffness_scale: float,
    surface: str,
    out_path: str,
    T: int,
    seed: int,
    device: str,
    gaussian_root: str,
    force_scale: float,
) -> str:
    preset = SURFACE_PRESETS[surface]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    trainer, best_path = prepare_trainer(
        donor, n_ctrl_parts, cfg_type, material, motion, T, seed, device,
    )
    gs_path = gaussian_path(donor, gaussian_root)
    label = f"k={stiffness_scale:g} {surface}"
    print(f"  Rendering {label} -> {out_path}")

    trainer.visualize_force(
        best_path,
        gs_path,
        n_ctrl_parts=n_ctrl_parts,
        force_scale=force_scale,
        stiffness_scale=stiffness_scale,
        collide_elas=preset["collide_elas"],
        collide_fric=preset["collide_fric"],
        video_path=out_path,
    )
    _annotate_video(out_path, label)
    return out_path


def _annotate_video(path: str, label: str) -> None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = path + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.putText(
            frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )
        writer.write(frame)
    cap.release()
    writer.release()
    os.replace(tmp, path)


def hstack_videos(left: str, right: str, out_path: str, title: str = "") -> None:
    cap_l = cv2.VideoCapture(left)
    cap_r = cv2.VideoCapture(right)
    fps = cap_l.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h),
    )
    while True:
        ok_l, fl = cap_l.read()
        ok_r, fr = cap_r.read()
        if not (ok_l and ok_r):
            break
        if fr.shape[:2] != fl.shape[:2]:
            fr = cv2.resize(fr, (fl.shape[1], fl.shape[0]))
        combo = np.hstack([fl, fr])
        if title:
            cv2.putText(
                combo, title, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (50, 50, 50), 2,
            )
        writer.write(combo)
    cap_l.release()
    cap_r.release()
    writer.release()
    print(f"  Comparison -> {out_path}")


def infer_material(donor: str) -> str:
    for mat, donors in DEFAULT_DONORS.items():
        if any(d[0] == donor for d in donors):
            return mat
    if "cloth" in donor:
        return "cloth"
    if "rope" in donor:
        return "rope"
    if "sloth" in donor:
        return "sloth"
    return "cloth"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default=None, help="Scenario npz tag to render.")
    parser.add_argument(
        "--compare_defaults",
        action="store_true",
        help="Render k=0.5 vs k=2.0 side-by-side for cloth/rope/sloth.",
    )
    parser.add_argument("--donor", type=str, default=None)
    parser.add_argument("--motion", type=str, default=None)
    parser.add_argument("--stiffness", type=float, default=None)
    parser.add_argument("--surface", type=str, default="concrete")
    parser.add_argument("--n_ctrl_parts", type=int, default=None)
    parser.add_argument("--cfg_type", type=str, default=None, choices=["real", "cloth"])
    parser.add_argument("--material", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="scenario_sweeps/forward_force_results/videos")
    parser.add_argument("--T", type=int, default=120)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gaussian_path", type=str, default="./gaussian_output")
    parser.add_argument("--force_scale", type=float, default=30000.0)
    args = parser.parse_args()

    if args.device == "cpu":
        print("NOTE: --device cpu works for sweeps; video rendering still uses GPU for Gaussians if available.")

    if args.compare_defaults:
        for mat, spec in DEFAULT_COMPARE.items():
            seed = SEED_BY_MATERIAL.get(mat, 42)
            low_path = os.path.join(
                args.out_dir, f"{mat}__k{spec['k_low']}__{spec['surface']}.mp4",
            )
            high_path = os.path.join(
                args.out_dir, f"{mat}__k{spec['k_high']}__{spec['surface']}.mp4",
            )
            cmp_path = os.path.join(
                args.out_dir, f"compare_{mat}_k{spec['k_low']}_vs_k{spec['k_high']}.mp4",
            )
            print(f"\n=== {mat} ===")
            render_one(
                spec["donor"], spec["n_ctrl"], spec["cfg"], mat, spec["motion"],
                spec["k_low"], spec["surface"], low_path, args.T, seed,
                args.device, args.gaussian_path, args.force_scale,
            )
            render_one(
                spec["donor"], spec["n_ctrl"], spec["cfg"], mat, spec["motion"],
                spec["k_high"], spec["surface"], high_path, args.T, seed,
                args.device, args.gaussian_path, args.force_scale,
            )
            hstack_videos(
                low_path, high_path, cmp_path,
                title=f"{mat}: stiffness x{spec['k_low']} (left) vs x{spec['k_high']} (right)",
            )
        print(f"\nDone. Videos in {args.out_dir}/")
        return

    if args.tag:
        parsed = parse_tag(args.tag)
        donor = parsed["donor"]
        motion = parsed["motion"]
        stiffness = parsed["stiffness_scale"]
        surface = parsed["surface"]
        material = args.material or infer_material(donor)
        n_ctrl = args.n_ctrl_parts
        cfg_type = args.cfg_type
        if n_ctrl is None or cfg_type is None:
            for mat, donors in DEFAULT_DONORS.items():
                for d, nc, ct in donors:
                    if d == donor:
                        n_ctrl = nc
                        cfg_type = ct
                        material = mat
                        break
        assert n_ctrl is not None and cfg_type is not None, f"Unknown donor {donor}"
        seed = args.seed if args.seed is not None else SEED_BY_MATERIAL.get(material, 42)
        out_path = os.path.join(args.out_dir, f"{args.tag}.mp4")
        render_one(
            donor, n_ctrl, cfg_type, material, motion, stiffness, surface,
            out_path, args.T, seed, args.device, args.gaussian_path, args.force_scale,
        )
        return

    if args.donor and args.stiffness is not None:
        material = args.material or infer_material(args.donor)
        motion = args.motion or MOTION_BY_MATERIAL.get(material, "hold_release")
        n_ctrl = args.n_ctrl_parts
        cfg_type = args.cfg_type
        if n_ctrl is None or cfg_type is None:
            for mat, donors in DEFAULT_DONORS.items():
                for d, nc, ct in donors:
                    if d == args.donor:
                        n_ctrl, cfg_type = nc, ct
                        material = mat
        seed = args.seed if args.seed is not None else SEED_BY_MATERIAL.get(material, 42)
        tag = f"{args.donor}__{motion}__k{args.stiffness:g}__{args.surface}"
        out_path = os.path.join(args.out_dir, f"{tag}.mp4")
        render_one(
            args.donor, n_ctrl, cfg_type, material, motion, args.stiffness,
            args.surface, out_path, args.T, seed, args.device,
            args.gaussian_path, args.force_scale,
        )
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
