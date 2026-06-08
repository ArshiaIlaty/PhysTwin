"""generate_scenario_sweeps.py — counterfactual physics scenario sweeps.

For each donor case, run a fixed synthetic motion while sweeping:
  - object stiffness  (scale on object spring_Y)
  - surface proxy     (collide_elas / collide_fric presets)

Outputs one .npz per (donor, motion, stiffness, surface) to dataset_scenarios/.
Use analyze_scenarios.py to plot peak-force sensitivity curves.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import numpy as np
import torch

from generate_synthetic import (
    DONORS,
    MAX_DISP_RATIO,
    MAX_F_RATIO,
    _setup_trainer,
    make_motion,
)
from extract_dataset import SUMMARY_FEATURE_NAMES, _object_category, pad_forces, summary_features
from augment_dataset import NEW_FEATURE_NAMES


OUT_DIR = "dataset_scenarios"

# Surface proxies — not real material models, but ordered friction/restitution knobs.
SURFACE_PRESETS = {
    "concrete": {"collide_elas": 0.20, "collide_fric": 0.85},
    "grass":    {"collide_elas": 0.05, "collide_fric": 0.40},
    "rubber":   {"collide_elas": 0.50, "collide_fric": 0.70},
}

DEFAULT_STIFFNESS_SCALES = (0.5, 1.0, 2.0)

# One stable donor per material for fast sweeps (override with --donors).
# hold_release spikes on rope; linear_push stays within donor force regime.
MOTION_BY_MATERIAL = {
    "rope": "linear_push",
    "cloth": "hold_release",
    "sloth": "hold_release",
}

# Rope needs smaller synthetic offsets to stay in the donor's calibrated regime.
MOTION_AMP_BY_MATERIAL = {
    "rope": (0.08, 0.18),
    "cloth": (0.15, 0.35),
    "sloth": (0.15, 0.35),
}

SEED_BY_MATERIAL = {
    "rope": 99,
    "cloth": 42,
    "sloth": 42,
}

DEFAULT_DONORS = {
    "rope":  [("single_push_rope_1", 1, "real")],
    "cloth": [("single_clift_cloth_3", 1, "cloth")],
    "sloth": [("double_stretch_sloth", 2, "real")],
}


def _build_features(pos: np.ndarray, ctrl_out: np.ndarray) -> np.ndarray:
    X = summary_features(pos)
    ctrl_centroid = ctrl_out.mean(axis=1)
    ctrl_centroid_disp = ctrl_centroid - ctrl_centroid[0:1]
    obj_centroid = pos.mean(axis=1)
    rel_motion = ctrl_centroid - obj_centroid
    diffs = pos - ctrl_centroid[:, None, :]
    obj_to_ctrl = np.linalg.norm(diffs, axis=-1)
    nearest_dist = obj_to_ctrl.min(axis=1, keepdims=True)
    mean_contact_dist = obj_to_ctrl.mean(axis=1, keepdims=True)
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    vel_mag = np.linalg.norm(vel, axis=-1)
    mean_vel_mag = vel_mag.mean(axis=1, keepdims=True)
    max_vel_mag = vel_mag.max(axis=1, keepdims=True)
    centroid_vel = vel.mean(axis=1)
    extra = np.concatenate(
        [
            ctrl_centroid_disp,
            nearest_dist,
            mean_contact_dist,
            rel_motion,
            mean_vel_mag,
            max_vel_mag,
            centroid_vel,
        ],
        axis=1,
    ).astype(np.float32)
    return np.concatenate([X, extra], axis=1).astype(np.float32)


def _validate_scenario(
    positions: np.ndarray,
    forces: np.ndarray,
    donor_max_force: float,
    donor_extent: float,
    donor_centroid: np.ndarray,
    stiffness_scale: float,
) -> tuple[bool, str]:
    """Like generate_synthetic._validate but scales force cap with stiffness."""
    if np.any(~np.isfinite(positions)) or np.any(~np.isfinite(forces)):
        return False, "NaN/Inf"
    fmax = float(np.linalg.norm(forces, axis=-1).max())
    force_cap = donor_max_force * MAX_F_RATIO * max(stiffness_scale, 1.0)
    if fmax > force_cap:
        return False, f"spike (max|F|={fmax:.0f} vs cap {force_cap:.0f})"
    obj_max_excursion = float(
        np.linalg.norm(positions - donor_centroid[None, None, :], axis=-1).max()
    )
    if obj_max_excursion > donor_extent * MAX_DISP_RATIO:
        return False, (
            f"runaway (excursion={obj_max_excursion:.3f}, "
            f"cap {donor_extent * MAX_DISP_RATIO:.3f})"
        )
    return True, "ok"


def sweep_donor(
    material: str,
    case_name: str,
    n_ctrl_parts: int,
    cfg_type: str,
    motion: str = "hold_release",
    stiffness_scales: tuple[float, ...] = DEFAULT_STIFFNESS_SCALES,
    surfaces: tuple[str, ...] = tuple(SURFACE_PRESETS),
    T: int = 120,
    seed: int = 0,
    amp_min: float | None = None,
    amp_max: float | None = None,
    device: str = "cuda:0",
) -> dict:
    if amp_min is None or amp_max is None:
        default_amp = MOTION_AMP_BY_MATERIAL.get(material, (0.15, 0.35))
        amp_min = default_amp[0] if amp_min is None else amp_min
        amp_max = default_amp[1] if amp_max is None else amp_max
    print(f"\n=== Scenario sweep: {case_name} ({material}) motion={motion} device={device} ===")
    trainer, best_path = _setup_trainer(case_name, n_ctrl_parts, cfg_type, device=device)

    orig_ctrl = trainer.simulator.controller_points.detach().cpu().numpy()
    ctrl_rest = orig_ctrl[0]
    donor_ctrl_extent = float(
        np.linalg.norm(orig_ctrl - orig_ctrl[0:1], axis=-1).max()
    ) + 1e-6

    pos_ref, f_ref, _, _ = trainer.extract_force_data(best_path, n_ctrl_parts=n_ctrl_parts)
    donor_max_force = float(np.linalg.norm(f_ref, axis=-1).max())
    donor_centroid = pos_ref.mean(axis=(0, 1))
    donor_extent = float(
        np.linalg.norm(pos_ref.max(axis=(0, 1)) - pos_ref.min(axis=(0, 1)))
    )
    print(f"  donor max|F|={donor_max_force:.0f} N")

    rng = np.random.RandomState(seed)
    ctrl_synth = make_motion(
        motion, T, ctrl_rest.shape[0], ctrl_rest, donor_ctrl_extent, rng,
        amp_min=amp_min, amp_max=amp_max,
    )
    ctrl_tensor = torch.tensor(
        ctrl_synth,
        device=trainer.simulator.controller_points.device,
        dtype=trainer.simulator.controller_points.dtype,
    )
    trainer.simulator.controller_points = ctrl_tensor
    trainer.controller_points = ctrl_tensor
    trainer.dataset.frame_len = T

    os.makedirs(OUT_DIR, exist_ok=True)
    stats = {"accepted": 0, "rejected": 0, "scenarios": []}

    for stiffness in stiffness_scales:
        for surface in surfaces:
            preset = SURFACE_PRESETS[surface]
            tag = f"{case_name}__{motion}__k{stiffness:g}__{surface}"
            try:
                pos, f, ctrl_out, _meta = trainer.extract_force_data(
                    best_path,
                    n_ctrl_parts=n_ctrl_parts,
                    stiffness_scale=stiffness,
                    collide_elas=preset["collide_elas"],
                    collide_fric=preset["collide_fric"],
                )
            except Exception as e:
                stats["rejected"] += 1
                print(f"  [{tag}] REJECTED exception: {type(e).__name__}: {e}")
                continue

            ok, reason = _validate_scenario(
                pos, f, donor_max_force, donor_extent, donor_centroid, stiffness,
            )
            if not ok:
                stats["rejected"] += 1
                print(f"  [{tag}] REJECTED {reason}")
                continue

            X_full = _build_features(pos, ctrl_out)
            y_per = pad_forces(f, max_ctrl_parts=2)
            y_net = f.sum(axis=1)
            peak_force = float(np.linalg.norm(y_net, axis=-1).max())
            mean_force = float(np.linalg.norm(y_net, axis=-1).mean())

            out_path = os.path.join(OUT_DIR, tag + ".npz")
            np.savez(
                out_path,
                X=X_full,
                y_per_ctrl=y_per,
                y_net=y_net.astype(np.float32),
                positions=pos.astype(np.float32),
                controller_pos=ctrl_out.astype(np.float32),
                case_name=tag,
                material=cfg_type,
                object_category=_object_category(case_name),
                n_ctrl_parts=n_ctrl_parts,
                feature_names=np.array(list(SUMMARY_FEATURE_NAMES) + NEW_FEATURE_NAMES),
                source_donor=case_name,
                motion_type=motion,
                stiffness_scale=float(stiffness),
                surface_preset=surface,
                collide_elas=float(preset["collide_elas"]),
                collide_fric=float(preset["collide_fric"]),
                peak_net_force=peak_force,
                mean_net_force=mean_force,
            )
            stats["accepted"] += 1
            stats["scenarios"].append(
                {
                    "tag": tag,
                    "peak_net_force": peak_force,
                    "mean_net_force": mean_force,
                    "stiffness_scale": stiffness,
                    "surface_preset": surface,
                }
            )
            print(f"  [{tag}] OK peak|F|={peak_force:.0f} N")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--materials",
        type=str,
        default="rope,cloth,sloth",
        help="Comma-separated materials (default: rope,cloth,sloth — no toy).",
    )
    parser.add_argument(
        "--motion",
        type=str,
        default="auto",
        help="Motion preset, or 'auto' for per-material defaults (rope→linear_push).",
    )
    parser.add_argument(
        "--stiffness",
        type=str,
        default="0.5,1.0,2.0",
        help="Comma-separated stiffness scale factors.",
    )
    parser.add_argument(
        "--surfaces",
        type=str,
        default="concrete,grass,rubber",
        help="Comma-separated surface preset names.",
    )
    parser.add_argument("--T", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Simulation device: cuda:0 or cpu (use cpu when GPU is occupied).",
    )
    args = parser.parse_args()

    stiffness_scales = tuple(float(x) for x in args.stiffness.split(",") if x.strip())
    surfaces = tuple(x.strip() for x in args.surfaces.split(",") if x.strip())
    for s in surfaces:
        if s not in SURFACE_PRESETS:
            raise SystemExit(f"Unknown surface preset: {s}")

    summary = {}
    for mat in [m.strip() for m in args.materials.split(",") if m.strip()]:
        donors = DEFAULT_DONORS.get(mat) or DONORS.get(mat, [])
        if not donors:
            print(f"!! Unknown material {mat}, skipping")
            continue
        mat_stats = []
        for idx, (case, n_ctrl, cfg) in enumerate(donors[:1]):  # one donor per material
            motion = (
                MOTION_BY_MATERIAL.get(mat, "hold_release")
                if args.motion == "auto"
                else args.motion
            )
            mat_seed = SEED_BY_MATERIAL.get(mat, args.seed + idx)
            s = sweep_donor(
                mat, case, n_ctrl, cfg,
                motion=motion,
                stiffness_scales=stiffness_scales,
                surfaces=surfaces,
                T=args.T,
                seed=mat_seed,
                device=args.device,
            )
            mat_stats.append(s)
        summary[mat] = mat_stats

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "_scenario_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    total = sum(s["accepted"] for ms in summary.values() for s in ms)
    print(f"\n=== DONE: {total} scenario npz files -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
