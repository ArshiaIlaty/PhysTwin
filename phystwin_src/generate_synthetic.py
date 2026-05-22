"""generate_synthetic.py — Level-2 synthetic trajectory generation.

For each donor case (longest + cleanest fit per material), drive the
calibrated spring-mass simulator with new controller-motion patterns and
harvest (positions, forces). Save accepted trajectories to
`dataset_synth_raw/<donor>__synth_<motion>_<i>.npz`.

Sanity gates per generated trajectory:
  - No NaN/Inf in positions or forces.
  - max(|F|) <= MAX_F_RATIO * donor's recorded max(|F|).
  - max(|position|) within a sane envelope (no particle flying away).

These guard against numerical blow-ups outside the regime PhysTwin was
calibrated for. Rejected trajectories are NOT saved; their rejection reason
is logged.

Motion types (parameterised by random seed):
  1. linear_push   — controller moves linearly from rest to a random offset
                     over the first half, then back to rest.
  2. sinusoidal    — controller oscillates around rest along a random axis.
  3. random_walk   — controller does a small-step random walk with a soft
                     restoring force back toward rest.
  4. hold_release  — push to a target offset, hold for half the duration,
                     then return.

All motions START from the donor's recorded controller_points[0] so the
spring rest lengths (calibrated to that initial controller pose) stay
consistent.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch


# Donor lists per material — multi-donor, all training-set cases that are
# physically stable (we EXCLUDE cases with extreme force spikes / numerical
# blow-ups in their recorded trajectory: single_push_sloth, single_lift_dinosor,
# double_lift_zebra). Each tuple is (case_name, n_ctrl_parts, cfg_type).
DONORS = {
    "rope":  [
        ("rope_double_hand",     2, "real"),
        ("single_push_rope_1",   1, "real"),
        ("single_push_rope_4",   1, "real"),
        ("single_lift_rope",     1, "real"),
        ("single_push_rope",     1, "real"),
    ],
    "cloth": [
        ("single_lift_cloth",    1, "cloth"),
        ("single_lift_cloth_4",  1, "cloth"),
        ("single_lift_cloth_3",  1, "cloth"),
        ("double_lift_cloth_3",  2, "cloth"),
        ("double_lift_cloth_1",  2, "cloth"),
        ("single_lift_cloth_1",  1, "cloth"),
        ("single_clift_cloth_3", 1, "cloth"),
        ("single_clift_cloth_1", 1, "cloth"),
    ],
    "sloth": [
        ("double_stretch_sloth", 2, "real"),
        ("single_lift_sloth",    1, "real"),
        ("double_lift_sloth",    2, "real"),
        # single_push_sloth EXCLUDED (866 kN spring blow-up)
    ],
    "toy":   [
        ("double_stretch_zebra", 2, "real"),
        ("single_lift_zebra",    1, "real"),
        # double_lift_zebra EXCLUDED (1.9 MN max force, fit numerical issue)
        # single_lift_dinosor EXCLUDED (6 MN max force)
    ],
}

MOTION_TYPES = ["linear_push", "sinusoidal", "random_walk", "hold_release"]

# Sanity gate constants:
MAX_F_RATIO = 1.5            # reject if max|F| > 1.5 × donor recorded max|F|
MAX_DISP_RATIO = 2.0         # reject if any particle moves > 2 × donor bbox diag
OUT_DIR = "dataset_synth_raw"


def _setup_trainer(case_name: str, n_ctrl_parts: int, cfg_type: str):
    """Reproduce extract_dataset.py's trainer construction."""
    # Imports kept inside the function — they pull in the full qqtt stack
    # which initializes Warp and CUDA on import.
    from qqtt import InvPhyTrainerWarp
    from qqtt.utils import cfg, logger

    cfg_file = "configs/cloth.yaml" if cfg_type == "cloth" else "configs/real.yaml"
    cfg.load_from_yaml(cfg_file)

    base_path = "./data/different_types"
    base_dir = f"./experiments/{case_name}"

    optimal_path = f"./experiments_optimization/{case_name}/optimal_params.pkl"
    with open(optimal_path, "rb") as f:
        cfg.set_optimal_params(pickle.load(f))

    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array([np.linalg.inv(c) for c in c2ws])
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        meta = json.load(f)
    cfg.intrinsics = np.array(meta["intrinsics"])
    cfg.WH = meta["WH"]

    logger.set_log_file(path="/tmp", name=f"synth_{case_name}")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )
    best_models = glob.glob(f"experiments/{case_name}/train/best_*.pth")
    assert best_models, f"No best_*.pth for {case_name}"
    return trainer, best_models[0]


def _smooth_ramp(T: int, peak_frame: int = None) -> np.ndarray:
    """Return [T] array that ramps 0→1 by `peak_frame` then 1→0. Cosine eased."""
    peak_frame = peak_frame or (T // 2)
    out = np.zeros(T, dtype=np.float32)
    for t in range(T):
        if t < peak_frame:
            out[t] = 0.5 - 0.5 * np.cos(np.pi * t / max(peak_frame, 1))
        else:
            tail = T - peak_frame
            phase = (t - peak_frame) / max(tail, 1)
            out[t] = 0.5 + 0.5 * np.cos(np.pi * phase)
    return out


def make_motion(motion_type: str, T: int, K: int, ctrl_rest: np.ndarray,
                 ctrl_extent: float, rng: np.random.RandomState,
                 amp_min: float = 0.10, amp_max: float = 0.40) -> np.ndarray:
    """Return synthetic controller trajectory [T, K, 3].

    Args:
        ctrl_rest: [K, 3] — donor's frame-0 controller positions.
        ctrl_extent: scalar — donor's controller motion bbox diagonal (m).
                     Used as a length scale for synthetic offsets.
        amp_min, amp_max: fraction of ctrl_extent to use as the offset
                          amplitude. Lowering these raises acceptance rate
                          (less spring stress); raising them broadens force
                          distribution coverage.
    """
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction) + 1e-9
    # Cap synthetic motions a bit smaller than the donor's recorded extent
    # so we stay in the regime PhysTwin was calibrated for.
    amp = ctrl_extent * rng.uniform(amp_min, amp_max)
    offsets = np.zeros((T, 3), dtype=np.float32)

    if motion_type == "linear_push":
        ramp = _smooth_ramp(T)
        offsets = direction[None, :] * (amp * ramp[:, None])

    elif motion_type == "sinusoidal":
        freq = rng.uniform(1.0, 3.0)               # cycles over the trajectory
        env = _smooth_ramp(T, peak_frame=int(T * 0.3))
        phase = np.linspace(0, 2 * np.pi * freq, T)
        offsets = direction[None, :] * (amp * (np.sin(phase) * env)[:, None])

    elif motion_type == "random_walk":
        step = (ctrl_extent * 0.04) * rng.normal(size=(T, 3)).astype(np.float32)
        # Soft restoring force toward rest
        pos = np.zeros(3, dtype=np.float32)
        for t in range(T):
            pos = pos + step[t] - 0.06 * pos
            offsets[t] = pos
        # Cap to amp to prevent runaway
        norms = np.linalg.norm(offsets, axis=1, keepdims=True) + 1e-9
        scale = np.minimum(1.0, amp / norms)
        offsets = offsets * scale

    elif motion_type == "hold_release":
        # Push to target in first 30%, hold to 70%, release to rest by 100%
        push_end = int(T * 0.3); hold_end = int(T * 0.7)
        for t in range(T):
            if t < push_end:
                a = 0.5 - 0.5 * np.cos(np.pi * t / max(push_end, 1))
            elif t < hold_end:
                a = 1.0
            else:
                phase = (t - hold_end) / max(T - hold_end, 1)
                a = 0.5 + 0.5 * np.cos(np.pi * phase)
            offsets[t] = direction * (amp * a)
    else:
        raise ValueError(motion_type)

    # Broadcast offsets across controller points (rigid hand motion)
    ctrl = ctrl_rest[None, :, :] + offsets[:, None, :]
    return ctrl.astype(np.float32)


def _validate(positions: np.ndarray, forces: np.ndarray,
              donor_max_force: float, donor_extent: float,
              donor_centroid: np.ndarray) -> tuple[bool, str]:
    if np.any(~np.isfinite(positions)) or np.any(~np.isfinite(forces)):
        return False, "NaN/Inf"
    fmax = float(np.linalg.norm(forces, axis=-1).max())
    if fmax > donor_max_force * MAX_F_RATIO:
        return False, f"spike (max|F|={fmax:.0f} vs cap {donor_max_force*MAX_F_RATIO:.0f})"
    # particle excursion check: compare max distance from donor centroid
    obj_max_excursion = float(np.linalg.norm(positions - donor_centroid[None, None, :], axis=-1).max())
    if obj_max_excursion > donor_extent * MAX_DISP_RATIO:
        return False, f"runaway (excursion={obj_max_excursion:.3f}, cap {donor_extent*MAX_DISP_RATIO:.3f})"
    return True, "ok"


def generate_for_donor(material: str, case_name: str, n_ctrl_parts: int, cfg_type: str,
                        n_per_motion: int = 6, T: int = 120, rng_seed: int = 0,
                        amp_min: float = 0.10, amp_max: float = 0.40):
    print(f"\n=== Donor: {case_name} (material={material}, n_ctrl={n_ctrl_parts}) ===")
    trainer, best_path = _setup_trainer(case_name, n_ctrl_parts, cfg_type)
    # Donor reference statistics
    orig_ctrl = trainer.simulator.controller_points.detach().cpu().numpy()  # [T_donor, K, 3]
    ctrl_rest = orig_ctrl[0]                                                 # [K, 3]
    donor_ctrl_displacements = orig_ctrl - orig_ctrl[0:1]
    donor_ctrl_extent = float(np.linalg.norm(donor_ctrl_displacements, axis=-1).max()) + 1e-6
    print(f"  donor ctrl_extent (max ‖disp‖ over recorded path): {donor_ctrl_extent:.4f} m")

    # Run the recorded trajectory once via extract_force_data to read donor max|F|.
    print("  running donor's recorded trajectory once to get reference max|F|…")
    pos_ref, f_ref, _, _ = trainer.extract_force_data(best_path, n_ctrl_parts=n_ctrl_parts)
    donor_max_force = float(np.linalg.norm(f_ref, axis=-1).max())
    donor_centroid = pos_ref.mean(axis=(0, 1))
    donor_extent = float(np.linalg.norm(pos_ref.max(axis=(0, 1)) - pos_ref.min(axis=(0, 1))))
    print(f"  donor max|F|={donor_max_force:.0f} N, bbox diag={donor_extent:.4f} m")

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.RandomState(rng_seed)
    stats = {"accepted": 0, "rejected": 0, "by_reason": {}}

    for motion in MOTION_TYPES:
        print(f"  -- motion={motion} ({n_per_motion} variations) --")
        for i in range(n_per_motion):
            ctrl_synth = make_motion(motion, T, ctrl_rest.shape[0], ctrl_rest,
                                       donor_ctrl_extent, rng,
                                       amp_min=amp_min, amp_max=amp_max)
            # Hot-swap into the simulator
            ctrl_tensor = torch.tensor(ctrl_synth, device=trainer.simulator.controller_points.device,
                                         dtype=trainer.simulator.controller_points.dtype)
            trainer.simulator.controller_points = ctrl_tensor
            trainer.controller_points = ctrl_tensor
            trainer.dataset.frame_len = T

            try:
                pos, f, ctrl_out, _meta = trainer.extract_force_data(best_path, n_ctrl_parts=n_ctrl_parts)
            except Exception as e:
                stats["rejected"] += 1
                stats["by_reason"].setdefault("exception", 0)
                stats["by_reason"]["exception"] += 1
                print(f"     [{motion}#{i}] REJECTED: exception: {type(e).__name__}: {e}")
                continue

            ok, reason = _validate(pos, f, donor_max_force, donor_extent, donor_centroid)
            if not ok:
                stats["rejected"] += 1
                stats["by_reason"].setdefault(reason.split()[0], 0)
                stats["by_reason"][reason.split()[0]] += 1
                print(f"     [{motion}#{i}] REJECTED: {reason}")
                continue

            # Build the .npz with the same schema as dataset_v2/*
            from extract_dataset import summary_features, pad_forces, _object_category, SUMMARY_FEATURE_NAMES
            X = summary_features(pos)
            y_per = pad_forces(f, max_ctrl_parts=2)
            y_net = f.sum(axis=1)

            # Compute v2's extra features inline to match dataset_v2 schema
            from augment_dataset import NEW_FEATURE_NAMES, augment_case  # noqa

            # Compute the extra features here directly (avoid re-IO)
            ctrl_centroid = ctrl_out.mean(axis=1)
            ctrl_centroid_disp = ctrl_centroid - ctrl_centroid[0:1]
            obj_centroid = pos.mean(axis=1)
            rel_motion = ctrl_centroid - obj_centroid
            diffs = pos - ctrl_centroid[:, None, :]
            obj_to_ctrl = np.linalg.norm(diffs, axis=-1)
            nearest_dist = obj_to_ctrl.min(axis=1, keepdims=True)
            mean_contact_dist = obj_to_ctrl.mean(axis=1, keepdims=True)
            vel = np.zeros_like(pos); vel[1:] = pos[1:] - pos[:-1]
            vel_mag = np.linalg.norm(vel, axis=-1)
            mean_vel_mag = vel_mag.mean(axis=1, keepdims=True)
            max_vel_mag = vel_mag.max(axis=1, keepdims=True)
            centroid_vel = vel.mean(axis=1)
            extra = np.concatenate([ctrl_centroid_disp, nearest_dist, mean_contact_dist,
                                      rel_motion, mean_vel_mag, max_vel_mag, centroid_vel],
                                     axis=1).astype(np.float32)
            X_full = np.concatenate([X, extra], axis=1).astype(np.float32)
            feat_names = list(SUMMARY_FEATURE_NAMES) + NEW_FEATURE_NAMES

            synth_name = f"{case_name}__synth_{motion}_{i}"
            out_path = os.path.join(OUT_DIR, synth_name + ".npz")
            np.savez(
                out_path,
                X=X_full,
                y_per_ctrl=y_per,
                y_net=y_net.astype(np.float32),
                positions=pos.astype(np.float32),
                controller_pos=ctrl_out.astype(np.float32),
                case_name=synth_name,
                material=cfg_type,
                object_category=_object_category(case_name),
                n_ctrl_parts=n_ctrl_parts,
                feature_names=np.array(feat_names),
                source_donor=case_name,
                motion_type=motion,
            )
            stats["accepted"] += 1
            mag = np.linalg.norm(f, axis=-1)
            print(f"     [{motion}#{i}] OK T={T} max|F|={mag.max():.0f} mean|F|={mag.mean():.0f}")

    print(f"  -> donor {case_name}: accepted={stats['accepted']}  rejected={stats['rejected']}  "
            f"({stats['by_reason']})")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--materials", type=str, default="rope,cloth,sloth,toy",
                          help="Comma-separated materials to generate for.")
    parser.add_argument("--n_per_motion", type=int, default=6,
                          help="Synthetic trajectories per motion-type per donor.")
    parser.add_argument("--T", type=int, default=120, help="Length of each synthetic trajectory.")
    parser.add_argument("--amp_min", type=float, default=0.10,
                          help="Lower bound of synthetic motion amplitude as fraction of donor ctrl extent.")
    parser.add_argument("--amp_max", type=float, default=0.40,
                          help="Upper bound of synthetic motion amplitude. Lower = higher accept rate.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    materials = [m.strip() for m in args.materials.split(",") if m.strip()]
    summary = {}
    for m in materials:
        if m not in DONORS:
            print(f"!! Unknown material {m}, skipping")
            continue
        material_summary = {"accepted": 0, "rejected": 0, "by_donor": {}}
        for d_idx, (case_name, n_ctrl_parts, cfg_type) in enumerate(DONORS[m]):
            seed_per_donor = args.seed * 1000 + d_idx
            s = generate_for_donor(m, case_name, n_ctrl_parts, cfg_type,
                                    n_per_motion=args.n_per_motion, T=args.T,
                                    rng_seed=seed_per_donor,
                                    amp_min=args.amp_min, amp_max=args.amp_max)
            material_summary["accepted"] += s["accepted"]
            material_summary["rejected"] += s["rejected"]
            material_summary["by_donor"][case_name] = s
        summary[m] = material_summary
    print("\n=== SUMMARY ===")
    for m, s in summary.items():
        print(f"  {m}: accepted={s['accepted']}  rejected={s['rejected']}  donors={list(s['by_donor'].keys())}")
    with open(os.path.join(OUT_DIR, "_synth_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
