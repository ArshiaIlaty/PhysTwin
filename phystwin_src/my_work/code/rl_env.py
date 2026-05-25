#!/usr/bin/env python3
"""
rl_env.py — Gym-like environment wrapping PhysTwin closed-loop control.

Designed for PPO with BC warm-start. The observation matches Step 2's
supervised policy input (43-dim scaled features), and the action matches
the policy's output (6-dim scaled per-group centroid Δ, unscaled by
per-material target scaler before applying to the simulator).

So BC warm-start is "load the supervised policy directly into the RL
actor's trunk+head" — no remapping needed.

For training efficiency, we focus on a single case + profile per env
instance. Training across multiple cases would multiply trainer-setup
cost (~10 s per case) by the number of cases visited.

Plan: my_work/docs/closed_loop_control/rl_plan.md
"""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# --- Path fixup: this repo has two qqtt/ trees; use phystwin_src's. ---
SCRIPT_DIR = Path(__file__).resolve().parent
MY_WORK = SCRIPT_DIR.parent
REPO_ROOT = MY_WORK.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from features import compute_31d_features  # noqa: E402
from run_closed_loop import (  # noqa: E402
    setup_trainer, infer_n_ctrl_parts_and_cfg_type,
)
from sklearn.cluster import KMeans  # noqa: E402
import warp as wp  # noqa: E402
from qqtt.utils import cfg  # noqa: E402

logger = logging.getLogger("rl_env")


class PhysTwinForceEnv:
    """Single-case force-tracking environment.

    Observation (43,): [state31, force_now_padded(6), force_goal_padded(6)]
        Returned UNSCALED. The RL trainer should apply the supervised
        feature scaler before feeding the policy (so warm-started weights
        see in-distribution inputs).

    Action (6,): per-group centroid Δ in SCALED units (same as supervised
        policy output). The env unscales using per-material target_scaler
        before applying to the simulator.

    Reward: -‖F_achieved - F_goal‖_2 / scale, where scale is a
        per-material force magnitude divisor (so rewards from rope/cloth/
        sloth are comparable when training across materials).

    Episode: T frames of one trajectory. No early termination.
    """

    def __init__(
        self,
        case_name: str,
        profile: str = "policy_ramp",
        feat_scaler: dict = None,
        target_scalers: dict = None,
        ramp_scale_range: tuple = (0.6, 1.4),
        ramp_direction: str = "recorded_mean",
        force_reward_scale: float = 10000.0,
        max_action_m: float = 0.02,
        base_path: str = "./data/different_types",
    ):
        """
        Args:
          case_name: PhysTwin case to roll out (e.g. 'single_push_rope_1')
          profile: 'policy_ramp' (synthetic ramp goal) or 'policy_recorded_goal'
          feat_scaler: dict with 'mean','std' arrays of shape (43,)
          target_scalers: dict {material: {'mean','std' arrays shape (6,)}}
          ramp_scale_range: (min, max) random multiplier for ramp peak per episode
          ramp_direction: 'x_axis' or 'recorded_mean'
          force_reward_scale: divisor in reward (units: N)
          max_action_m: |action| clip in meters/frame (safety bound)
        """
        self.case_name = case_name
        self.profile = profile
        self.feat_scaler = feat_scaler
        self.target_scalers = target_scalers
        self.ramp_scale_range = ramp_scale_range
        self.ramp_direction = ramp_direction
        self.force_reward_scale = force_reward_scale
        self.max_action_m = max_action_m

        # ---------- Trainer setup (one-time, heavy) ----------
        n_ctrl_parts, cfg_type = infer_n_ctrl_parts_and_cfg_type(case_name)
        self.n_ctrl_parts = n_ctrl_parts
        self.cfg_type = cfg_type
        logger.info("Setting up trainer for %s (n_ctrl=%d, cfg=%s)",
                    case_name, n_ctrl_parts, cfg_type)
        self.trainer, self.best_model_path = setup_trainer(
            case_name, n_ctrl_parts, cfg_type, base_path
        )

        # Load the checkpoint params into simulator (one-time)
        ckpt = torch.load(self.best_model_path, map_location=cfg.device)
        self.spring_Y = ckpt["spring_Y"]
        self.num_object_springs = ckpt["num_object_springs"]
        self.trainer.simulator.set_spring_Y(
            torch.log(self.spring_Y).detach().clone()
        )
        self.trainer.simulator.set_collide(
            ckpt["collide_elas"].detach().clone(),
            ckpt["collide_fric"].detach().clone(),
        )
        self.trainer.simulator.set_collide_object(
            ckpt["collide_object_elas"].detach().clone(),
            ckpt["collide_object_fric"].detach().clone(),
        )

        # ---------- Group controller points (mirror run_policy) ----------
        first_frame_ctrl = self.trainer.simulator.controller_points[0]
        self.K = int(first_frame_ctrl.shape[0])
        if n_ctrl_parts == 1:
            self.group_ids = np.zeros(self.K, dtype=np.int64)
            force_indexes = [torch.arange(self.K, device=cfg.device)]
        else:
            km = KMeans(n_clusters=n_ctrl_parts, random_state=0, n_init=10)
            self.group_ids = km.fit_predict(first_frame_ctrl.cpu().numpy()).astype(np.int64)
            force_indexes = [
                torch.tensor(np.where(self.group_ids == i)[0], device=cfg.device)
                for i in range(n_ctrl_parts)
            ]
        self.group_masks_torch = [
            torch.from_numpy(self.group_ids == g).to(cfg.device)
            for g in range(n_ctrl_parts)
        ]

        # ---------- Force-spring bookkeeping per group ----------
        control_springs = self.trainer.init_springs[self.num_object_springs:]
        self.force_springs, self.force_rest_lengths, self.force_spring_Y = [], [], []
        for i in range(n_ctrl_parts):
            fs, frl, fsy = [], [], []
            for j in range(len(control_springs)):
                if (control_springs[j][0] - self.trainer.num_all_points) in force_indexes[i]:
                    fs.append(control_springs[j])
                    frl.append(self.trainer.init_rest_lengths[j + self.num_object_springs])
                    fsy.append(self.spring_Y[j + self.num_object_springs])
            fs_t = torch.vstack(fs); fs_t[:, 0] -= self.trainer.num_all_points
            self.force_springs.append(fs_t)
            self.force_rest_lengths.append(torch.tensor(frl, device=cfg.device))
            self.force_spring_Y.append(torch.tensor(fsy, device=cfg.device))

        self.num_object_points = int(self.trainer.num_all_points)
        self.first_frame_ctrl = first_frame_ctrl.detach().clone()

        # Pre-load recorded data for goal construction
        v2_path = MY_WORK / "results" / "dataset_v2" / f"{case_name}.npz"
        d = np.load(v2_path, allow_pickle=True)
        self.recorded_y_per_ctrl = d["y_per_ctrl"].astype(np.float32)  # [T, 2, 3]
        self.material = str(d["object_category"])
        self.T = int(self.recorded_y_per_ctrl.shape[0])
        logger.info("Env ready: case=%s material=%s T=%d K=%d",
                    case_name, self.material, self.T, self.K)

        # Episode state
        self.frame = 0
        self.curr_target = None  # torch [K, 3]
        self.F_goal = None        # [T, 2, 3]
        self.positions_buf = None
        self.controller_buf = None
        self.forces_buf = None
        self.last_force = None

    # ---------- env API ----------

    def reset(self, seed: int = None):
        """Reset simulator to frame 0, pick a new F_goal, return initial obs."""
        if seed is not None:
            np.random.seed(seed)

        # Reset simulator state
        self.trainer.simulator.set_init_state(
            self.trainer.simulator.wp_init_vertices,
            self.trainer.simulator.wp_init_velocities,
        )

        # Build F_goal for this episode based on profile
        self.F_goal = self._make_goal()

        # Initialize buffers
        self.positions_buf = np.empty((self.T, self.num_object_points, 3), dtype=np.float32)
        self.controller_buf = np.empty((self.T, self.K, 3), dtype=np.float32)
        self.forces_buf = np.empty((self.T, self.n_ctrl_parts, 3), dtype=np.float32)

        # Frame 0
        self.curr_target = self.first_frame_ctrl.detach().clone()
        x0 = wp.to_torch(self.trainer.simulator.wp_states[0].wp_x, requires_grad=False).clone()
        self.positions_buf[0] = x0.detach().cpu().numpy()
        self.controller_buf[0] = self.curr_target.cpu().numpy()
        self.forces_buf[0] = self._compute_force(x0, self.curr_target)
        self.last_force = self.forces_buf[0]
        self.frame = 0

        return self._make_obs()

    def step(self, action_scaled: np.ndarray):
        """Apply action, advance simulator one frame, return (obs, reward, done, info).

        action_scaled: shape (6,) in SCALED action space (same as supervised policy
                       output). Will be unscaled per-material before applying.
        """
        t = self.frame + 1
        if t >= self.T:
            raise RuntimeError("step called after episode end")

        # Unscale action via per-material target scaler
        ts = self.target_scalers[self.material] if self.target_scalers else {"mean": 0, "std": 1}
        action_m = action_scaled * ts["std"] + ts["mean"]   # [6]
        action_m = action_m.reshape(2, 3).astype(np.float32)

        # Safety clip
        action_m = np.clip(action_m, -self.max_action_m, +self.max_action_m)
        # Mask unused groups
        if action_m.shape[0] > self.n_ctrl_parts:
            action_m[self.n_ctrl_parts:] = 0.0

        # Rigid-translate each group
        prev_target = self.curr_target.detach().clone()
        new_target = self.curr_target.detach().clone()
        for g in range(self.n_ctrl_parts):
            d_t = torch.from_numpy(action_m[g]).to(cfg.device, dtype=new_target.dtype)
            new_target[self.group_masks_torch[g]] += d_t
        self.curr_target = new_target

        # Step simulator
        self.trainer.simulator.set_controller_interactive(prev_target, new_target)
        if self.trainer.simulator.object_collision_flag:
            self.trainer.simulator.update_collision_graph()
        wp.capture_launch(self.trainer.simulator.forward_graph)

        x_now = wp.to_torch(
            self.trainer.simulator.wp_states[-1].wp_x, requires_grad=False
        )
        self.trainer.simulator.set_init_state(
            self.trainer.simulator.wp_states[-1].wp_x,
            self.trainer.simulator.wp_states[-1].wp_v,
        )
        torch.cuda.synchronize()

        # Record
        self.positions_buf[t] = x_now.detach().cpu().numpy()
        self.controller_buf[t] = new_target.cpu().numpy()
        force_now = self._compute_force(x_now, new_target)
        self.forces_buf[t] = force_now
        self.last_force = force_now

        # Reward: -‖F_achieved - F_goal[t]‖ / scale, summed over active groups
        goal_t = self.F_goal[t][:self.n_ctrl_parts]   # [n_ctrl, 3]
        err = np.linalg.norm(force_now - goal_t, axis=-1).sum()
        reward = float(-err / self.force_reward_scale)

        self.frame = t
        done = (t >= self.T - 1)
        info = {
            "frame": t,
            "force_achieved": force_now.copy(),
            "force_goal": goal_t.copy(),
            "ctrl_drift": float(np.linalg.norm(
                self.controller_buf[t] - self.controller_buf[0], axis=-1
            ).max()),
        }
        return self._make_obs(), reward, done, info

    def render_history(self):
        """Return the trajectory dict suitable for saving as a rollout npz."""
        return {
            "positions": self.positions_buf[:self.frame + 1],
            "controller_pos": self.controller_buf[:self.frame + 1],
            "forces": self.forces_buf[:self.frame + 1],
            "F_goal": self.F_goal[:self.frame + 1],
            "case_name": self.case_name,
            "material": self.material,
            "n_ctrl_parts": np.int8(self.n_ctrl_parts),
            "group_ids": self.group_ids.astype(np.int8),
        }

    # ---------- helpers ----------

    def _compute_force(self, x_now_torch, ctrl_torch):
        out = np.empty((self.n_ctrl_parts, 3), dtype=np.float32)
        for j in range(self.n_ctrl_parts):
            fv = self.trainer.get_force_vector(
                x_now_torch, self.force_springs[j], self.force_rest_lengths[j],
                self.force_spring_Y[j], self.trainer.num_all_points, ctrl_torch,
            )
            out[j] = fv.detach().cpu().numpy()
        return out

    def _make_goal(self):
        """Generate this episode's F_goal trajectory based on profile."""
        if self.profile == "policy_recorded_goal":
            return self.recorded_y_per_ctrl.copy()

        if self.profile == "policy_ramp":
            # Synthetic ramp 0 → peak → 0, peak magnitude randomized per episode
            mag = np.linalg.norm(self.recorded_y_per_ctrl, axis=-1)
            base_peak = float(np.percentile(mag[mag > 0], 50)) if (mag > 0).any() else 1.0
            scale = np.random.uniform(*self.ramp_scale_range)
            peak_mag = base_peak * scale
            half = self.T // 2
            ramp_up = np.linspace(0.0, peak_mag, half, dtype=np.float32)
            ramp_down = np.linspace(peak_mag, 0.0, self.T - half, dtype=np.float32)
            ramp = np.concatenate([ramp_up, ramp_down])
            F_goal = np.zeros((self.T, 2, 3), dtype=np.float32)
            # Direction
            if self.ramp_direction == "recorded_mean":
                for g in range(self.n_ctrl_parts):
                    Fg = self.recorded_y_per_ctrl[:, g, :]
                    w = np.linalg.norm(Fg, axis=-1, keepdims=True)
                    num = (Fg * w).sum(axis=0)
                    direction = num / (np.linalg.norm(num) + 1e-9)
                    F_goal[:, g, :] = ramp[:, None] * direction[None, :]
            else:
                F_goal[:, 0, 0] = ramp
            return F_goal

        raise ValueError(f"unknown profile {self.profile}")

    def _make_obs(self):
        """Build the 43-dim observation from current state."""
        positions_so_far = self.positions_buf[:self.frame + 1]
        ctrl_so_far = self.controller_buf[:self.frame + 1]
        feats = compute_31d_features(positions_so_far, ctrl_so_far)
        state31 = feats[-1]  # [31]

        force_now_pad = np.zeros((2, 3), dtype=np.float32)
        force_now_pad[:self.n_ctrl_parts] = self.last_force
        next_idx = min(self.frame + 1, self.T - 1)
        force_goal_pad = self.F_goal[next_idx]  # [2, 3]

        obs_unscaled = np.concatenate(
            [state31, force_now_pad.flatten(), force_goal_pad.flatten()]
        ).astype(np.float32)
        return obs_unscaled  # caller applies feat_scaler

    def get_scaled_obs(self, obs_unscaled):
        """Apply the supervised feature scaler to an observation."""
        if self.feat_scaler is None:
            return obs_unscaled
        return ((obs_unscaled - self.feat_scaler["mean"]) / self.feat_scaler["std"]).astype(np.float32)


class MultiCasePhysTwinEnv:
    """Wraps multiple PhysTwinForceEnv instances; samples one per reset().

    Lets a single shared policy train across cases and materials. The
    wrapper delegates step/reset/get_scaled_obs to the currently-selected
    sub-env, so the PPO training loop doesn't need to change.

    Per-material reward scaling: each sub-env is created with a material-
    appropriate `force_reward_scale`, so sloth (large forces) doesn't
    dominate the gradient over rope (small forces).
    """

    DEFAULT_MATERIAL_SCALES = {"rope": 10000.0, "cloth": 15000.0, "sloth": 30000.0}

    def __init__(self, case_specs, feat_scaler, target_scalers,
                  ramp_direction: str = "recorded_mean",
                  ramp_scale_range: tuple = (0.6, 1.4),
                  reward_scales: dict = None,
                  max_action_m: float = 0.02,
                  sample_weights: list = None):
        """
        Args:
          case_specs: list of (case_name, profile) tuples, e.g.
            [("single_push_rope_1", "policy_ramp"),
             ("double_lift_cloth_3", "policy_ramp"),
             ("double_stretch_sloth", "policy_ramp")]
          sample_weights: optional list of weights for random env selection
                          (default: uniform)
          reward_scales: per-material force scale dict; defaults to
                         {rope: 10k, cloth: 15k, sloth: 30k}
        """
        if reward_scales is None:
            reward_scales = self.DEFAULT_MATERIAL_SCALES
        self.envs = []
        self.specs = list(case_specs)
        logger.info("MultiCaseEnv: building %d sub-envs", len(self.specs))
        for case_name, profile in self.specs:
            material = self._material_from_case(case_name)
            scale = reward_scales.get(material, 10000.0)
            env = PhysTwinForceEnv(
                case_name=case_name, profile=profile,
                feat_scaler=feat_scaler, target_scalers=target_scalers,
                ramp_direction=ramp_direction,
                ramp_scale_range=ramp_scale_range,
                force_reward_scale=scale, max_action_m=max_action_m,
            )
            self.envs.append(env)
            logger.info("  loaded %s (%s) reward_scale=%.0f", case_name, material, scale)
        self.feat_scaler = feat_scaler
        self.target_scalers = target_scalers
        self.sample_weights = (np.array(sample_weights, dtype=np.float64)
                               if sample_weights is not None else None)
        if self.sample_weights is not None:
            self.sample_weights /= self.sample_weights.sum()
        self.current_env = None
        self.current_spec = None

    @staticmethod
    def _material_from_case(name: str) -> str:
        n = name.lower()
        if "rope" in n: return "rope"
        if "cloth" in n: return "cloth"
        if "sloth" in n: return "sloth"
        if "zebra" in n or "dinosor" in n: return "toy"
        return "unknown"

    def reset(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        if self.sample_weights is not None:
            idx = np.random.choice(len(self.envs), p=self.sample_weights)
        else:
            idx = np.random.randint(len(self.envs))
        self.current_env = self.envs[idx]
        self.current_spec = self.specs[idx]
        return self.current_env.reset(seed=seed)

    def step(self, action):
        return self.current_env.step(action)

    def get_scaled_obs(self, obs):
        return self.current_env.get_scaled_obs(obs)

    @property
    def T(self):
        return self.current_env.T if self.current_env is not None else self.envs[0].T


# ---------- standalone smoke test (G1) ----------

def main():
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--case_name", default="single_push_rope_1")
    p.add_argument("--profile", default="policy_ramp")
    p.add_argument("--scaler_dir", type=Path,
                   default=MY_WORK / "results" / "models_policy_fixD_targeted" / "seed_1")
    p.add_argument("--n_steps", type=int, default=10)
    args = p.parse_args()

    with open(args.scaler_dir / "feat_scaler.pkl", "rb") as f:
        feat_scaler = pickle.load(f)
    with open(args.scaler_dir / "target_scalers.pkl", "rb") as f:
        target_scalers = pickle.load(f)

    env = PhysTwinForceEnv(
        case_name=args.case_name, profile=args.profile,
        feat_scaler=feat_scaler, target_scalers=target_scalers,
    )

    print("=== G1 env smoke test ===")
    obs = env.reset(seed=0)
    print(f"reset OK. obs.shape={obs.shape}  obs range=[{obs.min():.4f}, {obs.max():.4f}]")

    total_reward = 0.0
    for i in range(args.n_steps):
        # Zero action (no motion)
        action = np.zeros(6, dtype=np.float32)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if i < 3 or i == args.n_steps - 1:
            print(f"  step {i}: r={reward:+.4f}  "
                  f"|F_a|={np.linalg.norm(info['force_achieved']):.1f}N  "
                  f"|F_g|={np.linalg.norm(info['force_goal']):.1f}N  "
                  f"done={done}")
        if done:
            break
    print(f"=== done. total_reward={total_reward:.3f}  steps={i+1} ===")


if __name__ == "__main__":
    main()
