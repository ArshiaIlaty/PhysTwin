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

from features import compute_31d_features, get_material_descriptor  # noqa: E402
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
        action_penalty_alpha: float = 0.0,
        stiffness_normalizer: float = 9.0,
        overshoot_beta: float = 0.0,
        v2_policy_dir=None,
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
          action_penalty_alpha: Fix G — coefficient for stiffness-scaled action
              penalty. reward -= alpha * exp(log_spring_Y - C) * ||action_m||^2,
              where action_m is the *unscaled* action in meters. 0 disables.
          stiffness_normalizer: Fix G — C in the exponent above. ~9 matches a
              typical rope's mean log_spring_Y, so rope penalty ≈ alpha*||a||^2,
              soft cloth gets less, stiff materials get more.
          overshoot_beta: Fix G — coefficient on the optional overshoot
              penalty: reward -= beta * sum_g max(0, ||F_achieved_g|| - ||F_goal_g||)^2.
              Force units are Newtons (no normalization), so beta should be
              very small (e.g. 1e-9). 0 disables.
        """
        self.case_name = case_name
        self.profile = profile
        self.feat_scaler = feat_scaler
        self.target_scalers = target_scalers
        self.ramp_scale_range = ramp_scale_range
        self.ramp_direction = ramp_direction
        self.force_reward_scale = force_reward_scale
        self.max_action_m = max_action_m
        self.action_penalty_alpha = float(action_penalty_alpha)
        self.stiffness_normalizer = float(stiffness_normalizer)
        self.overshoot_beta = float(overshoot_beta)

        # Policy v2 mode: obs = flat scaled [past k_past×44 | goal k_goal×6 |
        # cond 6] assembled like make_policy_callable_v2 in run_closed_loop.py.
        # The env keeps its own history + prev_action and scales internally
        # (get_scaled_obs becomes identity), so the PPO loop is unchanged.
        self.v2 = None
        if v2_policy_dir is not None:
            import json as _json
            import pickle as _pickle
            from pathlib import Path as _Path
            v2_dir = _Path(v2_policy_dir)
            with open(v2_dir / "arch.json") as f:
                arch = _json.load(f)
            with open(v2_dir / "scalers_v2.pkl", "rb") as f:
                scalers = _pickle.load(f)
            from run_closed_loop import material_vector_for_arch
            self.v2 = {
                "arch": arch, "scalers": scalers,
                "k_past": int(arch["k_past"]), "k_goal": int(arch["k_goal"]),
                "use_prev_action": bool(arch.get("use_prev_action", True)),
                "stiff": material_vector_for_arch(arch, case_name),
                # set per episode in reset() (ramp peak randomizes the goal):
                "cond_scaled": None,
            }
            self.obs_dim = (self.v2["k_past"] * int(arch["frame_dim"])
                            + self.v2["k_goal"] * int(arch["goal_dim"])
                            + int(arch["cond_dim"]))
            self._v2_history = None
            self._v2_prev_action = None

        # Material descriptor: 0-vec (Fix-A baseline), 1-vec (Fix E), 2-vec (Fix F).
        # We infer the right dim from the feat_scaler: 43=no descriptor, 44=Fix E,
        # 45=Fix F. If feat_scaler indicates a descriptor is expected, compute it.
        self.material_descriptor = None
        if self.v2 is None:
            self.obs_dim = 43
            if feat_scaler is not None:
                md_dim = int(feat_scaler["mean"].shape[0]) - 43
                if md_dim == 1:
                    from features import get_log_spring_Y_mean
                    self.material_descriptor = np.array(
                        [get_log_spring_Y_mean(case_name)], dtype=np.float32)
                elif md_dim == 2:
                    self.material_descriptor = get_material_descriptor(case_name)
                self.obs_dim = 43 + md_dim
        self.act_dim = 6

        # Fix G: per-case log spring stiffness drives the action-penalty multiplier.
        # Compute unconditionally so shaping works even without a material descriptor
        # in the obs (Fix-A / 43-dim baseline). Falls back to the normalizer if the
        # checkpoint can't be read, which neutralizes the stiffness term.
        from features import get_log_spring_Y_mean as _get_log_Y  # noqa: E402
        try:
            self.log_spring_Y_mean = float(_get_log_Y(case_name))
        except Exception as e:
            logger.warning("Fix G: could not read log_spring_Y_mean for %s (%s); "
                           "using stiffness_normalizer (penalty multiplier=1).",
                           case_name, e)
            self.log_spring_Y_mean = float(stiffness_normalizer)
        self.stiffness_multiplier = float(np.exp(
            self.log_spring_Y_mean - self.stiffness_normalizer))

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

        if self.v2 is not None:
            from collections import deque
            from policy_v2 import cmd_scale_from_goal_traj
            self._v2_history = deque(maxlen=self.v2["k_past"])
            self._v2_prev_action = np.zeros(6, dtype=np.float32)
            # cmd_scale is per-episode: the ramp profile randomizes the peak,
            # and the conditioning must describe THIS episode's command.
            cs = cmd_scale_from_goal_traj(self.F_goal, self.n_ctrl_parts)
            sc = self.v2["scalers"]["cond"]
            cond = np.concatenate([self.v2["stiff"], [cs]]).astype(np.float32)
            self.v2["cond_scaled"] = ((cond - sc["mean"]) / sc["std"]).astype(np.float32)

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
        # Policy v2: the applied (clipped+masked) action is the next frame's
        # prev_action input — same convention as make_policy_callable_v2.
        if self.v2 is not None and self.v2["use_prev_action"]:
            self._v2_prev_action = action_m.reshape(6).copy()

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
        err = float(np.linalg.norm(force_now - goal_t, axis=-1).sum())
        r_force = -err / self.force_reward_scale

        # Fix G: stiffness-scaled action penalty (action in meters, only active
        # groups). Already masked above, so unused-group rows are zero.
        r_action = 0.0
        if self.action_penalty_alpha != 0.0:
            a_sq = float(np.sum(action_m[:self.n_ctrl_parts] ** 2))
            r_action = -(self.action_penalty_alpha
                          * self.stiffness_multiplier * a_sq)

        # Fix G (optional): overshoot penalty — only penalize when achieved
        # force magnitude exceeds goal magnitude (per group).
        r_overshoot = 0.0
        if self.overshoot_beta != 0.0:
            a_mag = np.linalg.norm(force_now, axis=-1)              # [n_ctrl]
            g_mag = np.linalg.norm(goal_t, axis=-1)                 # [n_ctrl]
            excess = np.maximum(0.0, a_mag - g_mag)
            r_overshoot = -float(self.overshoot_beta * np.sum(excess ** 2))

        reward = float(r_force + r_action + r_overshoot)

        self.frame = t
        done = (t >= self.T - 1)
        info = {
            "frame": t,
            "force_achieved": force_now.copy(),
            "force_goal": goal_t.copy(),
            "ctrl_drift": float(np.linalg.norm(
                self.controller_buf[t] - self.controller_buf[0], axis=-1
            ).max()),
            "r_force": float(r_force),
            "r_action": float(r_action),
            "r_overshoot": float(r_overshoot),
            "action_sq_m": float(np.sum(action_m[:self.n_ctrl_parts] ** 2)),
            "stiffness_mult": float(self.stiffness_multiplier),
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
        """Build the observation from current state (43-dim, or v2 flat)."""
        positions_so_far = self.positions_buf[:self.frame + 1]
        ctrl_so_far = self.controller_buf[:self.frame + 1]
        feats = compute_31d_features(positions_so_far, ctrl_so_far)
        state31 = feats[-1]  # [31]

        force_now_pad = np.zeros((2, 3), dtype=np.float32)
        force_now_pad[:self.n_ctrl_parts] = self.last_force

        if self.v2 is not None:
            return self._make_obs_v2(state31, force_now_pad)

        next_idx = min(self.frame + 1, self.T - 1)
        force_goal_pad = self.F_goal[next_idx]  # [2, 3]

        parts = [state31, force_now_pad.flatten(), force_goal_pad.flatten()]
        if self.material_descriptor is not None:
            parts.append(self.material_descriptor)
        obs_unscaled = np.concatenate(parts).astype(np.float32)
        return obs_unscaled  # caller applies feat_scaler

    def _make_obs_v2(self, state31, force_now_pad):
        """Flat SCALED v2 obs [past k×44 | goal k×6 | cond 6].

        Mirrors make_policy_callable_v2: append the current raw frame to the
        history, repeat-pad the oldest entry (valid=0), preview the commanded
        goal from frame+1 (driver-call index = frame+1 in training-row terms).
        """
        v2 = self.v2
        frame43 = np.concatenate(
            [state31, force_now_pad.flatten(), self._v2_prev_action]
        ).astype(np.float32)
        self._v2_history.append(frame43)

        rows = list(self._v2_history)
        n_pad = v2["k_past"] - len(rows)
        rows = [rows[0]] * n_pad + rows
        valid = np.array([0.0] * n_pad + [1.0] * len(self._v2_history),
                         dtype=np.float32)
        fr = v2["scalers"]["frame"]
        past = (np.stack(rows) - fr["mean"]) / fr["std"]
        past = np.concatenate([past, valid[:, None]], axis=1)  # [k_past, 44]

        idx = np.clip(self.frame + 1 + np.arange(v2["k_goal"]), 0, self.T - 1)
        go = v2["scalers"]["goal"]
        goal = (self.F_goal[idx].reshape(v2["k_goal"], 6) - go["mean"]) / go["std"]

        return np.concatenate(
            [past.flatten(), goal.flatten(), v2["cond_scaled"]]
        ).astype(np.float32)

    def get_scaled_obs(self, obs_unscaled):
        """Apply the supervised feature scaler to an observation."""
        if self.v2 is not None:
            return obs_unscaled   # v2 obs is emitted already scaled
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
                  sample_weights: list = None,
                  action_penalty_alpha: float = 0.0,
                  stiffness_normalizer: float = 9.0,
                  overshoot_beta: float = 0.0,
                  v2_policy_dir=None):
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
                action_penalty_alpha=action_penalty_alpha,
                stiffness_normalizer=stiffness_normalizer,
                overshoot_beta=overshoot_beta,
                v2_policy_dir=v2_policy_dir,
            )
            self.envs.append(env)
            logger.info("  loaded %s (%s) reward_scale=%.0f  stiff_mult=%.3f",
                        case_name, material, scale, env.stiffness_multiplier)
        self.feat_scaler = feat_scaler
        self.target_scalers = target_scalers
        self.sample_weights = (np.array(sample_weights, dtype=np.float64)
                               if sample_weights is not None else None)
        if self.sample_weights is not None:
            self.sample_weights /= self.sample_weights.sum()
        self.current_env = None
        self.current_spec = None
        # All sub-envs share obs/act dims (set by feat_scaler shape)
        self.obs_dim = self.envs[0].obs_dim
        self.act_dim = self.envs[0].act_dim

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
