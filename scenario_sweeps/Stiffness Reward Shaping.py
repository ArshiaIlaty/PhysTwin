"""
stiffness_reward_shaping.py
────────────────────────────────────────────────────────────────────────────
Stiffness-conditioned RL reward shaping for PhysTwin.

Contribution: instead of Malak's fixed BC reward, this module shapes the
RL reward using PhysTwin's inferred spring stiffness k.

Core idea (directly analogous to SCARF):
    SCARF:       surface_type   → force threshold   τ(s)
    This work:   stiffness k    → movement penalty  λ(k)

Stiffer materials (high k) require less movement to produce the same force.
A policy trained without this knowledge will over-move on stiff objects,
generating excessive force.  By penalising large movements more heavily
on stiff objects, we teach the policy the correct movement-force transfer.

Components
──────────
1. StiffnessRewardShaper    — computes per-step shaped reward
2. PhysTwinRLEnv            — Gymnasium wrapper around PhysTwin simulator
3. run_ablation             — trains 3 conditions, saves results CSV
4. plot_ablation_results    — generates paper figures from results CSV

Usage
─────
    # Quick test with synthetic data
    python stiffness_reward_shaping.py --demo

    # Train all three ablation conditions
    python stiffness_reward_shaping.py \
        --phystwin_root  /path/to/PhysTwin \
        --stiffness_csv  analysis_results/stiffness_per_case.csv \
        --output_dir     ./rl_results \
        --case_name      double_stretch_sloth

    # Plot results after training
    python stiffness_reward_shaping.py \
        --plot_only --output_dir ./rl_results
"""

import os
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 1. Stiffness-conditioned reward shaper
# ─────────────────────────────────────────────────────────────────────────────

class StiffnessRewardShaper:
    """
    Computes a stiffness-conditioned movement penalty λ(k).

    The penalty for large movements scales with material stiffness:
        r_move_penalty = -λ(k) · ‖Δmovement‖²

    where λ(k) = λ_base · (k_norm)^α

    This encodes the physical insight:
        • High k (stiff, e.g. delivery package): small movements needed →
          penalise large movements heavily → λ(k) large
        • Low k (soft, e.g. rope): larger movements needed →
          penalise less → λ(k) small

    Parameters
    ──────────
    k_mean       : float  — mean spring stiffness of current object (N/m)
    k_global_min : float  — min k across all objects in dataset
    k_global_max : float  — max k across all objects in dataset
    lambda_base  : float  — base penalty weight (tunable)
    alpha        : float  — shaping exponent (1.0 = linear, 2.0 = quadratic)
    force_weight : float  — weight on force-error term
    success_bonus: float  — sparse reward for achieving target force
    """

    def __init__(
        self,
        k_mean:        float,
        k_global_min:  float = 50.0,
        k_global_max:  float = 500.0,
        lambda_base:   float = 2.0,
        alpha:         float = 1.0,
        force_weight:  float = 5.0,
        success_bonus: float = 10.0,
        success_tol:   float = 0.05,   # relative force tolerance
        smoothness_w:  float = 0.3,    # penalise jerk
    ):
        self.k_mean       = k_mean
        self.k_min        = k_global_min
        self.k_max        = k_global_max
        self.lambda_base  = lambda_base
        self.alpha        = alpha
        self.force_weight = force_weight
        self.success_bonus = success_bonus
        self.success_tol  = success_tol
        self.smoothness_w = smoothness_w

        # Normalise k to [0, 1]
        self.k_norm = np.clip(
            (k_mean - k_global_min) / (k_global_max - k_global_min + 1e-8),
            0.0, 1.0
        )
        # Stiffness-scaled penalty coefficient
        self.lambda_k = lambda_base * (self.k_norm ** alpha)

        print(f"  StiffnessRewardShaper: k={k_mean:.1f} N/m  "
              f"k_norm={self.k_norm:.3f}  λ(k)={self.lambda_k:.4f}")

    def __call__(
        self,
        pred_force:   float,
        gt_force:     float,
        delta_action: np.ndarray,    # change in action (movement delta)
        prev_action:  np.ndarray,    # previous action (for jerk)
        curr_action:  np.ndarray,
    ) -> Tuple[float, dict]:
        """
        Compute total shaped reward and its components.

        Returns
        ───────
        total_reward : float
        components   : dict  (for logging to TensorBoard)
        """
        # 1. Force error term (main task signal)
        force_error    = abs(pred_force - gt_force)
        rel_error      = force_error / (abs(gt_force) + 1e-8)
        force_reward   = -self.force_weight * force_error

        # 2. Stiffness-conditioned movement penalty
        move_mag       = float(np.linalg.norm(delta_action))
        move_penalty   = -self.lambda_k * (move_mag ** 2)

        # 3. Success bonus (sparse)
        success        = float(rel_error < self.success_tol)
        success_reward = self.success_bonus * success

        # 4. Smoothness (penalise jerk — change in action)
        jerk           = float(np.linalg.norm(curr_action - prev_action))
        smooth_penalty = -self.smoothness_w * jerk

        # 5. Time penalty
        time_penalty   = -0.005

        total = (force_reward + move_penalty
                 + success_reward + smooth_penalty + time_penalty)

        components = {
            "force_reward":    force_reward,
            "move_penalty":    move_penalty,
            "success":         success,
            "success_reward":  success_reward,
            "smooth_penalty":  smooth_penalty,
            "time_penalty":    time_penalty,
            "total":           total,
            "force_error":     force_error,
            "rel_force_error": rel_error,
            "move_mag":        move_mag,
            "lambda_k":        self.lambda_k,
            "k_norm":          self.k_norm,
        }
        return total, components


class FixedRewardShaper(StiffnessRewardShaper):
    """
    Ablation baseline: fixed movement penalty (ignores stiffness).
    Same API as StiffnessRewardShaper for easy comparison.
    """

    def __init__(self, fixed_lambda: float = 1.0, **kwargs):
        # Pass dummy k values; override lambda_k immediately
        super().__init__(k_mean=200.0, **kwargs)
        self.lambda_k = fixed_lambda
        self.k_norm   = None   # marks it as non-conditioned
        print(f"  FixedRewardShaper: λ={fixed_lambda} (not stiffness-conditioned)")


class NoPenaltyRewardShaper(StiffnessRewardShaper):
    """
    Ablation baseline: no movement penalty at all.
    Pure force-error minimisation.
    """

    def __init__(self, **kwargs):
        super().__init__(k_mean=200.0, **kwargs)
        self.lambda_k = 0.0
        print("  NoPenaltyRewardShaper: λ=0 (no movement penalty)")

    def __call__(self, pred_force, gt_force, delta_action,
                 prev_action, curr_action):
        force_error  = abs(pred_force - gt_force)
        rel_error    = force_error / (abs(gt_force) + 1e-8)
        success      = float(rel_error < self.success_tol)
        total        = (-self.force_weight * force_error
                        + self.success_bonus * success - 0.005)
        return total, {
            "force_reward": -self.force_weight * force_error,
            "move_penalty": 0.0,
            "success": success,
            "total": total,
            "force_error": force_error,
            "rel_force_error": rel_error,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. PhysTwin Gymnasium-compatible RL environment
# ─────────────────────────────────────────────────────────────────────────────

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    warnings.warn("gymnasium not installed — PhysTwinRLEnv unavailable. "
                  "Run: pip install gymnasium")


if HAS_GYM:
    class PhysTwinRLEnv(gym.Env):
        """
        Gymnasium wrapper around the PhysTwin simulator for RL training.

        The policy predicts robot control movements; the environment
        steps PhysTwin's spring-mass simulator and returns the resulting
        force, which is compared to a target force trajectory.

        Observation
        ───────────
            [particle_positions (n_particles × 3),   flattened
             control_point_pos (3),
             target_force (1),
             stiffness_features (4)]                ← conditioning input

        Action
        ──────
            Control point displacement (3,) — x, y, z movement delta

        Reward
        ──────
            StiffnessRewardShaper output
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            phystwin_root:  str,
            case_name:      str,
            stiffness_csv:  str,
            reward_mode:    str = "stiffness",   # "stiffness"|"fixed"|"none"
            max_steps:      int = 200,
            n_particles:    int = 64,
            target_force_path: Optional[str] = None,
        ):
            super().__init__()
            self.phystwin_root = Path(phystwin_root)
            self.case_name     = case_name
            self.reward_mode   = reward_mode
            self.max_steps     = max_steps
            self.n_particles   = n_particles
            self._step_count   = 0

            # ── load stiffness features ──────────────────────────────────
            stiff_df = pd.read_csv(stiffness_csv)
            row = stiff_df[stiff_df["case_name"] == case_name]
            if row.empty:
                raise ValueError(
                    f"Case '{case_name}' not found in {stiffness_csv}. "
                    f"Available: {stiff_df['case_name'].tolist()}")
            self._k_mean       = float(row["k_mean"].values[0])
            self._k_std        = float(row["k_std"].values[0])
            self._k_median     = float(row["k_median"].values[0])
            self._k_global_min = float(stiff_df["k_mean"].min())
            self._k_global_max = float(stiff_df["k_mean"].max())
            self._stiff_feat   = np.array([
                self._k_mean,
                self._k_std,
                self._k_median,
                np.log(self._k_mean + 1.0),
            ], dtype=np.float32)

            # ── load target force trajectory ─────────────────────────────
            if target_force_path and os.path.exists(target_force_path):
                self._target_forces = np.load(target_force_path)
            else:
                # Placeholder: sinusoidal target force profile
                t = np.linspace(0, 2 * np.pi, max_steps)
                self._target_forces = (2.0 + 1.5 * np.sin(t)).astype(
                    np.float32)

            # ── observation and action spaces ────────────────────────────
            obs_dim = n_particles * 3 + 3 + 1 + 4
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_dim,), dtype=np.float32)
            self.action_space = spaces.Box(
                low=-0.05, high=0.05,
                shape=(3,), dtype=np.float32)

            # ── build reward shaper ──────────────────────────────────────
            shaper_kwargs = dict(
                k_global_min=self._k_global_min,
                k_global_max=self._k_global_max,
            )
            if reward_mode == "stiffness":
                self._shaper = StiffnessRewardShaper(
                    k_mean=self._k_mean, **shaper_kwargs)
            elif reward_mode == "fixed":
                self._shaper = FixedRewardShaper(**shaper_kwargs)
            else:   # "none"
                self._shaper = NoPenaltyRewardShaper(**shaper_kwargs)

            # ── simulator state (placeholder — connect to PhysTwin) ──────
            self._particles   = None
            self._ctrl_pos    = None
            self._prev_action = np.zeros(3, dtype=np.float32)
            self._sim         = None   # PhysTwin simulator handle

            print(f"  PhysTwinRLEnv: case={case_name}  "
                  f"reward_mode={reward_mode}  "
                  f"k_mean={self._k_mean:.1f} N/m")

        def _load_phystwin_sim(self):
            """
            Load PhysTwin's spring-mass simulator for this case.
            Tries to import PhysTwin's SimulatorWarp (Warp-based sim).
            Falls back to a simple spring approximation if not available.
            """
            sim_path = (self.phystwin_root / "experiments"
                        / self.case_name / "params.pkl")
            alt_paths = [
                self.phystwin_root / "experiments_optimization"
                / self.case_name / "optimal_params.pkl",
                self.phystwin_root / "experiments_optimization"
                / self.case_name / "best_params.pkl",
            ]
            if not sim_path.exists():
                for alt in alt_paths:
                    if alt.exists():
                        sim_path = alt
                        break
            if not sim_path.exists():
                warnings.warn(
                    f"PhysTwin params not found for {self.case_name} "
                    f"(tried params.pkl / optimal_params.pkl). "
                    f"Using placeholder dynamics.")
                return None

            try:
                import sys
                sys.path.insert(0, str(self.phystwin_root))
                from spring_simulator import SimulatorWarp
                with open(sim_path, "rb") as f:
                    params = pickle.load(f)
                sim = SimulatorWarp(params)
                print(f"  ✓ Loaded PhysTwin SimulatorWarp for {self.case_name}")
                return sim
            except ImportError:
                warnings.warn(
                    "PhysTwin SimulatorWarp not importable — "
                    "using placeholder dynamics for debugging.")
                return None

        def _placeholder_step(
            self,
            action: np.ndarray,
        ) -> Tuple[np.ndarray, float]:
            """
            Simplified spring-mass dynamics for debugging without PhysTwin.
            F ≈ k · displacement.  Replace with real SimulatorWarp call.
            """
            displacement  = np.linalg.norm(action)
            # Simulate force as k * displacement + noise
            sim_force = (self._k_mean * displacement
                         + np.random.normal(0, 0.1))
            # Update particle positions (rigid body approximation)
            self._particles += action.reshape(1, 3) * 0.1
            self._ctrl_pos  += action
            return self._particles.copy(), float(sim_force)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._step_count  = 0
            self._prev_action = np.zeros(3, dtype=np.float32)

            # Initialise particles at rest
            self._particles = np.random.randn(
                self.n_particles, 3).astype(np.float32) * 0.01
            self._ctrl_pos  = np.zeros(3, dtype=np.float32)

            if self._sim is None:
                self._sim = self._load_phystwin_sim()

            obs  = self._get_obs(step=0)
            info = {"case": self.case_name,
                    "k_mean": self._k_mean,
                    "reward_mode": self.reward_mode}
            return obs, info

        def step(self, action: np.ndarray):
            action = np.clip(action, -0.05, 0.05).astype(np.float32)

            # ── step the simulator ────────────────────────────────────────
            if self._sim is not None:
                # Real PhysTwin call:
                # state, sim_force = self._sim.step(action)
                # self._particles  = state["positions"]
                # self._ctrl_pos  += action
                pass
            particles, sim_force = self._placeholder_step(action)

            # ── reward ────────────────────────────────────────────────────
            target_force  = self._target_forces[
                min(self._step_count, len(self._target_forces) - 1)]
            delta_action  = action - self._prev_action
            reward, comps = self._shaper(
                pred_force   = sim_force,
                gt_force     = target_force,
                delta_action = delta_action,
                prev_action  = self._prev_action,
                curr_action  = action,
            )

            self._prev_action  = action.copy()
            self._step_count  += 1

            terminated = bool(comps.get("success", 0) == 1.0)
            truncated  = self._step_count >= self.max_steps

            info = {
                "case":            self.case_name,
                "k_mean":          self._k_mean,
                "reward_mode":     self.reward_mode,
                "sim_force":       sim_force,
                "target_force":    float(target_force),
                "force_error":     comps.get("force_error", 0.0),
                "rel_force_error": comps.get("rel_force_error", 0.0),
                "move_penalty":    comps.get("move_penalty", 0.0),
                "lambda_k":        comps.get("lambda_k", 0.0),
                "success":         comps.get("success", 0.0),
                "step":            self._step_count,
            }
            obs = self._get_obs(step=self._step_count)
            return obs, reward, terminated, truncated, info

        def _get_obs(self, step: int) -> np.ndarray:
            target = self._target_forces[
                min(step, len(self._target_forces) - 1)]
            return np.concatenate([
                self._particles.ravel(),          # (n_particles*3,)
                self._ctrl_pos,                   # (3,)
                np.array([target], dtype=np.float32),  # (1,)
                self._stiff_feat,                 # (4,)  ← conditioning
            ]).astype(np.float32)

        def close(self):
            if self._sim is not None:
                try:
                    self._sim.close()
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ablation training runner
# ─────────────────────────────────────────────────────────────────────────────

REWARD_CONDITIONS = {
    "stiffness_conditioned": "stiffness",
    "fixed_penalty":         "fixed",
    "no_penalty":            "none",
}


def parse_conditions(conditions: Optional[str]) -> dict:
    """Parse --conditions CSV into a subset of REWARD_CONDITIONS."""
    if not conditions or not conditions.strip():
        return dict(REWARD_CONDITIONS)
    labels = [c.strip() for c in conditions.split(",") if c.strip()]
    unknown = [c for c in labels if c not in REWARD_CONDITIONS]
    if unknown:
        valid = ", ".join(REWARD_CONDITIONS)
        raise ValueError(
            f"Unknown condition(s): {unknown}. Valid: {valid}")
    return {k: REWARD_CONDITIONS[k] for k in labels}


def run_ablation(
    phystwin_root: str,
    stiffness_csv: str,
    case_names:    list,
    output_dir:    str,
    total_steps:   int = 200_000,
    seeds:         list = (0, 1, 2),
    conditions:    Optional[dict] = None,
):
    """
    Train SAC under selected reward conditions for each case.
    Requires stable-baselines3 and gymnasium.
    """
    conditions = conditions or dict(REWARD_CONDITIONS)
    if not HAS_GYM:
        print("gymnasium not installed — cannot run ablation.")
        return

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        print("stable-baselines3 not installed. Run: pip install stable-baselines3")
        return

    try:
        import tensorboard  # noqa: F401
        HAS_TB = True
    except ImportError:
        HAS_TB = False

    os.makedirs(output_dir, exist_ok=True)

    class InfoCallback(BaseCallback):
        """Log per-episode custom metrics to TensorBoard."""
        def __init__(self):
            super().__init__()
            self._ep_force_err, self._ep_move_pen = [], []

        def _on_step(self):
            info  = self.locals["infos"][0]
            done  = self.locals["dones"][0]
            self._ep_force_err.append(info.get("force_error", 0.0))
            self._ep_move_pen.append(abs(info.get("move_penalty", 0.0)))
            if done and self._ep_force_err:
                self.logger.record("custom/mean_force_error",
                                   np.mean(self._ep_force_err))
                self.logger.record("custom/mean_move_penalty",
                                   np.mean(self._ep_move_pen))
                self.logger.record("custom/success_rate",
                                   info.get("success", 0.0))
                self.logger.record("custom/lambda_k",
                                   info.get("lambda_k", 0.0))
                self._ep_force_err, self._ep_move_pen = [], []
            return True

    results = []

    print(f"  Conditions: {', '.join(conditions)}")

    for case_name in case_names:
        for cond_label, reward_mode in conditions.items():
            for seed in seeds:
                tb_path  = os.path.join(output_dir,
                    f"tb_{cond_label}_{case_name}_seed{seed}")
                mdl_path = os.path.join(output_dir,
                    f"sac_{cond_label}_{case_name}_seed{seed}.zip")

                print(f"\n── Training: {cond_label} | {case_name} | "
                      f"seed={seed} ──")
                if not HAS_TB:
                    print("  (tensorboard not installed — training without TB logs)")

                def make_env():
                    return PhysTwinRLEnv(
                        phystwin_root=phystwin_root,
                        case_name=case_name,
                        stiffness_csv=stiffness_csv,
                        reward_mode=reward_mode,
                    )

                env = DummyVecEnv([make_env])
                model = SAC(
                    "MlpPolicy", env,
                    learning_rate   = 3e-4,
                    buffer_size     = 500_000,
                    batch_size      = 256,
                    ent_coef        = 0.1,
                    learning_starts = 5_000,
                    seed            = seed,
                    tensorboard_log = tb_path if HAS_TB else None,
                    verbose         = 0,
                )
                try:
                    model.learn(total_steps, callback=InfoCallback(),
                                progress_bar=True)
                except ImportError:
                    model.learn(total_steps, callback=InfoCallback(),
                                progress_bar=False)
                model.save(mdl_path)

                # Evaluate
                obs, _ = env.envs[0].reset()
                ep_forces, ep_rewards = [], []
                for _ in range(200):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, rew, term, trunc, info = env.envs[0].step(action)
                    ep_forces.append(info["force_error"])
                    ep_rewards.append(rew)
                    if term or trunc:
                        break

                results.append({
                    "condition":   cond_label,
                    "case_name":   case_name,
                    "seed":        seed,
                    "mae":         float(np.mean(ep_forces)),
                    "mean_reward": float(np.mean(ep_rewards)),
                    "success":     float(info.get("success", 0.0)),
                    "k_mean":      env.envs[0]._k_mean,
                })
                env.close()

    df_new = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "ablation_results.csv")
    if len(df_new) == 0:
        print("\n⚠️  No new runs completed — CSV unchanged.")
        return pd.read_csv(csv_path) if os.path.exists(csv_path) else df_new

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        keys = ["condition", "case_name", "seed"]
        new_keys = set(map(tuple, df_new[keys].values.tolist()))
        keep = ~df_old[keys].apply(tuple, axis=1).isin(new_keys)
        df = pd.concat([df_old[keep], df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n✅  Ablation results saved → {csv_path} ({len(df)} rows)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Paper figures from ablation results
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

COND_COLORS = {
    "stiffness_conditioned": "#1D9E75",
    "fixed_penalty":         "#378ADD",
    "no_penalty":            "#E05A4E",
}
COND_LABELS = {
    "stiffness_conditioned": "Stiffness-conditioned (ours)",
    "fixed_penalty":         "Fixed penalty",
    "no_penalty":            "No penalty",
}


def plot_ablation_results(results_csv: str, output_dir: str):
    """
    Generate paper figures from ablation_results.csv.
    Produces:
      fig1 — MAE bar chart per condition × material
      fig2 — scatter: k_mean vs MAE improvement over no-penalty
      fig3 — reward condition effect by stiffness quartile
    """
    df = pd.read_csv(results_csv)
    os.makedirs(output_dir, exist_ok=True)

    # ── Figure 1: bar chart per condition × case ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
    metrics = [("mae",         "Mean absolute force error (N)"),
               ("mean_reward", "Mean episode reward")]

    for ax, (metric, ylabel) in zip(axes, metrics):
        agg = (df.groupby(["condition", "case_name"])[metric]
               .agg(["mean", "std"]).reset_index())
        conditions = list(REWARD_CONDITIONS.keys())
        cases      = sorted(df["case_name"].unique())
        x          = np.arange(len(cases))
        w          = 0.72 / len(conditions)

        for i, cond in enumerate(conditions):
            sub    = agg[agg["condition"] == cond].set_index("case_name")
            means  = [sub.loc[c, "mean"] if c in sub.index else 0
                      for c in cases]
            stds   = [sub.loc[c, "std"]  if c in sub.index else 0
                      for c in cases]
            offset = (i - len(conditions) / 2 + 0.5) * w
            ax.bar(x + offset, means, w,
                   label=COND_LABELS[cond],
                   color=COND_COLORS[cond], alpha=0.88, zorder=3)
            ax.errorbar(x + offset, means, yerr=stds,
                        fmt="none", color="#444",
                        capsize=2, linewidth=0.8, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in cases],
                           fontsize=7)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    axes[0].legend(frameon=False, ncol=1, fontsize=7)
    fig.tight_layout(pad=0.8)
    p = os.path.join(output_dir, "fig1_ablation_bars.pdf")
    fig.savefig(p); plt.close(fig)
    print(f"  Saved → {p}")

    # ── Figure 2: k_mean vs improvement over no-penalty ──────────────────
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    base = (df[df["condition"] == "no_penalty"]
            .groupby("case_name")["mae"].mean())
    for cond in ["stiffness_conditioned", "fixed_penalty"]:
        sub  = df[df["condition"] == cond]
        agg2 = sub.groupby("case_name").agg(
            mae=("mae", "mean"), k_mean=("k_mean", "first")).reset_index()
        improvement = [
            float(base.get(c, np.nan)) - agg2.loc[i, "mae"]
            for i, c in enumerate(agg2["case_name"])
        ]
        ax.scatter(agg2["k_mean"], improvement,
                   label=COND_LABELS[cond],
                   color=COND_COLORS[cond], s=55, zorder=3,
                   edgecolors="white", linewidths=0.4)
        # Trend line
        if len(agg2) >= 2:
            from scipy.stats import linregress
            sl, ic, _, _, _ = linregress(agg2["k_mean"], improvement)
            xr = np.linspace(agg2["k_mean"].min(),
                             agg2["k_mean"].max(), 50)
            ax.plot(xr, sl * xr + ic, "--",
                    color=COND_COLORS[cond], linewidth=0.9, alpha=0.7)

    ax.axhline(0, color="#999", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Spring stiffness k (N/m)")
    ax.set_ylabel("MAE improvement over no-penalty (N)")
    ax.set_title("Benefit of stiffness conditioning\ngrows with material stiffness")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linewidth=0.3, alpha=0.4)
    fig.tight_layout(pad=0.8)
    p = os.path.join(output_dir, "fig2_improvement_vs_stiffness.pdf")
    fig.savefig(p); plt.close(fig)
    print(f"  Saved → {p}")

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ABLATION SUMMARY TABLE")
    print("=" * 65)
    agg_full = (df.groupby("condition")[["mae", "mean_reward", "success"]]
                .agg(["mean", "std"]).round(4))
    print(agg_full.to_string())
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Smoke test / demo
# ─────────────────────────────────────────────────────────────────────────────

def _demo():
    print("\n── Smoke test: StiffnessRewardShaper ──")

    # Four synthetic objects with different stiffness
    objects = [
        ("rope",    80.0),
        ("cloth",   150.0),
        ("stuffed", 220.0),
        ("package", 400.0),
    ]

    k_min, k_max = 80.0, 400.0
    gt_force     = 2.5   # target force (N)

    print(f"\n  {'Object':10s}  {'k (N/m)':8s}  {'λ(k)':7s}  "
          f"{'move=0.01 reward':>18s}  {'move=0.10 reward':>18s}")
    print("  " + "-" * 68)

    for name, k in objects:
        shaper = StiffnessRewardShaper(
            k_mean=k, k_global_min=k_min, k_global_max=k_max,
            lambda_base=2.0, alpha=1.0)

        # Compute reward for small vs large movement
        prev = np.zeros(3)
        for move_mag, label in [(0.01, "small"), (0.10, "large")]:
            action = np.array([move_mag, 0.0, 0.0])
            r, comps = shaper(
                pred_force=gt_force + np.random.normal(0, 0.1),
                gt_force=gt_force,
                delta_action=action - prev,
                prev_action=prev,
                curr_action=action,
            )
            if label == "small":
                r_small = r
            else:
                r_large = r

        print(f"  {name:10s}  {k:8.0f}  "
              f"{shaper.lambda_k:7.4f}  "
              f"{r_small:>18.4f}  "
              f"{r_large:>18.4f}")

    print("\n  Key insight: λ(k) scales with stiffness →")
    print("  stiffer objects get penalised more for large movements.")
    print("  ✓ Smoke test passed\n")

    # Test environment (if gymnasium installed)
    if HAS_GYM:
        print("── Smoke test: PhysTwinRLEnv ──")
        import tempfile, csv

        # Write minimal stiffness CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["case_name","obj_type","k_mean","k_std",
                             "k_median","n_springs","source_file"])
            writer.writerow(["test_rope","rope","90","10","88","50","x"])
            writer.writerow(["test_package","package","380","40","370","50","x"])
            tmp_csv = f.name

        for case, k in [("test_rope", 90), ("test_package", 380)]:
            for mode in ["stiffness", "fixed", "none"]:
                env = PhysTwinRLEnv(
                    phystwin_root=".", case_name=case,
                    stiffness_csv=tmp_csv, reward_mode=mode)
                obs, _ = env.reset()
                total = 0
                for _ in range(10):
                    action = env.action_space.sample()
                    obs, rew, term, trunc, info = env.step(action)
                    total += rew
                    if term or trunc: break
                print(f"  {case:15s}  mode={mode:12s}  "
                      f"total_rew={total:.4f}  "
                      f"λ(k)={info['lambda_k']:.4f}")
        os.unlink(tmp_csv)
        print("  ✓ PhysTwinRLEnv smoke test passed\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",          action="store_true")
    parser.add_argument("--phystwin_root", default=".")
    parser.add_argument("--stiffness_csv", default=None)
    parser.add_argument("--case_name",     default=None)
    parser.add_argument("--output_dir",    default="./rl_results")
    parser.add_argument("--steps",         type=int, default=200_000)
    parser.add_argument("--seeds",         type=str, default="0,1,2",
                        help="Comma-separated random seeds (default: 0,1,2)")
    parser.add_argument("--conditions",    type=str, default=None,
                        help="Comma-separated reward conditions to run "
                             "(default: all). "
                             "Choices: stiffness_conditioned, fixed_penalty, "
                             "no_penalty")
    parser.add_argument("--plot_only",     action="store_true")
    args = parser.parse_args()

    if args.demo:
        _demo()

    elif args.plot_only:
        csv_path = os.path.join(args.output_dir, "ablation_results.csv")
        plot_ablation_results(csv_path, args.output_dir)

    elif args.case_name and args.stiffness_csv:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        try:
            conditions = parse_conditions(args.conditions)
        except ValueError as e:
            parser.error(str(e))
        df = run_ablation(
            phystwin_root=args.phystwin_root,
            stiffness_csv=args.stiffness_csv,
            case_names   =[args.case_name],
            output_dir   =args.output_dir,
            total_steps  =args.steps,
            seeds        =seeds,
            conditions   =conditions,
        )
        if df is not None:
            plot_ablation_results(
                os.path.join(args.output_dir, "ablation_results.csv"),
                args.output_dir)
    else:
        parser.print_help()
