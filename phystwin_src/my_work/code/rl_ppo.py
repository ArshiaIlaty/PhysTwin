#!/usr/bin/env python3
"""
rl_ppo.py — Minimal vanilla PPO for the PhysTwin force-tracking env.

Single-env (PhysTwin isn't easily parallelizable without the unreleased
batched simulator). Actor+critic with shared trunk option, state-independent
log_std. BC warm-start by loading a supervised policy state_dict into the
actor's net.

Plan: my_work/docs/closed_loop_control/rl_plan.md
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

SCRIPT_DIR = Path(__file__).resolve().parent
MY_WORK = SCRIPT_DIR.parent
REPO_ROOT = MY_WORK.parent
RESULTS = MY_WORK / "results"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from rl_env import PhysTwinForceEnv, MultiCasePhysTwinEnv  # noqa: E402

logger = logging.getLogger("rl_ppo")


# ----------------------------- networks ----------------------------------

class Actor(nn.Module):
    """MLP outputting action mean; state-independent log_std parameter."""
    def __init__(self, in_dim=43, hidden=256, out_dim=6, init_log_std=-2.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.log_std = nn.Parameter(torch.full((out_dim,), float(init_log_std)))

    def forward(self, obs):
        mean = self.net(obs)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def dist(self, obs):
        mean, std = self.forward(obs)
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, in_dim=43, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


# ----------------------------- rollout buffer ----------------------------

class RolloutBuffer:
    def __init__(self, n_steps, obs_dim, act_dim, device):
        self.obs = torch.zeros((n_steps, obs_dim), device=device)
        self.actions = torch.zeros((n_steps, act_dim), device=device)
        self.logprobs = torch.zeros(n_steps, device=device)
        self.rewards = torch.zeros(n_steps, device=device)
        self.dones = torch.zeros(n_steps, device=device)
        self.values = torch.zeros(n_steps, device=device)
        self.n = n_steps
        self.ptr = 0

    def add(self, obs, action, logprob, reward, done, value):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.ptr += 1

    def reset(self):
        self.ptr = 0


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, last_value=0.0):
    """Generalized advantage estimation. All inputs are 1D tensors of length N."""
    N = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(N)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


# ----------------------------- PPO update --------------------------------

def ppo_update(actor, critic, opt_a, opt_c, buf, last_value,
                gamma=0.99, lam=0.95, clip=0.2, n_epochs=10, batch_size=64,
                ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                target_kl=0.02, logratio_clip=10.0):
    adv, ret = compute_gae(buf.rewards, buf.values, buf.dones,
                             gamma=gamma, lam=lam, last_value=last_value)
    # Normalize advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N = buf.n
    idx = np.arange(N)
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [],
              "clip_frac": [], "early_stops": 0, "epochs_done": 0}
    stopped = False
    for epoch in range(n_epochs):
        if stopped:
            break
        stats["epochs_done"] = epoch + 1
        np.random.shuffle(idx)
        for start in range(0, N, batch_size):
            mb = idx[start:start + batch_size]
            mb_obs = buf.obs[mb]
            mb_act = buf.actions[mb]
            mb_old_logprob = buf.logprobs[mb]
            mb_adv = adv[mb]
            mb_ret = ret[mb]

            dist = actor.dist(mb_obs)
            new_logprob = dist.log_prob(mb_act).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            # Clip the log-ratio to prevent exp() overflow when policy drifts
            log_ratio = (new_logprob - mb_old_logprob).clamp(-logratio_clip, logratio_clip)
            ratio = log_ratio.exp()
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_pred = critic(mb_obs)
            value_loss = ((value_pred - mb_ret) ** 2).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            opt_a.zero_grad(); opt_c.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            opt_a.step(); opt_c.step()

            with torch.no_grad():
                # Use the clipped log_ratio for KL too, to keep it finite
                approx_kl = (log_ratio.exp() - 1 - log_ratio).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip).float().mean().item()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.item())
            stats["approx_kl"].append(approx_kl)
            stats["clip_frac"].append(clip_frac)

            # Early stop if KL exceeds target
            if target_kl is not None and approx_kl > 1.5 * target_kl:
                stats["early_stops"] += 1
                stopped = True
                break

    aggregated = {}
    for k, v in stats.items():
        if isinstance(v, list):
            aggregated[k] = float(np.mean(v)) if v else 0.0
        else:
            aggregated[k] = v
    return aggregated


# ----------------------------- training loop -----------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    # Load scalers from BC warm-start dir
    bc_dir = args.bc_policy_dir
    with open(bc_dir / "feat_scaler.pkl", "rb") as f:
        feat_scaler = pickle.load(f)
    with open(bc_dir / "target_scalers.pkl", "rb") as f:
        target_scalers = pickle.load(f)

    # Build env (multi-case if --case_specs given, else single-case)
    if args.case_specs:
        specs = []
        for cs in args.case_specs:
            if ":" in cs:
                case_name, profile = cs.split(":", 1)
            else:
                case_name, profile = cs, args.profile
            specs.append((case_name, profile))
        env = MultiCasePhysTwinEnv(
            case_specs=specs,
            feat_scaler=feat_scaler,
            target_scalers=target_scalers,
            ramp_direction=args.ramp_direction,
            max_action_m=args.max_action_m,
        )
        logger.info("Multi-case training across %d (case, profile) pairs", len(specs))
    else:
        env = PhysTwinForceEnv(
            case_name=args.case_name,
            profile=args.profile,
            feat_scaler=feat_scaler,
            target_scalers=target_scalers,
            force_reward_scale=args.force_reward_scale,
            ramp_direction=args.ramp_direction,
            max_action_m=args.max_action_m,
        )
    obs_dim = int(env.obs_dim); act_dim = int(env.act_dim)
    logger.info("env episode length T=%d, obs_dim=%d, act_dim=%d",
                env.T, obs_dim, act_dim)

    # Build actor + critic — sized for the env's obs_dim (43 / 44 / 45)
    actor = Actor(in_dim=obs_dim, hidden=args.hidden, out_dim=act_dim,
                   init_log_std=args.init_log_std).to(device)
    critic = Critic(in_dim=obs_dim, hidden=args.hidden).to(device)

    # BC warm-start: load supervised policy weights into actor.net
    if not args.no_warm_start:
        bc_sd = torch.load(bc_dir / "policy.pt", map_location=device, weights_only=True)
        # Supervised PolicyMLP stored as Sequential(net=...) — strip "net." prefix if present
        clean_sd = {}
        for k, v in bc_sd.items():
            new_k = k[4:] if k.startswith("net.") else k
            clean_sd[new_k] = v
        actor.net.load_state_dict(clean_sd, strict=True)
        logger.info("BC warm-start: loaded %d params from %s",
                    sum(v.numel() for v in clean_sd.values()), bc_dir)

    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    # Rollout buffer (single env)
    buf = RolloutBuffer(args.n_steps_per_update, obs_dim, act_dim, device)

    # Output dir
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    # Save scalers in run dir so eval can find them
    with open(out_dir / "feat_scaler.pkl", "wb") as f:
        pickle.dump(feat_scaler, f)
    with open(out_dir / "target_scalers.pkl", "wb") as f:
        pickle.dump(target_scalers, f)
    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()}, f, indent=2)

    # Training loop
    total_steps = 0
    update_idx = 0
    train_log = []
    episode_returns = []
    episode_lengths = []
    episode_release_fracs = []
    best_mean_ret = -float("inf")
    t0 = time.time()

    obs_un = env.reset(seed=args.seed)
    ep_return = 0.0; ep_len = 0
    ep_force_traj = []
    while total_steps < args.total_steps:
        buf.reset()
        for step in range(args.n_steps_per_update):
            obs_scaled = env.get_scaled_obs(obs_un)
            obs_t = torch.from_numpy(obs_scaled).float().to(device)
            with torch.no_grad():
                dist = actor.dist(obs_t)
                action = dist.sample()
                logprob = dist.log_prob(action).sum(-1)
                value = critic(obs_t)

            action_np = action.cpu().numpy()
            obs_un_next, reward, done, info = env.step(action_np)
            ep_return += reward
            ep_len += 1
            ep_force_traj.append(np.linalg.norm(info["force_achieved"], axis=-1).sum())

            buf.add(obs_t, action, logprob,
                     torch.tensor(reward, device=device).float(),
                     torch.tensor(float(done), device=device),
                     value)
            total_steps += 1

            obs_un = obs_un_next
            if done:
                # Compute episode release fraction
                ftraj = np.array(ep_force_traj)
                peak_i = ftraj.argmax(); rise = ftraj[peak_i] - ftraj[0]
                fall = ftraj[peak_i] - ftraj[-1]
                rel_frac = float(max(0.0, fall) / max(rise, 1e-6))
                episode_returns.append(ep_return)
                episode_lengths.append(ep_len)
                episode_release_fracs.append(rel_frac)
                ep_return = 0.0; ep_len = 0; ep_force_traj = []
                obs_un = env.reset(seed=args.seed + total_steps)

        # Bootstrap last value
        with torch.no_grad():
            obs_scaled = env.get_scaled_obs(obs_un)
            last_value = critic(torch.from_numpy(obs_scaled).float().to(device)).item()

        # PPO update — for the first few updates, freeze the actor and only
        # train the critic so it has sensible values before influencing the
        # actor. Avoids the "good BC init gets destroyed by random-critic
        # advantages" failure mode we saw in G3-smoke.
        actor_frozen = update_idx < args.critic_warmup_steps
        if actor_frozen:
            for p in actor.parameters():
                p.requires_grad = False
        else:
            for p in actor.parameters():
                p.requires_grad = True

        stats = ppo_update(
            actor, critic, opt_a, opt_c, buf, last_value,
            gamma=args.gamma, lam=args.gae_lam, clip=args.clip,
            n_epochs=args.ppo_epochs, batch_size=args.batch_size,
            ent_coef=args.ent_coef, vf_coef=args.vf_coef,
            target_kl=args.target_kl,
        )
        stats["actor_frozen"] = bool(actor_frozen)

        # Log
        recent = episode_returns[-20:] if episode_returns else [-999]
        recent_rel = episode_release_fracs[-20:] if episode_release_fracs else [0.0]
        mean_ret = float(np.mean(recent))
        mean_rel = float(np.mean(recent_rel))
        log_row = {
            "update": update_idx,
            "total_steps": total_steps,
            "mean_recent_return": mean_ret,
            "mean_recent_release_frac": mean_rel,
            "n_episodes_done": len(episode_returns),
            "log_std_mean": float(actor.log_std.mean().item()),
            "elapsed_sec": time.time() - t0,
            **stats,
        }
        train_log.append(log_row)
        logger.info(
            "update=%d steps=%d eps=%d  ret=%+.3f  release=%.2f  pi_loss=%.3f  "
            "v_loss=%.3f  ent=%.3f  kl=%.4f  log_std=%.2f  epochs=%d  %s",
            update_idx, total_steps, len(episode_returns),
            mean_ret, mean_rel,
            stats["policy_loss"], stats["value_loss"], stats["entropy"],
            stats["approx_kl"], log_row["log_std_mean"],
            stats["epochs_done"],
            "[actor frozen]" if stats.get("actor_frozen") else "",
        )

        # Checkpoint
        if update_idx % args.ckpt_every == 0:
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
                        out_dir / "checkpoints" / f"update_{update_idx:04d}.pt")
        if mean_ret > best_mean_ret and len(episode_returns) >= 5:
            best_mean_ret = mean_ret
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
                        out_dir / "best.pt")
            log_row["saved_best"] = True
        # Always save latest
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
                    out_dir / "final.pt")

        # Persist log
        with open(out_dir / "train_log.json", "w") as f:
            json.dump(train_log, f, indent=2)
        with open(out_dir / "metrics.json", "w") as f:
            json.dump({
                "total_steps": total_steps,
                "best_mean_return": best_mean_ret,
                "final_mean_return": mean_ret,
                "final_mean_release_frac": mean_rel,
                "n_updates": update_idx + 1,
                "n_episodes": len(episode_returns),
                "elapsed_sec": time.time() - t0,
            }, f, indent=2)

        update_idx += 1

    logger.info("training done. final mean_return=%.3f release=%.2f best=%+.3f (saved to %s)",
                mean_ret, mean_rel, best_mean_ret, out_dir)


# ----------------------------- CLI ---------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    # Env
    p.add_argument("--case_name", default="single_push_rope_1")
    p.add_argument("--profile", default="policy_ramp",
                   choices=["policy_ramp", "policy_recorded_goal"])
    p.add_argument("--case_specs", nargs="+", default=None,
                   help="Multi-case training. Provide as 'case_name:profile' "
                        "strings, e.g.  --case_specs "
                        "single_push_rope_1:policy_ramp "
                        "double_lift_cloth_3:policy_ramp "
                        "double_stretch_sloth:policy_ramp")
    p.add_argument("--ramp_direction", default="recorded_mean",
                   choices=["recorded_mean", "x_axis"])
    p.add_argument("--force_reward_scale", type=float, default=10000.0,
                   help="Divisor in reward (Newtons)")
    p.add_argument("--max_action_m", type=float, default=0.02,
                   help="Clip per-frame action magnitude (m)")
    # PPO
    p.add_argument("--total_steps", type=int, default=200_000)
    p.add_argument("--n_steps_per_update", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--ppo_epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--lr_actor", type=float, default=3e-5,
                   help="Lower than typical PPO (3e-4) because we warm-start "
                        "from a good supervised policy and need to avoid "
                        "destroying it in early updates.")
    p.add_argument("--lr_critic", type=float, default=3e-4)
    p.add_argument("--target_kl", type=float, default=0.02,
                   help="Early-stop PPO updates when approx KL exceeds 1.5×this.")
    p.add_argument("--ppo_epochs_warm", type=int, default=20,
                   help="If --critic_warmup_steps > 0, run this many epochs "
                        "on the critic alone first.")
    p.add_argument("--critic_warmup_steps", type=int, default=5,
                   help="Pre-train critic for N PPO updates with actor frozen "
                        "before allowing actor updates. Stabilizes early training.")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--init_log_std", type=float, default=-2.0)
    # Other
    p.add_argument("--bc_policy_dir", type=Path,
                   default=RESULTS / "models_policy_fixD_targeted" / "seed_1")
    p.add_argument("--no_warm_start", action="store_true")
    p.add_argument("--out", type=Path,
                   default=RESULTS / "rl_runs" / "default")
    p.add_argument("--ckpt_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)


if __name__ == "__main__":
    main()
