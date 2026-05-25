# RL plan — PPO with BC warm-start

After 4 failed goal-side improvements + 1 partial success (Fix D-targeted),
attempt the principled fix for the state-distribution-shift problem: train
a policy via reinforcement learning. Reward = closed-loop force tracking
error. The agent explores its own state distribution, naturally fixing the
shift problem that broke Fixes A-C.

**Total isolation from current results** — everything under `rl_*` prefix
in code, results, scripts, docs. The Fix D-targeted policy (current best)
stays untouched and remains the demo deliverable until/unless RL beats it.

## Approach: PPO + BC warm-start

- **Initialize** the PPO actor weights from Fix D-targeted policy (so we
  start near a working solution, not random).
- **PPO updates** fine-tune the policy via on-policy rollouts that score
  achieved-vs-goal force tracking error.
- **Same architecture** as Fix D-targeted (43→256→256→6 MLP). Add a small
  Gaussian noise head for exploration (state-independent log_std).
- **Same feature pipeline** as the supervised policy (state31 + force_now
  + force_goal, scaled by saved scalers).

## Why this could fix what Fix A-C couldn't

The fundamental issue (per lessons.md) is **state-distribution shift**:
the supervised policy never saw closed-loop-induced states at training
time, so its responses there are garbage. RL solves this by definition —
the agent generates training data from its OWN rollouts.

Specifically for the rope ramp release problem:
- Supervised policy never saw "current force = 8 kN, goal = 0 kN" inputs.
- RL agent will encounter such inputs during its own rollouts, get bad
  reward, and learn to retract.

## Folder layout (separation guarantee)

```
my_work/code/
  rl_env.py           PhysTwin → gym-like reset/step wrapper
  rl_ppo.py           PPO trainer (custom minimal, no SB3 dep)
  rl_eval.py          eval harness (reuse run_closed_loop.py mostly)

my_work/results/
  rl_runs/run_<tag>/
    checkpoints/     periodic actor/critic state_dicts
    metrics.json     per-update training metrics
    train_log.json   per-episode reward/length
    config.json      hyperparameters
  rl_eval_<tag>/     eval rollouts (same npz schema as eval_*)

my_work/scripts/slurm/
  rl_train.sbatch    one PPO run
  rl_eval.sbatch     post-training eval on the standard 14-case set

my_work/docs/closed_loop_control/
  rl_plan.md         this file
  rl_review.md       post-run writeup
```

Nothing under non-`rl_*` paths gets modified.

## Environment design (rl_env.py)

`PhysTwinForceEnv(case_name, profile, max_frames)`:

- **observation_space**: `Box(43,)` — `[state31, force_now_padded(6), force_goal_padded(6)]`
- **action_space**: `Box(6,)` — per-group centroid Δ in meters, clipped to
  ±5 cm/frame (safety bound from training-data percentile 99.9 ≤ 3 cm).
- **reset()**: loads/resets the trainer for `case_name`, sets initial
  state, picks an F_goal trajectory based on `profile`. Returns initial obs.
- **step(action)**: applies rigid translation to controller points (mirror
  `run_policy`), `set_controller_interactive`, `wp.capture_launch`,
  computes new state, computes reward.
- **reward**: `-‖F_achieved - F_goal‖ / scale_per_material`. Scale chosen
  per-material to normalize across materials (use baseline mean force
  magnitude as the divisor).
- **episode_length**: T frames (the trajectory length). No early termination.

Training scenarios:
- Pick uniformly at random from the same 11 cases × {recorded_goal, ramp}
  combinations Step 4 evaluated.
- Ramp parameters randomized per episode (peak ∈ [0.5×, 1.5×] of recorded
  median, direction = recorded_mean).

## PPO implementation (rl_ppo.py)

Minimal vanilla PPO:
- Actor: 2-layer MLP outputting mean; state-independent log_std parameter
  vector (initialized to -2.0).
- Critic: 2-layer MLP outputting scalar value, separate from actor.
- Rollout: collect 2048 steps (~20 episodes) per update.
- GAE λ=0.95, γ=0.99.
- 10 PPO epochs per update, batch size 64.
- Clip ratio 0.2, entropy coef 0.01, value coef 0.5.
- Adam: actor lr=3e-4, critic lr=1e-3.

Total budget: 200K env steps per training run ≈ 2000 episodes ≈ 30-45 min
of GPU compute (PhysTwin step ~1-5ms).

## BC warm-start

```python
# Load Fix D-targeted state_dict
fd_sd = torch.load("models_policy_fixD_targeted/seed_1/policy.pt")
# Copy into actor (matching layer names where possible)
actor.net.load_state_dict(fd_sd)
# Critic starts random (no warm-start for value function)
# Initialize log_std to -2.0 (small exploration noise)
```

The actor is now "good enough to push, ramps still fail." PPO refines.

## Verification gates

| Gate | Cost | Pass criterion |
|---|---|---|
| G1 — env smoke | <1 min | `env.reset()` works, 10 `env.step()` calls without crash, observation shape correct (43,), reward is finite and negative |
| G2 — random-policy env loop | ~1 min | 100 episodes with random actions complete; reward distribution roughly bimodal (some episodes get OK reward by luck, some terrible) |
| G3 — PPO smoke | ~5 min | 5K env steps with BC warm-start: train doesn't crash; policy gradient computed; advantage stats sensible (mean ≈ 0, std > 0) |
| G4 — short PPO run | ~10 min | 50K env steps: mean episode reward improves over training (track via train_log.json); no NaN in policy weights |
| **G5 — full PPO run** | ~45 min | **200K env steps. Pass = on the rope ramp test, the RL policy achieves release fraction > 0.5 (vs 0.04 baseline, 0.87 Fix D-targeted).** Hard gate. |
| G6 — eval sweep | ~12 min | Run the standard Step 4 14-case eval on the trained RL policy. Compare per-material to baseline / Fix D / Fix D-targeted. |

## Risks

| Risk | Symptom | Mitigation |
|---|---|---|
| PPO doesn't converge in budget | reward plateau or worse | Try 500K steps, lower lr, more rollout per update. If still no good after 2 attempts, declare unfit and pivot. |
| Reward scale dominated by one material | policy overfits to that material | Per-material reward normalization (already planned). |
| Critic learns slowly without warm-start | high-variance gradients | Pre-train critic for ~5K steps with BC actor frozen, then unfreeze. |
| Action saturation (always at ±5 cm bound) | gripper drifts off | Lower the bound to ±2 cm, or use tanh squash instead of clip. |
| Catastrophic forgetting of replay quality | reward might focus on ramp wins, replay regresses | Mix replay (recorded_goal) and ramp scenarios 50/50 in training. |
| Compute budget overrun | training takes hours | Stop after 2 hours; if rope ramp release > 0.3, ship; else declare unfit |

## Acceptance criteria

RL is **successful** if:
1. G5 ramp release > 0.87 (beats Fix D-targeted) AND replay metrics
   within +0.1 of Fix D-targeted aggregates.
2. Or: G5 ramp release > 0.5 AND clearly improves something Fix D-targeted
   couldn't (e.g., cloth ramp control).

RL is **partial** if G5 reproduces but doesn't beat Fix D-targeted —
informative but doesn't replace it as the demo policy.

RL is **unsuccessful** if PPO doesn't converge or degrades performance.
We've still learned: even RL doesn't fix this problem given our compute
budget, which strongly motivates MPC as the principled fix.

## Scope estimate

| Phase | Time |
|---|---|
| Write rl_env.py | 2 hr |
| Write rl_ppo.py (vanilla PPO) | 3 hr |
| Write rl_eval.py wrapper | 30 min |
| G1+G2 env gates | 30 min |
| G3+G4 PPO smoke gates | 1 hr (mostly compute + iteration) |
| G5 full PPO run | 45 min compute + 30 min monitoring |
| G6 eval sweep | 15 min compute + 30 min analysis |
| Write rl_review.md | 1 hr |
| **Total** | **~10 hours** of focused work, ~2 hr of compute |

This is materially less than the 3-5 days I initially estimated because:
1. We have a strong BC warm-start (skip the random-init burn-in)
2. PhysTwin is faster per-step than I assumed (~1-5ms not 30 sec)
3. We reuse the existing eval harness wholesale
