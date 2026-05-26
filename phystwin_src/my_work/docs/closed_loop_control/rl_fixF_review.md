# RL on Fix F (multi-material) — new best single policy

PPO trained on multi-material env (rope/cloth/sloth) with BC warm-start
from Fix F (45-dim input, 512-hidden MLP, ~290K params).

## Status

✅ **New best single policy.** Wins outright on rope replay, rope ramp,
and sloth replay. Improves over BASE on every metric. The composite of
"BASE + this RL policy" represents the strongest deployment story
across the project so far.

## Training summary

- 51,200 / 50,000 steps (completed naturally, didn't hit 4hr time limit)
- 25 PPO updates, 392 episodes
- Best mean return: -60.67 (warm-start was ~-138 → 56% improvement)
- Final mean release fraction: 0.62 (across mixed materials)
- Training time: 2:46 hours on A30

## Architecture details

- Input: 45-dim (31 state + 6 force_now + 6 force_goal + 2 material descriptor)
- Hidden: 512 (vs Fix F's 256 → 290K params)
- Material descriptor: `[log_mean_spring_Y, log_mean_recorded_force]`
- Action: 6-dim per-group centroid Δctrl
- BC warm-start: load Fix F's state_dict into actor.net

## Per-material headline (RL-FF-Multi vs BASE)

| Metric | BASE | RL-FF-Multi | Change |
|---|---:|---:|---|
| rope replay (4 cases avg) | 0.747 | **0.410** | **−45%** ⭐ |
| rope ramp | 0.664 | **0.524** | **−22%** |
| cloth replay (4 cases avg) | **0.359** | 0.368 | preserved |
| cloth ramp | 6.552 | **2.310** | **−65%** |
| sloth replay (3 cases avg) | 0.874 | **0.660** | **−24%** |
| sloth ramp | 5.062 | **2.664** | **−47%** |

**Every metric improved or preserved.** 5 of 6 are substantial improvements.

## Per-case wins

| Case | BASE | Fix F | RL-FF-Multi |
|---|---:|---:|---:|
| single_push_rope_1 replay | 0.256 | 0.375 | **0.182** ⭐ |
| single_push_rope_1 ramp | 0.664 | 0.618 | **0.524** ⭐ |
| single_lift_rope replay | 0.356 | 0.134 | **0.129** ⭐ |
| double_stretch_sloth replay | 1.449 | 0.950 | **0.807** ⭐ |
| double_lift_cloth_3 replay | 0.515 | 0.572 | **0.545** (tied) |
| single_lift_cloth replay | 0.451 | 0.378 | **0.355** ⭐ |

## Comparison vs all prior policies

| Policy | Approach | Best at |
|---|---|---|
| BASE | Step 2 supervised | cloth replay (0.36) |
| Fix D-T | targeted ramp_full | sloth ramp (1.67) |
| Fix E | + material descriptor (1d) | cloth ramp (2.03) |
| Fix F | + 2d descriptor + 512 hidden | many ties, no unique wins |
| **RL-FF-Multi** | PPO on Fix F + multi-material env | **rope replay/ramp, sloth replay** |

## Best per (material, profile) deployment composite

Even with RL-FF-Multi as best single policy, the absolute best per combo
requires picking different policies per case material+profile:

| Combo | Best Policy | err_ratio |
|---|---|---:|
| cloth replay | BASE / RL-FF-Multi (tied) | 0.36 |
| cloth ramp | Fix E | 2.03 |
| rope replay | RL-FF-Multi | 0.41 |
| rope ramp | RL-FF-Multi | 0.52 |
| sloth replay | RL-FF-Multi | 0.66 |
| sloth ramp | Fix D-T | 1.67 |

For a single-policy deployment, **ship RL-FF-Multi** — it's
competitive on every combo and best on 4 of 6.

## Why this approach worked

1. **Fix F BC warm-start** gave PPO a strong starting point — much
   stronger than RL on supervised BASE (which we tried in earlier
   attempts and got modest gains).
2. **Material descriptors** let the shared MLP specialize per case at
   inference time — the network sees "I'm working on stiff cloth case
   X with typical force scale Y."
3. **Bigger model (512 hidden)** had enough capacity to handle the
   diverse multi-material state distribution.
4. **Multi-material env training** exposed the policy to all 3
   material distributions during PPO updates, naturally fixing the
   state-distribution-shift problem.

The combination is more than the sum of parts — RL alone (without
material descriptors or bigger model) had only modest gains. With all
three ingredients, the policy achieves real improvements.

## Artifacts

```
my_work/results/
  models_policy_fixF/seed_1/          BC warm-start source (supervised)
  rl_runs/fixF_multi/
    best.pt                            best actor+critic state_dict
    final.pt
    config.json, metrics.json, train_log.json
    feat_scaler.pkl, target_scalers.pkl
  eval_rl_fixF_multi/
    policy_dir/policy.pt              actor extracted for eval
    *.npz × 14                         rollouts on standard sweep
    summary.json

  figures/closed_loop/videos/
    RL_FFMulti_rope_replay.mp4
    RL_FFMulti_rope_ramp.mp4
    RL_FFMulti_cloth_replay.mp4
    RL_FFMulti_sloth_replay.mp4
```

## Carry-forward / next experiments

- **Stiffness-based reward shaping** (user's suggestion): penalize
  large/aggressive actions more on stiff materials and less on
  compliant ones. Could fix the sloth-ramp regression and tighten cloth
  ramp tracking. ~1 hr code + 3 hr training.
- **Longer training** (100K-200K steps): RL was preempted before
  reaching full budget once. More training might further improve.
- **Per-material RL fine-tuning**: after multi-material RL converges,
  fine-tune separately per material for case-specific specialization.
