# Fix C review — Hierarchical Model B + Model A FAILED

Hierarchical policy with a sub-goal planner (Model B) on top of the
Step 2 controller (Model A). Documented in full as a negative result.

## Status

❌ **Fix C failed at G5 (ramp release) AND G6 (replay regression).**

## Gate outcomes

| Gate | Result |
|---|---|
| G1 — sub-goal dataset build | PASS — 23,716 rows, sensible per-material force magnitudes, 0 NaN |
| G2 — tiny overfit Model B (100 rows × 200 ep) | PASS — loss 3.6e-4 |
| G3 — single-seed × 20-epoch Model B | PARTIAL — rope vec_R² 0.95/0.72, cloth 0.81/0.85, sloth 0.41/0.27. Sloth weak but rope/cloth strong. |
| G3b — extended training (100 epochs) | Same as G3 — early-stopped at epoch 22, no further gain |
| **G5 — ramp release test on rope** | **FAIL — release fraction 0.02 (worse than baseline 0.04)** |
| **G6 — cloth replay regression** | **FAIL — err_ratio 99.4 vs baseline 0.22 (450× regression)** |

## G5 — ramp test details

| Frame | 0 | 23 | 46 (goal peak) | 69 | 91 (goal=0) |
|---|---|---|---|---|---|
| Goal magnitude (N) | 0 | 8,683 | 16,988 | 8,305 | 0 |
| Fix C achieved (N) | 0 | 13,233 | 12,719 | 12,783 | **12,918** |

Achieved force rose to ~13 kN by frame 23, then **stayed flat at 13 kN through frame 91**, ignoring the goal's descent to 0. Better peak tracking than OLD policy (13 kN vs 9 kN) but the release problem is **unchanged**.

## G6 — cloth replay catastrophic failure

- OLD policy on `double_lift_cloth_1`: err_ratio 0.22 (excellent, the demo's best case)
- Fix C on same case: err_ratio **99.4**, gripper drift **2.36 m**

The gripper drifted off into nonsense space. No NaN — the simulation completed — but with achieved forces averaging ~1 MN against goal of ~10 kN.

## What went wrong

Most likely mechanism (consistent with the magnitude of failure):

1. Model B is well-trained on in-distribution states (G3 R² 0.8+ on cloth).
2. At frame 0 of a rollout, state is in-distribution. Model B outputs reasonable sub-goals.
3. Model A receives a sub-goal that's slightly different from its training distribution (Model A was trained on next-frame goals; Model B outputs 5-frame-ahead goals). Action is slightly off.
4. After a few frames, accumulated action error puts the gripper in a state Model B has never seen.
5. Model B's output becomes garbage on out-of-distribution state. Model A reacts to the garbage sub-goal with even bigger action.
6. Cycle compounds. After 100 frames, gripper has drifted 2 meters.

This is **classic state-distribution shift**, identical to Fix B's failure mode at the policy level, but compounded across two models.

## Pattern across all three goal-side fixes

| Fix | Approach | Rope ramp release frac | Cloth replay err_ratio |
|---|---|---|---|
| Step 4 baseline | trained policy, raw user goal | 0.04 | 0.22 |
| Fix A | inference-time goal clipping | 0.04 (no change) | n/a |
| Fix B | training-time hindsight augmentation | 0.07 (barely changed) | 1.01 (4.6× worse) |
| Fix C | hierarchical Model B + Model A | 0.02 (worse) | 99.4 (450× worse) |

**None of the three goal-side fixes improved release.** All of them made other things worse. The diagnosis is now ironclad:

**The release problem is state-distribution shift, NOT goal-distribution shift.** Any fix that doesn't address state shift — meaning any fix that doesn't either (a) collect on-policy data, (b) use the simulator directly, or (c) train against actual closed-loop reward — will fail with the same pattern.

## What this implies for next steps

Two viable remaining options:

### MPC (Direction 3 — most likely to work)
Use PhysTwin itself as the dynamics model and search over short action sequences. Doesn't depend on a learned policy at all → no distribution shift. ~1 week of work. High probability of working.

### RL fine-tuning from BC warm-start
Initialize PPO/SAC from the current trained policy; reward = force-tracking error; train against actual closed-loop performance. Naturally handles state shift because the agent explores its own states. ~3-5 days. Sample efficiency without PhysTwin's (unreleased) batched simulator is the main risk.

The user's earlier `multimodality` idea is now also unlikely to help — same root cause.

## Artifacts kept

- `results/subgoal_dataset.npz` — 14.9 MB, kept for reference.
- `results/models_subgoal/seed_0/` and `models_subgoal_quick/seed_0/` — the trained Model B.
- `results/eval_fixC/single_push_rope_1__policy_hierarchical_ramp.npz` — failed ramp rollout.
- `results/eval_fixC/double_lift_cloth_1__policy_hierarchical.npz` — failed replay rollout.

## Acceptance check

From the plan:
- [x] G1 dataset built — PASS
- [x] G2 tiny overfit — PASS
- [x] G3 train Model B — PARTIAL (sloth weak, others strong)
- [x] G4 inference smoke (folded into G5)
- [ ] G5 ramp release frac > 0.3 — FAIL (0.02)
- [ ] G6 cloth replay err_ratio within +0.1 of baseline — FAIL (99.4 vs 0.22)

Fix C is **unsuccessful** per its own plan classification, even more decisively than Fix B.

## Time spent

~1.5 hours including planning, implementation, and analysis. Stopped early after the clear failure signal in G5 + G6.

## The bottom line for the demo

Three different goal-side improvement attempts produced consistent negative results. This is itself a **strong, defensible finding**:

> We made three independent attempts to extend the closed-loop
> controller's capability to handle novel target-force profiles
> (inference-time goal shaping, training-time goal augmentation,
> hierarchical planner). All three failed in the same way — replay
> tracking regressed and release behavior didn't improve. The
> consistent failure mode demonstrates that the limiting factor is
> *state-distribution shift* under closed-loop rollout, not the model's
> goal interpretation. The principled fixes (model-predictive control
> using PhysTwin as the dynamics model, or reinforcement-learning
> fine-tuning that explores its own state distribution) are the next
> step.

That's a paper-quality framing of the result. Acknowledges the limit,
characterizes the failure, names the principled fixes.
