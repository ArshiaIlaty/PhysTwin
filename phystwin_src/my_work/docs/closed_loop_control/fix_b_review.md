# Fix B review — Hindsight goal augmentation FAILED

Plan: [fix_b_plan.md](fix_b_plan.md). Negative result documented in
full for the project record.

## Status

❌ **Fix B failed at G3 (the hard gate).** Stopped before full
5-seed/100-epoch training because the failure mode is informative,
clear, and not the kind that more compute would resolve.

## Gate outcomes

| Gate | Result |
|---|---|
| G1 dataset build | PASS — 105,336 rows (3.73× baseline), k counts {1: 28k, 5: 27k, 10: 26k, 20: 24k}, 0 NaN |
| G2 single-seed × 20-epoch train | PARTIAL — 5/6 val vec_R² improved, **cloth/g0 regressed -0.16** (0.52 → 0.35) |
| **G3 ramp + replay test** | **FAIL on both axes — see below** |
| G4 full 5-seed × 100-epoch retrain | SKIPPED (G3 made it pointless) |
| G5 full Step 4 eval sweep | SKIPPED |

## G3 — the failure in numbers

### Ramp release test (the actual fix target)

| | OLD policy (Step 3 G5) | Fix B policy |
|---|---|---|
| Peak achieved force (goal 17 kN) | 9.1 kN (0.54×, undershoot) | **19.1 kN (1.13×, overshoot)** |
| End-frame achieved (goal 0 kN) | 8.7 kN | **17.8 kN** |
| Release fraction (fall / rise) | 0.04 | 0.07 |

The release problem is **NOT fixed**. The new policy still doesn't
return toward zero — and now also overshoots the peak. The failure is
qualitatively the same as the OLD policy, just shifted to bigger
numbers.

### Replay regression test (cloth, the best baseline case)

| | OLD policy (Step 4) | Fix B policy |
|---|---|---|
| `double_lift_cloth_1` err_ratio | **0.22** (excellent) | **1.01** (uncontrolled — error equals goal magnitude) |

The model regressed by **4.6×** on its strongest case. Not a small
trade-off — Fix B broke the in-distribution capability while failing
to add the out-of-distribution capability.

## The val-loss vs deployment mismatch (the real lesson)

```
epoch  OLD train  OLD val  NEW train  NEW val
   0    0.834     0.792    0.708     0.572
  10    0.255     0.444    0.364     0.394
  19    0.192     0.414    0.285     0.329
```

Fix B's val loss is **lower** (0.33 vs 0.41 at epoch 19, still
decreasing). By the standard SL metric, the augmented model is
"better."

But the closed-loop deployment metrics show the augmented model is
**much worse**. This is the classic gap between predictive accuracy
and control quality.

### Why this happens

Hindsight augmentation says "for a given `(state_t, force_now)`,
multiple different `force_goal` values share the same action label —
namely whatever the operator recorded at frame t." This trains the
model to ignore `force_goal` when predicting the action. The model
converges to an "average action" for the state, regardless of goal.

In supervised loss terms, that's fine — predicting the average
minimizes the per-row MSE across the augmented goal distribution.
That's why NEW val loss is lower (the model is doing the easier
"predict the mean" task).

But in closed-loop control, we DO want the action to depend on the
goal. Asking for force 0 when current is 8 kN should produce a
different action from asking for force 10 kN. The fix-B model lost
this discrimination — it now produces roughly the same action
regardless of the goal, which is why ramp peak overshoots (it ignores
the eventual zero goal and just keeps pushing) and cloth replay
regressed (it ignores the goal trajectory and produces some other
default trajectory).

### Why this is a fundamental finding

This is independent of training duration, augmentation magnitude, or
seed. The action labels at different `k` values for the same
`(state, force_now)` are inconsistent BY CONSTRUCTION — they're all
the same recorded action paired with different hindsight goals. The
loss function can't simultaneously satisfy all of them, so it picks
the average. No amount of additional training fixes this.

## What this tells us about the release problem

Three possibilities for why the policy can't release:
1. **Goal-distribution shift** — policy never saw "big goal jump"
   inputs. Fix B should fix.
2. **State-distribution shift** — policy never saw the states that
   arise mid-rollout when goals diverge from natural dynamics.
3. **Insufficient model capacity** — even with the right data, the
   MLP can't represent the inverse map.

Fix B's failure rules out #1. The dataset HAS the
big-goal-jump examples now, and the policy still can't release.

Most likely culprit is **#2 (state-distribution shift)**. During
recorded trajectories, the gripper position evolves smoothly — the
state at frame 30 of a hold_release is close to the state at frame 30
of any push trajectory of similar length. The policy learns a
"trajectory-natural" action mapping. Closed-loop rollouts with novel
goals produce states the model has never seen, where the predicted
action is meaningless.

The fix for #2 is on-policy data collection (DAgger), or replacing
the one-step IDM with MPC over the simulator (which doesn't need a
learned policy at all — the simulator IS the dynamics model).

## Artifacts (kept for reference)

- `results/policy_dataset_fixB.npz` — 66.8 MB, kept as the augmented
  dataset spec.
- `results/models_policy_fixB_quick/seed_0/` — failed quick policy.
- `results/eval_fixB_quick/single_push_rope_1__policy_ramp.npz` — the
  G3 ramp rollout. Use for "what failure looks like even with our
  fix attempt" if needed in the demo.

## Acceptance criteria check

From [fix_b_plan.md](fix_b_plan.md):
- [x] Augmented dataset built and validated.
- [x] Quick training run completed.
- [ ] **G3 ramp test passes** — FAILED.
- [ ] Replay regression test stays within +0.1 — FAILED (cloth went
      from 0.22 to 1.01).
- ❌ Fix B is unsuccessful per the plan's own classification.

## Carry-forward

1. **Skip the remaining Fix B gates (G4, G5).** Full retraining
   wouldn't fix the architectural mismatch above. Saving ~10 min of
   compute and avoiding misleading "look, lower val loss!" numbers in
   the demo materials.
2. **The next-best improvement attempt is MPC** (project Direction 3
   from the original options). Use PhysTwin itself as the dynamics
   model and search over short action sequences. Doesn't depend on a
   learned policy at all, so the distribution-shift problem doesn't
   apply.
3. **Optional smaller-scope follow-up before MPC**: try training with
   an EXPLICIT cross-entropy-style penalty for ignoring the goal —
   e.g. add a regularizer that encourages action variance across
   different goals at the same state. ~2 hours. Likely doesn't help
   either (the action labels themselves don't have the variance we'd
   need).
4. **The negative result is itself valuable** for the demo: lets us
   say "we attempted goal-distribution augmentation; it didn't help;
   the root cause is state-distribution shift; the principled fix is
   model-based planning rather than learned one-step inverse
   dynamics." That's a real, defensible scientific story.

## Lessons logged separately

See [lessons.md](lessons.md) for the standalone rule on
val-loss-vs-closed-loop-mismatch.
