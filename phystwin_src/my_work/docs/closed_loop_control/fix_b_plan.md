# Fix B plan — Hindsight goal augmentation + retrain

Improvement work, deferred from Step 3. Reads Step 1's policy_dataset,
augments it with "hindsight" goals, retrains the policy, and re-evaluates.

## Explain it simply

**The problem.** Our current policy was trained on a very narrow
question:

> Given the current state and the force the gripper achieved in the
> *next* frame, what motion got us there?

Every training row is `(state_t, force_now=force[t], force_goal=
force[t+1], action=ctrl[t+1] − ctrl[t])`. The gap between `force_now`
and `force_goal` is always tiny — whatever happened naturally in one
frame of the recording.

At inference time we ask a *different* question:

> Given the current state and a force I want N frames from now, what
> should I do *now*?

The model never saw this question. So when we say "force_now = 8 kN,
force_goal = 0" (an 8 kN drop in one frame), it has no training
examples close to that input and outputs garbage. That's why Fix A's
goal-shaping didn't work — even when we fed small step-down goals, the
model's response was unconstrained outside its training distribution.

**The fix in one sentence.** Generate extra training rows where
`force_goal` comes from a *future frame* (not just the next one),
keeping the action label as whatever the operator actually did. Over
many trajectories, this teaches the model that even when the goal is
far away, the right one-step action is a small step in a sensible
direction.

**Why the action label is still valid.** The recorded action at
frame `t` is the first step of a trajectory that *eventually* achieves
`force[t+k]`. So labeling `(state_t, force_goal=force[t+k])` with the
recorded `action_t` is a valid inverse-dynamics label: "if you're here
and want to *eventually* reach this force, here's a good first step."
This is the same technique behind hindsight experience replay in RL —
just applied to supervised goal-conditioned imitation here.

**Why this should help release specifically.** During `hold_release`
training trajectories (we have 72 of them in `dataset_synth_raw`),
frame 30 has high force, frame 60 has near-zero force, the recorded
action at frame 30 is the start of a slow retract. Currently we train
on `(state_30, force_now=high, force_goal=barely-lower, action=
small_retract)` — tiny goal-current gap. With hindsight, we ALSO train
on `(state_30, force_now=high, force_goal=zero, action=small_retract)`.
Across hundreds of such rows, the model learns: "when current force is
high and goal force is much lower, output a small retract step."

That's the entire idea. Same model, same hyperparameters, just a wider
training distribution.

---

## Technical plan

### Goal & scope

Modify Step 1's `build_policy_dataset.py` to emit augmented rows;
retrain with Step 2's training script unchanged. Re-eval with Step 3
G5 (ramp release test) and Step 4 (full sweep). All other infrastructure
is reused.

End state: a new policy in
`results/models_policy_fixB/seed_{0..4}/` that, on G5 ramp, **achieves
force that rises AND returns toward 0** (the current policy only
rises). On Step 4 eval, replay-mode err_ratio stays at least as good
as the baseline.

### Data augmentation specifics

For each trajectory of length T in `dataset_v2/` + `dataset_synth_raw/`,
the original Step 1 emits T-1 rows (one per consecutive frame pair).
Fix B emits additional rows:

```text
for each frame t in [0, T-1):
  for k in {1, 5, 10, 20}:
    if t + k < T:
      emit row with:
        state       = X[t]
        force_now   = y_per_ctrl[t]
        force_goal  = y_per_ctrl[t + k]      # ← hindsight: future, not next-frame
        action      = ctrl_centroid[t+1] - ctrl_centroid[t]
        action_mask = original mask for this row
        material, case_name, motion_type as before
        k_lookahead = k                       # ← new field, useful for diagnostics
```

Note: `k=1` reproduces the original Step 1 row exactly. So the
augmented dataset is a superset of the original. Row count goes from
~28k to ~28k × (average rows that survive boundary) ≈ ~28k × 3.5 ≈
**~100k rows** (some rows near the end of each trajectory can't emit
the larger k values).

Decision: k ∈ {1, 5, 10, 20}. Reasoning:
- k=1 — original Step 1 behavior (preserved for compatibility).
- k=5 — "near-future" goal, half-second lookahead at 10 fps. Tests
  the model on small-but-not-trivial goal jumps.
- k=10 — "1 second lookahead." Spans the typical force-change duration.
- k=20 — "2 second lookahead." Spans most of a 50-90 frame
  trajectory. Forces the model to learn "long-horizon" goals.
- Skip k>20 because most of our trajectories are 80–120 frames; larger
  k truncates aggressively and runs into action-mismatch issues (the
  recorded action becomes a poor first-step toward goals that span
  multiple phase changes).

### Output schema

`results/policy_dataset_fixB.npz` — same fields as `policy_dataset.npz`
plus one new column `k_lookahead [R] int8`. Total ~100k rows. ~70 MB
on disk.

### Training (unchanged)

Reuse `train_policy.py` from Step 2 verbatim. Just point it at the new
dataset:

```bash
python train_policy.py \
    --data results/policy_dataset_fixB.npz \
    --seeds 0,1,2,3,4 \
    --out results/models_policy_fixB
```

Same architecture (43-input 256-hidden 6-output MLP), same per-category
target scaler, same random-block split, same 5 seeds, same 100 epochs.

Expected training time: ~4× the Step 2 run (~80 sec total) because the
dataset is ~4× larger. Still trivial — minutes.

### Verification gates

| Gate | What | Pass criterion |
|---|---|---|
| G1 | Build augmented dataset, sanity-check counts/NaN/k distribution | Row count ≈ 4× baseline; counts per `k_lookahead` value approximately equal (some attrition at large k near end-of-trajectory); same 8 validation checks from Step 1 pass |
| G2 | Train 1 seed × 20 epochs on the augmented data | Per-material val vec_R² ≥ Step 2 baseline (rope ≥0.6, cloth ≥0.4, sloth ≥0.4). If WORSE, the augmentation is hurting; abort. |
| G3 | Re-run Step 3 G5 ramp test with the new policy | Achieved force on rope ramp now goes UP AND BACK DOWN. Specifically: peak achieved ≥ 50% of peak goal, AND end-frame achieved < 50% of peak achieved. Either condition fails → Fix B didn't fix release. |
| G4 | Train full 5 seeds × 100 epochs | All seeds finish; per-material val vec_R² mean within ±0.1 of Step 2 baseline (no regression on the original objective) |
| G5 | Re-run Step 4 full eval sweep with the new policy | Per-material replay err_ratio ≤ Step 4 baseline + 0.1 (allow small regression for the release-capability gain); ramp cases: cloth + sloth err_ratio strictly < Step 4 baseline (this is the main improvement target) |

### Risks

| Risk | Symptom | Fix |
|---|---|---|
| Hindsight labels are wrong for long-k (multiple phase changes within k frames) | Val R² drops more than 0.1 from baseline | Cap k at smaller value (e.g. drop k=20, keep k ∈ {1, 5, 10}). G2 catches this. |
| Augmented dataset overweights certain motion types (synth has more frames than real → more hindsight rows) | Val R² uneven across materials | Optional: balance by trajectory rather than by row when computing the scaler. Not implementing in v1. |
| New policy regresses on replay-mode tracking (the original goal) | G5 shows err_ratio worse than Step 4 baseline | Try training a 2-headed model: one head for "next-frame goal" (original), one for "hindsight goal" (augmented). Out of scope for v1. |
| Per-frame compute slows because dataset is 4× larger | Training is too slow to iterate | At 100k rows and a 77k-param model on CPU, this is still ~5 min. Not a concern. |
| Hindsight labels for cloth/sloth ramps still don't help because the issue is compounding error, not training distribution | G5 fails | Document and move to MPC (the deeper fix). Fix B is the cheaper attempt; if it doesn't work we know the issue is deeper. |

### Scope estimate

| Phase | Time |
|---|---|
| Modify `build_policy_dataset.py` to emit augmented rows (~30 LOC change) | 30 min |
| G1 dataset rebuild + sanity check | 5 min compute + 10 min review |
| G2 single-seed training (20 epochs) | 5 min |
| G3 ramp test (re-render G5-style rollout with new policy) | 5 min |
| G4 full 5-seed retrain | 5 min |
| G5 full Step 4 eval sweep | 15 min compute |
| Write fix_b_review.md | 30 min |
| **Total** | **~2 hours** |

### Acceptance criteria

Fix B is "successful" when:

1. `results/policy_dataset_fixB.npz` exists with ~100k rows, k-lookahead
   counts roughly balanced.
2. `results/models_policy_fixB/seed_{0..4}/` exist with checkpoints +
   scalers + metrics.
3. **G3 ramp test passes**: on rope ramp, achieved force rises AND
   returns toward 0 (the actual fix target).
4. **G5 eval sweep**: replay-mode metrics don't regress more than 0.1
   from Step 4 baseline; cloth + sloth ramp err_ratio improves
   noticeably (say, < 2.0 vs the current 5–6.5).

Fix B is "partially successful" if G3 passes but G5 shows regression on
replay — we have to choose between release capability and tracking
quality. Document and let the user pick.

Fix B is "unsuccessful" if G3 fails — meaning even with hindsight
augmentation the policy doesn't learn release. That's strong evidence
the issue is compounding error / distribution shift in state space, not
goal-space, and the right fix is MPC (the more expensive option).

### What's NOT in scope for Fix B

- Modifying the architecture (still a frame-independent MLP).
- Adding state history / temporal context (would be Fix C, separate
  iteration).
- Multi-step action prediction (also out of scope).
- MPC over the simulator (entirely different approach; reserved for
  after Fix B's outcome).
- Hyperparameter tuning of the k values beyond {1, 5, 10, 20}.
