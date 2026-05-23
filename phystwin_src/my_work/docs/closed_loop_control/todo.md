# Closed-loop control — implementation tracker

Project overview: [explanation.md](explanation.md). Step breakdown:
[tasks.md](tasks.md).

## Locked decisions (2026-05-23)
- **Architecture**: MLP first (frame-independent). Upgrade to GRU/1D-conv
  only if MLP plateaus.
- **Action representation**: per-group centroid Δ, padded to 2 groups
  (output dim = 6). Rigid-translate K points in each group at rollout.
- **Force I/O**: per-group `y_per_ctrl` (input + goal both 2×3 padded).
  Mask second group for single-control cases.
- **Split**: random-block within-case, 5 seeds, per-category scaler. Same
  as v3.1.
- **Exclusions**: `single_push_sloth` (numerical force spikes), same as
  v3.1 training.

## Step 1 — Build policy dataset ✅ done 2026-05-23
- [x] Read [step1_plan.md](step1_plan.md).
- [x] Write `my_work/code/build_policy_dataset.py`.
- [x] Run on full corpus → `results/policy_dataset.npz` (17.9 MB, 28238 rows).
- [x] Inspect: per-material, per-motion, action mag, force residual.
- [x] Write [step1_review.md](step1_review.md).

Carried into Step 2: toy outlier handling, per-category scaler, per-material
loss weighting (see step1_review.md "Carry-forward to Step 2").

## Step 2 — Train MLP policy
- [ ] Write `stepN_plan.md` for step 2.
- [ ] Write `my_work/code/train_policy.py` (mirror train_models.py).
- [ ] Multi-seed run.
- [ ] Metrics: per-axis MSE, action-magnitude R², direction cosine.

## Step 3 — Closed-loop driver
- [ ] Write step3 plan.
- [ ] Add `run_policy()` to `qqtt/engine/trainer_warp.py`.
      Document the edit in [../../notes/upstream_changes.md](../../notes/upstream_changes.md).
- [ ] CLI driver `my_work/code/run_closed_loop.py`.
- [ ] Smoke test with a random policy (shape sanity).
- [ ] Smoke test with trained policy on replay target.

## Step 4 — Eval target-force profiles
- [ ] Write step4 plan.
- [ ] Implement profile library (replay / ramp / step / sinusoid).
- [ ] Run all combinations: held-out cases × profiles.

## Step 5 — Figures + videos
- [ ] Force-tracking plots.
- [ ] Gripper-trajectory plots.
- [ ] Summary bar chart.
- [ ] Per-motion-type breakdown (rope/cloth/sloth × {real, linear_push,
      sinusoidal, random_walk, hold_release}).
- [ ] Lightweight rollout video adapter (port `visualize_force`'s render
      loop to take policy-rollout arrays).
- [ ] (Optional) Gaussian-splatting render for the demo.
- [ ] (Optional) Side-by-side stitcher: goal/achieved force curve next to
      rendered video.

## Open questions (revisit after each step)
- Does the policy need to know `n_ctrl_parts` explicitly, or does the
  padded second-group of zeros already carry that signal?
- Should the action be in world frame or relative to controller frame?
  (Default: world frame, matches how synthetic trajectories were generated.)
- For the replay test, do we feed the recorded `controller_pos[0]` exactly,
  or also let the policy choose the initial action?
  (Default: feed recorded ctrl[0] as initial condition; policy starts at t=1.)
