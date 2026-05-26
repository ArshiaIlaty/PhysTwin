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

## Step 2 — Train MLP policy ✅ done 2026-05-23
- [x] Plan: [step2_plan.md](step2_plan.md).
- [x] `my_work/code/train_policy.py` (~530 LOC, supports all 5 gates as CLI flags).
- [x] G1 dataloader smoke — PASS in <1 s.
- [x] G2 tiny overfit — PASS, final loss 5.5e-4 (threshold 1e-3).
- [x] G3 single seed × 20 epochs — PASS, val vec_R² ≥ +0.47 every group.
- [x] G4 inspect predictions — PASS + 2 findings (F1, F2) logged.
- [x] G5 full 5 seeds × 100 epochs — PASS, all val vec_R² ≥ +0.54.
- [x] [step2_review.md](step2_review.md) with full table.
- [x] 3 lessons logged in [lessons.md](lessons.md) (material/cfg_type
      confusion, input-dim arithmetic, masked outputs are unconstrained).

**Carry-forward to Step 3**: Closed-loop driver MUST multiply policy
outputs by `n_ctrl_parts`-derived mask before applying to simulator (F2
finding). See step2_review.md "Carry-forward to Step 3" section for the
full per-step inference contract.

## Step 3 — Closed-loop driver ✅ done 2026-05-23
- [x] Plan: [step3_plan.md](step3_plan.md).
- [x] Shared `my_work/code/features.py` (self-test bit-exact vs stored
      dataset_v2).
- [x] Added `run_policy()` to `phystwin_src/qqtt/engine/trainer_warp.py`
      (~200 LOC). Documented in
      [../../notes/upstream_changes.md](../../notes/upstream_changes.md).
- [x] CLI driver `my_work/code/run_closed_loop.py` (5 profiles).
- [x] SLURM wrapper `my_work/scripts/slurm/closed_loop_gates.sbatch`.
- [x] G1 random — PASS (bounded, no NaN).
- [x] G2 zero — PASS (ctrl drift 0.00 m exactly).
- [x] **G3 replay** — PASS with characterization. Position error
      median 1.5 mm / max 8.1 mm vs `extract_force_data`. Root cause:
      recorded motion is NOT rigid per group (~1.4 mm/frame
      intra-group deviation, see lessons).
- [x] G4 trained policy / recorded F_goal (rope) — PASS, 25.6% force
      error (under 50% threshold).
- [x] G4b same on cloth — PASS, 54.4% force error, double-control path
      works.
- [x] G5 ramp — PASS (sim completes); partial finding: policy rises but
      doesn't release.
- [x] [step3_review.md](step3_review.md).
- [x] 2 new lessons in [lessons.md](lessons.md) (dual-qqtt, non-rigid
      recorded motion).

**Carry-forward to Step 4** (post Fix A): rigid-replay floor is ~6 mm/
ctrl-pt; replay-mode tracking is the headline metric (rope 25.6%
unshaped); ramp/step/sinusoid are limitation characterization only.
See step3_review.md addendum + lessons.md for the Fix A finding.

### Deferred to "if time permits"
- Fix B (retrain with hindsight goal augmentation) — likely won't help
  rope (training labels ambiguous) but might improve cloth/sloth.
  ~4 hours.
- MPC over the simulator (project Direction 3) — the real fix for the
  release / overshoot issue. ~1-2 weeks. Out of scope until Step 5
  ships.

## Step 4 — Eval target-force profiles ✅ done 2026-05-23
- [x] Plan: [step4_plan.md](step4_plan.md).
- [x] `my_work/code/eval_closed_loop.py` (subprocess wrapper + aggregator).
- [x] `my_work/scripts/slurm/eval_closed_loop.sbatch` (GATE env-selector).
- [x] G1 single-case smoke — PASS (err_ratio 0.256 reproduces Step 3 G4).
- [x] G2 per-material smoke — PASS (all 3 materials run cleanly).
- [x] G3 full sweep — PASS (14/14 rollouts, 0 NaN).
- [x] [step4_review.md](step4_review.md) with per-material replay table
      + ramp failure characterization.
- No new lessons (behaviors matched Step 3 carry-forward).

**Headline numbers**: cloth replay 0.36 ± 0.13 (4/4 controlled);
rope 0.39 across 3 controlled cases + 1 outlier; sloth 0.59 across 2 of
3 controlled cases. Ramp: rope borderline (0.66), cloth + sloth
uncontrolled. **Carry-forward to Step 5**: 9 controlled cases + 3
limitation cases ready for figures. Best video candidates:
double_lift_cloth_1 (0.22) and single_lift_sloth (0.29).

## Step 5 — Preliminary figures ✅ done 2026-05-23
- [x] Plan: [step5_plan.md](step5_plan.md).
- [x] `my_work/code/make_closed_loop_figures.py` — 5 PNGs from eval npzs.
- [x] 01 per-material bar chart.
- [x] 02 force-tracking grid (9 controlled replay cases).
- [x] 03 limitations panel (3 failure modes).
- [x] 04 highlight: double_lift_cloth_1 per-axis decomposition.
- [x] 05 ramp comparison (rope borderline, cloth + sloth blow up).
- [x] [step5_review.md](step5_review.md) with per-figure demo lines.

**Videos done 2026-05-23** — best of each material + 1 limitation:
- [x] `my_work/code/render_rollout_video.py` — matplotlib 3D animation
      + side-by-side force tracking plot, mp4 via cv2/mp4v (no ffmpeg).
- [x] Layout fix: explicit `fig.add_axes` positions (no panel overlap).
- [x] 3D legend (object particles / gripper 1 / gripper 2 / achieved /
      goal arrows) for self-documenting visuals.
- [x] `double_lift_cloth_1.mp4` (cloth, err 0.22) — 116f, 7.7 s, 2.08 MB.
- [x] `single_push_rope_1.mp4` (rope, err 0.26) — 92f, 6.1 s, 1.99 MB.
- [x] `single_lift_sloth.mp4` (sloth, err 0.29) — 85f, 5.7 s, 1.70 MB.
- [x] `double_lift_cloth_3__ramp_FAILURE.mp4` (cloth ramp blow-up,
      err 6.55) — 118f, 7.9 s, 1.91 MB.

**Side-finding logged**: PhysTwin's "controller" is a 30-particle
cluster (tracked hand pixels from RGB-D video), not a single point.
Documented in step5_review.md.

**Deferred (next iteration if needed)**:
- Additional rollout videos for the limitations panel
  (single_push_rope_4 outlier, double_stretch_sloth long-drift).
- Gaussian-splatting photorealistic render.

## Improvements pass (post-preliminary, per user direction)

### Fix B — Hindsight goal augmentation + retrain ❌ FAILED 2026-05-23
- [x] Plan: [fix_b_plan.md](fix_b_plan.md).
- [x] Modified `build_policy_dataset.py` (added `--k_lookaheads` flag).
- [x] G1 — built `policy_dataset_fixB.npz` (105k rows, k counts balanced).
- [x] G2 — 1-seed × 20-epoch trained; 5/6 val R² improved, cloth/g0
      regressed -0.16.
- [x] **G3 — ramp test FAILED.** Release fraction 0.07 (unchanged from
      0.04). NEW peak achieves 19 kN vs OLD's 9 kN — overshoots
      instead of undershoots. Cloth replay regressed 0.22 → 1.01.
- [ ] G4 — skipped (G3 failed, more training won't help).
- [ ] G5 — skipped.
- [x] [fix_b_review.md](fix_b_review.md) with full negative-result
      writeup + diagnosis.
- [x] Lesson logged: val loss ≠ closed-loop performance for one-step
      inverse-dynamics with goal augmentation.

**Key takeaway**: The release problem is state-distribution shift,
not goal-distribution shift. MPC over the simulator is the next
principled attempt.

### Fix C — Hierarchical Model B (sub-goal) + Model A (controller) ❌ FAILED 2026-05-23
- [x] Built `build_subgoal_dataset.py` and `subgoal_dataset.npz` (23k rows).
- [x] Trained Model B: rope vec_R² 0.95/0.72, cloth 0.81/0.85, sloth 0.41/0.27.
- [x] Added `policy_hierarchical` and `policy_hierarchical_ramp` profiles to
      `run_closed_loop.py`.
- [x] G5 ramp test FAILED — release fraction 0.02 (vs baseline 0.04).
- [x] G6 replay regression FAILED — cloth err_ratio 99.4 vs baseline 0.22
      (450× regression, 2.4 m gripper drift).
- [x] Written [fix_c_review.md](fix_c_review.md).
- [x] Logged lesson: three goal-side fixes all failed → release is state-shift.

### Fix D — Targeted synthetic ramp_full data ✅ PARTIAL WIN 2026-05-24
- [x] Added `ramp_full` motion family to `generate_synthetic.py` + `--motion_types`
      filter flag.
- [x] Generated 89 ramp_full trajectories (rope 27, cloth 46, sloth 16) on
      GPU sbatch.
- [x] Rebuilt `policy_dataset_fixD.npz` — 38,829 rows (+10,591 from ramp_full).
- [x] Retrained policy (seed 1, 100 epochs) on Fix D dataset.
- [x] G4 rope ramp test: release fraction 0.04 → 0.57 (x_axis)
      → **0.84 (recorded_mean)**. ✅ FIXED.
- [x] G6 full Step 4 sweep on Fix D policy: mixed results — cloth replay
      regressed (some cases catastrophically), rope/sloth replay improved
      or stable, ramps generally better.
- [x] Side-fix: added `--ramp_direction {x_axis,recorded_mean}` flag.
      `recorded_mean` is much more meaningful for demo videos.
- [x] Re-rendered rope ramp video with `recorded_mean` direction.
- [x] Written [fix_d_review.md](fix_d_review.md).

### Fix D-targeted — Cloth ramp_full removed, rope+sloth kept ✅ best policy 2026-05-24
- [x] Moved 46 cloth ramp_full files to
      `dataset_synth_raw_cloth_ramp_quarantine/`.
- [x] Rebuilt `policy_dataset_fixD_targeted.npz` (281 trajs, 33,355 rows).
- [x] Retrained `models_policy_fixD_targeted/seed_1`.
- [x] Re-eval: rope ramp release 0.87 (slightly better than Fix D 0.84,
      vs baseline 0.04).
- [x] Full sweep: cloth replay 1.77 → 1.00 (still regressed vs baseline 0.36
      but recovered significantly from Fix D); sloth REPLAY improves
      vs baseline 0.87 → 0.74; sloth ramp 5.06 → 1.67 (3× improvement).
- [x] Re-rendered 3 demo videos with `recorded_mean` direction:
      `fixDtargeted_rope_ramp.mp4` / `_cloth_replay.mp4` / `_sloth_replay.mp4`.
- [x] [fix_d_review.md](fix_d_review.md) addendum written.

**Fix D-targeted = best single policy for the demo.** Wins 4 of 6
(material, profile) combinations vs baseline + Fix D.

### RL — PPO with BC warm-start ✅ MODEST WIN 2026-05-24
- [x] Plan: [rl_plan.md](rl_plan.md).
- [x] Wrote `rl_env.py` (~330 LOC PhysTwin gym-like wrapper).
- [x] Wrote `rl_ppo.py` (~350 LOC minimal PPO from scratch, no SB3 dep).
- [x] Wrote `rl_eval.py` (~70 LOC bridge to existing eval harness).
- [x] G1 env smoke PASS.
- [x] G3 PPO smoke FAILED (KL inf, policy destroyed) → fixed in G3b
      with critic warmup + lr_actor 3e-5 + target_kl 0.02 + log-ratio clip.
- [x] G5 main run (53K/100K steps, preempted): release 0.89 on rope ramp.
- [x] G6 eval sweep: RL beats or ties Fix D-targeted on 11/14 cases.
      Big rope improvement (0.86 → 0.61). single_push_rope_4 outlier
      improved 40%. Cloth/sloth basically same.
- [x] [rl_review.md](rl_review.md) written.
- [x] RL_rope_ramp.mp4 rendered.
- [x] Lesson logged: PPO + BC warm-start requires critic warmup.

**RL is the new best single policy.** Improvements modest but
real, especially on rope. State-distribution-shift diagnosis validated.

### Fix E — Material descriptor (mean log spring_Y) ✅ 2026-05-25
- [x] Added `get_log_spring_Y_mean` to features.py.
- [x] Modified build_policy_dataset.py to include 1-dim material_descriptor.
- [x] Trained Fix E (43+1=44 dim, hidden=256).
- [x] Eval: cloth ramp 6.55 → 2.03 (huge win!); cloth replay recovered
      from Fix D-T's regression (1.00 → 0.37).
- [x] Lost: single_push_rope_4 outlier got worse (1.84 → 4.04).

### Fix F — 2-vec descriptor + 512 hidden ✅ 2026-05-25
- [x] Added `get_log_mean_force` and `get_material_descriptor` (2-vec).
- [x] Bumped hidden to 512 (78K → 290K params).
- [x] Rebuilt dataset and trained Fix F (45-dim).
- [x] Eval: BEST or tied on all 6 (material, profile) combos.
- [x] rope_4 outlier FIXED: 1.84 → 0.77 (controlled).
- [x] Updated load_policy_artifacts to infer in_dim/hidden from state_dict.

### RL on Fix F multi-material ✅ NEW HEADLINE 2026-05-25
- [x] Updated rl_env.py to compute material descriptor per env (43/44/45).
- [x] Updated rl_ppo.py to use env.obs_dim/act_dim.
- [x] Trained 51K steps multi-material with Fix F BC warm-start.
- [x] Eval: wins rope replay (0.41), rope ramp (0.52), sloth replay (0.66).
- [x] [rl_fixF_review.md](rl_fixF_review.md) written.
- [x] 4 demo videos rendered (rope replay/ramp, cloth replay, sloth replay).

### Fix G — Stiffness-shaped RL reward (next experiment)

User-proposed: penalize aggressive actions more heavily on stiff materials,
less on compliant ones. Should fix the sloth-ramp regression where
RL-FF-Multi (2.66) lost to Fix D-T (1.67).

Reward formula:
```
reward_t = -‖F_err‖ / force_scale                       (existing)
         - α * exp(log_spring_Y - C) * ‖action‖²        (NEW — stiffness-scaled action penalty)
         - β * max(0, ‖F_achieved‖ - ‖F_goal‖)²         (OPTIONAL — force overshoot penalty)
```

Where:
- `α`: action-penalty coefficient (start with 1.0, tune)
- `C ≈ 9.0`: normalizer (~typical rope log_spring_Y)
- `β`: overshoot coefficient (optional add-on)

Multiplier scaling:
- Soft cloth (log=7):    e^(-2) = 0.14× penalty (lets policy use bigger actions)
- Typical rope (log=9):  e^0 = 1.0× penalty
- Stiff cloth (log=11):  e^2 = 7.4× penalty (forces small motions)
- Stiff sloth (log=10.5): e^1.5 = 4.5× penalty

Implementation:
- [ ] Modify `PhysTwinForceEnv.step()` reward computation to include
      stiffness-scaled action penalty.
- [ ] Pass `log_spring_Y` from env init → reward fn (already available
      via self.material_descriptor).
- [ ] Add CLI args `--action_penalty_alpha`, `--stiffness_normalizer`,
      and `--overshoot_beta`.
- [ ] Smoke test with small α to confirm reward components are visible
      but don't dominate (~1 hr).
- [ ] Train 50K steps with shaped reward from Fix F BC warm-start
      (~3 hr compute).
- [ ] Eval on full 14-case sweep.
- [ ] Compare to RL-FF-Multi specifically on sloth ramp + cloth ramp.

Expected outcomes:
- ✅ Sloth ramp: likely improvement (currently 2.66, target back toward Fix D-T's 1.67)
- ✅ Cloth ramp: mild improvement on stiff cloth cases
- ⚠️ Rope: likely unchanged (stiffness range too narrow within rope)
- ⚠️ Risk: over-penalty kills force tracking → start with small α

Cost: ~1 hr code + ~3 hr compute.

### Deferred further
- [ ] MPC over the simulator (principled fix for release — bigger
      project, after goal-side fixes ruled out).
- [ ] RL fine-tuning from BC warm-start (alternative to MPC).
- [ ] Outlier handling (reweight training by case force range to fix
      single_push_rope_4-style failures).
- [ ] True per-material policies (one MLP each rather than one shared MLP
      with per-material scaler).

## Open questions (revisit after each step)
- Does the policy need to know `n_ctrl_parts` explicitly, or does the
  padded second-group of zeros already carry that signal?
- Should the action be in world frame or relative to controller frame?
  (Default: world frame, matches how synthetic trajectories were generated.)
- For the replay test, do we feed the recorded `controller_pos[0]` exactly,
  or also let the policy choose the initial action?
  (Default: feed recorded ctrl[0] as initial condition; policy starts at t=1.)
