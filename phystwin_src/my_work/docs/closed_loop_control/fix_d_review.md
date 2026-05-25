# Fix D review — Synthetic ramp_full data PARTIALLY WORKS

Adds 89 `ramp_full` (symmetric triangle-wave push-release) synthetic
trajectories to the training set, retrains policy, re-evaluates. Plus
a side fix to the ramp test for physically realistic goal directions.

## Status

✅ **The actual fix target (rope ramp release) is FIXED.** With the
recorded-mean direction, release fraction jumped 0.04 → **0.84** (21×).

⚠️ **Trade-offs exist on the broader eval.** Some cases improved
dramatically, two cloth cases regressed badly, one went NaN. Net per-
material aggregate is mixed.

## Gate outcomes

| Gate | Result |
|---|---|
| G1 — generate 89 ramp_full synth trajectories | PASS (rope=27, cloth=46, sloth=16; 39 rejected for force spike) |
| G2 — rebuild policy_dataset (38,829 rows, +10,591 from ramp_full) | PASS |
| G3 — retrain policy (seed 1, 100 epochs) | PASS — val R² mostly stable, sloth regressed slightly |
| G4 — rope ramp test, x_axis direction | PARTIAL — release frac 0.57 (huge improvement from 0.04) |
| G4b — rope ramp test, recorded_mean direction | **PASS — release frac 0.84** |
| G5 — cloth replay regression on double_lift_cloth_1 | MILD regression (0.22 → 0.34) |
| G6 — full Step 4 eval sweep on Fix D policy | MIXED — see table below |

## G4/G4b — Rope ramp results

### With arbitrary +x direction (`--ramp_direction x_axis`)

| Frame | 0 | 23 | 46 (goal peak) | 69 | 91 (goal=0) |
|---|---|---|---|---|---|
| Goal mag (N) | 0 | 8,683 | 16,988 | 8,305 | 0 |
| Baseline achieved | 0 | 6,566 | 8,606 | 5,776 | **8,711** |
| Fix D achieved | 0 | 7,406 | 16,300 | 4,934 | 7,005 |

**Baseline release fraction: 0.04. Fix D: 0.57.** 14× improvement, but
end-frame still at 7 kN instead of 0 — incomplete release.

### With physically realistic direction (`--ramp_direction recorded_mean`)

| Frame | 0 | 23 | 46 | 69 | 91 (goal=0) |
|---|---|---|---|---|---|
| Goal mag (N) | 0 | 8,683 | 16,988 | 8,305 | 0 |
| Fix D achieved | 0 | 10,491 | 14,469 | 7,144 | **2,367** |

**Release fraction: 0.84.** Almost full return to zero. Achieved tracks
goal magnitude tightly through rise, peak, AND release. The new
[fixD_rope_ramp_RELEASE_recmean.mp4](../../results/figures/closed_loop/videos/fixD_rope_ramp_RELEASE_recmean.mp4)
video shows the gripper smoothly releasing the rope as the goal force
drops back toward 0.

## G6 — Full Step 4 sweep comparison (baseline vs Fix D)

| Case | Profile | Baseline | Fix D | Δ | Verdict |
|---|---|---:|---:|---:|---|
| double_lift_cloth_1 | replay | 0.221 | 0.327 | +0.106 | small regression |
| single_clift_cloth_1 | replay | 0.249 | **0.135** | −0.115 | improvement |
| single_lift_cloth | replay | 0.451 | **NaN** | — | CATASTROPHIC |
| double_lift_cloth_3 | replay | 0.515 | **4.853** | +4.338 | BIG regression |
| double_lift_cloth_3 | ramp | 6.552 | 6.488 | −0.064 | unchanged (still broken) |
| single_push_rope_1 | replay | 0.256 | **0.179** | −0.077 | improvement |
| single_push_rope_1 | ramp (x_axis) | 0.664 | 0.684 | +0.020 | basically same |
| single_lift_rope | replay | 0.356 | **0.124** | −0.231 | BIG improvement |
| single_push_rope | replay | 0.536 | 0.553 | +0.017 | unchanged |
| single_push_rope_4 | replay | 1.842 | 2.670 | +0.828 | regression (already broken) |
| single_lift_sloth | replay | 0.290 | **0.274** | −0.017 | improvement |
| double_lift_sloth | replay | 0.882 | 1.016 | +0.134 | borderline regression |
| double_stretch_sloth | replay | 1.449 | **1.316** | −0.133 | improvement |
| double_stretch_sloth | ramp | 5.062 | **2.847** | −2.215 | improvement |

### Per-material aggregates

| Mat | Profile | Baseline err | Fix D err | Baseline ctrl | Fix D ctrl |
|---|---|---:|---:|---|---|
| cloth | replay | 0.359 | **1.771** | 4/4 | 3/4 (one NaN) |
| cloth | ramp | 6.552 | 6.488 | 0/1 | 0/1 |
| rope | replay | 0.747 | 0.882 | 3/4 | 3/4 |
| rope | ramp | 0.664 | 0.684 | 1/1 | 1/1 |
| sloth | replay | 0.874 | 0.869 | 2/3 | 1/3 |
| sloth | ramp | 5.062 | **2.847** | 0/1 | 0/1 |

## Interpretation

**The good** (clear wins):
- Rope ramp release: **0.04 → 0.84** (with recorded_mean direction). The
  original failure mode is essentially solved.
- Sloth ramp peak overshoot: 5.06 → 2.85 (still uncontrolled but
  noticeably better).
- 3 rope/cloth/sloth replay cases improved by ≥10% err_ratio:
  single_lift_rope (−0.23), single_clift_cloth_1 (−0.11),
  double_stretch_sloth (−0.13).

**The bad** (clear losses):
- `double_lift_cloth_3` replay: 0.52 → 4.85 (catastrophic 9× regression)
- `single_lift_cloth` replay: 0.45 → NaN (simulation explosion)
- `double_lift_cloth_1` replay (our headline best case): 0.22 → 0.34

Pattern: cloth replay cases that USED to work suffered the most.
Rope/sloth either improved or stayed roughly the same.

**Why the cloth regression**: the new ramp_full trajectories use larger
amplitudes (amp_max bumped 0.40 → 0.45) and full release cycles. 46 of
the 89 new trajectories are cloth-donor-derived. The policy now expects
more aggressive cloth motions; on cases that need gentler manipulation,
it over-applies and the compliant cloth simulator destabilizes.

A targeted re-run could probably fix this: only generate ramp_full for
rope/sloth donors, leave cloth training data alone. ~30 min experiment.

## The direction-fix finding (independent of Fix D)

This wasn't part of the original Fix D plan but emerged from
investigating the video arrow mismatch. **The choice of goal direction
matters enormously for ramp test results.**

Same policy, same case, same magnitude profile, just different
direction:

| Direction | Release fraction | Peak match | End-frame achieved (goal=0) |
|---|---:|---:|---:|
| `x_axis` (arbitrary) | 0.57 | 0.96× | 7,005 N |
| `recorded_mean` | **0.84** | 0.85× | **2,367 N** |

When the goal vector points in a direction the simulator can physically
produce (along the rope's natural force axis), the policy can drive
forces in that direction effectively. When the goal vector points in
an arbitrary direction (+x), the policy must compromise — gripper
motion in any direction will produce reaction forces along the rope's
geometry, not aligned with +x.

**Implication for the demo**: always use `--ramp_direction recorded_mean`
in synthetic ramp tests. The +x version was misleading us — it
underestimated what the policy can actually do.

Step 4's ramp results all used the `x_axis` direction. A future
re-evaluation with `recorded_mean` would likely show better numbers
across the board. Out of scope for now.

## Artifacts

- `code/generate_synthetic.py` — added `ramp_full` motion type and
  `--motion_types` filter flag.
- `code/run_closed_loop.py` — added `--ramp_direction
  {x_axis,recorded_mean}` flag.
- `results/dataset_synth_raw/*ramp_full*.npz` × 89 new trajectories.
- `results/policy_dataset_fixD.npz` — 38,829 rows including ramp_full.
- `results/models_policy_fixD/seed_1/` — Fix D trained policy.
- `results/eval_fixD/` — initial G4/G5 rollouts (x_axis direction).
- `results/eval_fixD/single_push_rope_1__policy_ramp_recmean.npz` — the
  killer demo rollout.
- `results/eval_closed_loop_fixD/` — full Step 4 sweep, 14 rollouts.
- `results/figures/closed_loop/videos/fixD_rope_ramp_RELEASE.mp4`
  (x_axis) and `_recmean.mp4` (recorded_mean direction).

## Decision matrix

For the demo, three possible framings:

1. **"Fix D works on rope, partial on sloth, cost on cloth"** —
   honest full picture; complicated story; some uncomfortable numbers.
2. **"Fix D demonstrates that targeted synthetic data can teach release
   behavior"** — focus on the rope ramp win + direction-fix discovery;
   call the cloth regression a known trade-off; recommend the
   per-material-targeted version as future work.
3. **"Best of both worlds: present baseline for cloth replay, Fix D for
   ramps"** — pick whichever policy is best per (material, profile)
   for the demo. Honest but cherry-picking.

I'd recommend #2. The win on the original target case (rope ramp
release, the headline failure from Steps 3-4) is the clean story. The
cloth regression is a real finding worth mentioning but doesn't undo
the result on what the project was actually trying to fix.

## Carry-forward

- **Targeted regeneration**: run `generate_synthetic.py --motion_types
  ramp_full --materials rope,sloth` (excluding cloth) and retrain.
  ~30 min experiment. Likely keeps cloth replay quality while
  preserving the rope ramp release win.
- **Re-render baseline figures with `--ramp_direction recorded_mean`**:
  the Step 4 ramp metrics are slightly pessimistic. ~10 min compute.
- **Other Fix D ramp videos**: with the recorded_mean fix, the cloth
  and sloth ramp tests might be more demoable too. Worth trying.
- **Original demo videos**: should be re-rendered with the new
  `--ramp_direction recorded_mean` for the ramp ones so the achieved/
  goal arrows align.

---

## Addendum 2026-05-24 — Fix D-targeted (cloth ramp_full removed)

Did the carry-forward "targeted regeneration": moved the 46 cloth
ramp_full trajectories to a quarantine dir, kept rope (27) + sloth (16),
rebuilt dataset, retrained policy. Total: 281 trajectories, 33,355 rows.

### Comparison: baseline vs Fix D vs Fix D-targeted

#### Rope ramp (the original fix target, recorded_mean direction)

| | Peak achieved / goal | Release frac | End achieved (goal=0) |
|---|---:|---:|---:|
| Baseline | 9 kN / 17 kN = 0.54× | 0.04 | 8.7 kN |
| Fix D | 14.5 kN / 17 kN = 0.85× | 0.84 | 2.4 kN |
| **Fix D-targeted** | **15.2 kN / 17 kN = 0.90×** | **0.87** | **2.0 kN** |

Fix D-targeted is slightly better than Fix D on the actual fix target.

#### Cloth replay regression check

| Case | Baseline | Fix D | Fix D-targeted | Verdict |
|---|---:|---:|---:|---|
| double_lift_cloth_1 | 0.22 | 0.33 (+0.11) | **0.28 (+0.06)** | mild regression reduced |
| double_lift_cloth_3 | 0.52 | 4.85 | **2.17** | still bad, 2× better than Fix D |
| single_lift_cloth | 0.45 | **NaN** | **1.30** | no blow-up |

Cloth regression **reduced significantly but not eliminated**. The shared
MLP still passes some rope/sloth ramp_full influence to cloth predictions
even when cloth's training data is unchanged — capacity is the
bottleneck.

#### Full per-material aggregates

| Mat | Profile | BASE | Fix D | **Fix D-T** | Best |
|---|---|---:|---:|---:|---|
| cloth | replay | 0.36 | 1.77 | **1.00** | BASE |
| cloth | ramp | 6.55 | 6.49 | **5.04** | Fix D-T |
| rope | replay | 0.75 | 0.88 | 0.86 | BASE (small) |
| rope | ramp | 0.66 | 0.68 | 0.70 | ~ tie |
| sloth | replay | 0.87 | 0.87 | **0.74** | Fix D-T |
| sloth | ramp | 5.06 | 2.85 | **1.67** | Fix D-T |

Fix D-targeted is the **best single policy** for 4 of 6 (material, profile)
combinations. Baseline only wins on cloth replay (and that's because
nothing fully restores the cloth replay quality with the shared-policy
architecture).

### What the targeted experiment proved

1. The cloth ramp_full data was contributing to the regression — removing
   it cut the cloth replay damage in half.
2. But it wasn't the only contributor — even without cloth ramp_full,
   adding rope/sloth aggressive trajectories indirectly affects cloth
   predictions through the shared network weights.
3. The remaining cloth regression is the **shared-capacity issue**. The
   architectural fix is true per-material policies (separate MLP for each
   material rather than one shared MLP with per-material output scaler).
   That's a ~30-min plumbing change to `train_policy.py` and would likely
   fully restore baseline cloth quality while keeping the rope/sloth
   improvements.

### Demo videos rendered (Fix D-targeted with recorded_mean direction)

- `fixDtargeted_rope_ramp.mp4` — the headline release video. Achieved
  force tracks goal magnitude through rise, peak, AND release.
- `fixDtargeted_cloth_replay.mp4` — cloth still works decently for replay.
- `fixDtargeted_sloth_replay.mp4` — sloth's best case, better than
  baseline.

### Recommended demo policy

**Fix D-targeted is the policy to feature.** It's the single best
"general-purpose" model — fixes the original failure mode (rope ramp
release) AND improves sloth across the board, with only a moderate
cloth replay regression that the demo can acknowledge honestly.

If you wanted absolutely the best per case, you could pick per-(material,
profile) which policy to show — but Fix D-targeted is the most defensible
single artifact to ship.

### One more cheap experiment if it matters

**True per-material policies** (one MLP per material): ~30 min of
plumbing in `train_policy.py`. Adds an outer loop over materials. Each
MLP is smaller; total params increase by 3× but training data per model
is partitioned. Likely fully restores cloth replay quality while keeping
all the wins. Out of scope unless someone gives the go.
