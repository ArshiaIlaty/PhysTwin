# Step 4 review — Closed-loop systematic evaluation

Plan: [step4_plan.md](step4_plan.md). Script:
[../../code/eval_closed_loop.py](../../code/eval_closed_loop.py). Rollouts:
[../../results/eval_closed_loop/](../../results/eval_closed_loop/). SLURM
logs: `eval_5265846{1,2,3}.{out,err}`.

## Status

✅ Done. All 3 verification gates passed; 14/14 rollouts ran with 0 NaN.

## Verification gate outcomes

| Gate | Wall time | Result |
|---|---|---|
| G1 single-case smoke | 28 s | PASS — `err_ratio = 0.256` matches Step 3 G4 exactly (regression check OK) |
| G2 per-material smoke | 2:13 | PASS — all 3 materials run cleanly with 1 case each |
| G3 full sweep | 10:49 | PASS — all 14 rollouts complete, 0 NaN, summary aggregated |

## Headline result — per-material aggregates

```
material  profile               n   err_ratio        overshoot  uncontrolled
cloth     policy_recorded_goal  4   0.36 ± 0.13      1.13       0/4
rope      policy_recorded_goal  4   0.75 ± 0.64      0.87       1/4
sloth     policy_recorded_goal  3   0.87 ± 0.47      1.06       1/3
cloth     policy_ramp           1   6.55             4.52       1/1
rope      policy_ramp           1   0.66             0.54       0/1
sloth     policy_ramp           1   5.06             2.26       1/1
```

`err_ratio` = mean ‖achieved − goal‖ / mean ‖goal‖. Lower is better; 1.0
means the average error equals the average goal magnitude (system
basically uncontrolled). `overshoot` = peak achieved / peak goal.
`uncontrolled` count = rollouts with `err_ratio > 1.0`.

## Per-case breakdown

```
case                       material  profile               T    err_ratio  overshoot  drift   goal_N
double_lift_cloth_1        cloth     policy_recorded_goal  116  0.22       0.93       0.21    10,462
single_clift_cloth_1       cloth     policy_recorded_goal   80  0.25       1.08       0.19    22,049
single_lift_cloth          cloth     policy_recorded_goal  173  0.45       1.85       0.32    61,423
double_lift_cloth_3        cloth     policy_recorded_goal  118  0.52       0.67       0.26     7,025
single_push_rope_1         rope      policy_recorded_goal   92  0.26       0.87       0.10    15,347
single_lift_rope           rope      policy_recorded_goal   50  0.36       0.83       0.20    14,986
single_push_rope           rope      policy_recorded_goal   58  0.54       0.56       0.13     4,018
single_push_rope_4         rope      policy_recorded_goal   83  1.84       1.19       0.03     2,560   ← outlier
single_lift_sloth          sloth     policy_recorded_goal   85  0.29       0.91       0.29    39,651
double_lift_sloth          sloth     policy_recorded_goal   62  0.88       0.52       0.16    47,294
double_stretch_sloth       sloth     policy_recorded_goal  192  1.45       1.75       0.08    20,163   ← outlier
single_push_rope_1         rope      policy_ramp            92  0.66       0.54       0.14     8,494
double_lift_cloth_3        cloth     policy_ramp           118  6.55       4.52       0.14     1,367   ← uncontrolled
double_stretch_sloth       sloth     policy_ramp           192  5.06       2.26       0.05     4,956   ← uncontrolled
```

## What this says, plainly

### Cloth is the strongest material (surprising finding)

All 4 cloth replay rollouts are controlled (err_ratio ≤ 0.52). Mean
0.36 ± 0.13. This is the **best-performing material** in the closed-loop
demo — better than rope, despite cloth being harder during training
(per Step 2 R² ordering, rope 0.82 > cloth 0.54). Possible reasons: the
training-loss R² doesn't translate directly to closed-loop tracking;
cloth's larger force scale makes the relative error favorable; or the
v3.1 features capture cloth dynamics well in spite of single-frame
prediction noise.

### Rope: 3 of 4 cases work, 1 outlier

- 3 controlled cases (single_push_rope_1, single_lift_rope,
  single_push_rope): err_ratio 0.26, 0.36, 0.54. Mean ≈ 0.39.
- 1 uncontrolled (single_push_rope_4): err_ratio 1.84 — the policy
  applies large forces but the case's recorded force is tiny (mean
  goal only 2.5 kN, the lowest in the dataset). Likely cause: the
  policy is calibrated to the training distribution's larger force
  scales and overshoots gentle cases.

### Sloth is the most variable

- single_lift_sloth: 0.29 (excellent)
- double_lift_sloth: 0.88 (borderline)
- double_stretch_sloth: 1.45 (uncontrolled — note this is a 192-frame
  trajectory, the longest in the sweep; compounding error has the most
  time to accumulate)

### Ramp confirms Step 3 finding

- **Rope ramp** is borderline (0.66 — not great but not catastrophic).
- **Cloth ramp** is broken (6.55) — the simulator blows up under the
  novel target, consistent with Step 3 Fix A finding.
- **Sloth ramp** is broken (5.06).

The ramp story for the demo: "monotone push-up targets are tractable
on rope; release-to-zero and compliant-material ramps are not." This
matches the Step 3 carry-forward exactly.

## Hiccups & fixes

None this step. All gates passed cleanly, the eval harness worked,
subprocess wrapping added ~2 seconds per case as predicted.

The one substantive surprise — cloth outperforming rope on closed-loop
tracking despite worse Step 2 R² — is a finding, not a bug. Documented
under "What this says, plainly."

## Acceptance criteria check

From [step4_plan.md](step4_plan.md):

- [x] 14 rollout npzs in `results/eval_closed_loop/`.
- [x] `summary.json` with per-rollout + per-material aggregates.
- [x] G1, G2, G3 all PASS.
- [x] Per-material replay table (above).
- [x] Ramp-failure characterization (above + Fix A addendum from Step 3).
- [x] No new lessons surfaced; behaviors matched expectations from
      Step 3 carry-forward.

## Carry-forward to Step 5

1. **The demo's headline number is `cloth = 0.36 ± 0.13 across 4
   cases`.** Rope can be reported as "0.39 across 3 of 4 controlled
   cases, 1 outlier on a very gentle case (2.5 kN recorded peak)."
2. **Per-case force-tracking plots** are the most valuable Step 5
   figure type. For each controlled case, plot `‖F_achieved(t)‖` vs
   `‖F_goal(t)‖`. 8 controlled replay cases + 1 controlled ramp case
   = 9 figure rows.
3. **For the limitations panel**, plot the 3 uncontrolled cases
   (single_push_rope_4, double_stretch_sloth, double_lift_cloth_3 ramp)
   to show what failure looks like.
4. **The lightweight video adapter (per Step 5 plan) is best
   demonstrated on `double_lift_cloth_1`** — lowest err_ratio (0.22)
   and visually interesting (two-hand cloth lift). Second choice:
   `single_lift_sloth` (sloth's best, 0.29).
5. **Outlier story**: single_push_rope_4 is interesting analytically
   (low recorded force, policy over-applies) but not video-worthy.
   Mention in text, skip the video.
6. **Total compute budget for Step 5**: figure generation is CPU-only;
   video rendering (if pursued) is ~5 min per case on GPU. Budget 6
   videos × 5 min ≈ 30 min of GPU time.
