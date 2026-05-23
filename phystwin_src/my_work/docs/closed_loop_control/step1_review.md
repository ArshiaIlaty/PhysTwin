# Step 1 review — Build policy dataset

Plan: [step1_plan.md](step1_plan.md). Script:
[../../code/build_policy_dataset.py](../../code/build_policy_dataset.py).
Output: [../../results/policy_dataset.npz](../../results/policy_dataset.npz)
+ [../../results/policy_dataset_summary.json](../../results/policy_dataset_summary.json).

## Status

✅ Done. All 8 validation checks pass with zero warnings.

## Artifacts

| File | Size | Notes |
|---|---|---|
| `results/policy_dataset.npz` | 17.9 MB | 12 arrays (state, force_now/goal, action, action_mask, material, case_name, n_ctrl_parts, source, motion_type, case_idx, group_ids) |
| `results/policy_dataset_summary.json` | small | Full per-material / per-motion stats |

## Headline numbers

- **238 trajectories processed** → **28,238 rows** (= Σ T-1 exactly).
- 20 real (after excluding `single_push_sloth`) + 218 synthetic.
- 8,141 rows from double-control cases, 20,097 from single-control.

### Per material

| Material | Rows | Trajectories |
|---|---:|---:|
| cloth | 15,294 | ~135 |
| rope | 7,682 | ~65 |
| sloth | 4,858 | ~36 |
| toy | 404 | 4 |

### Per motion-type

| Motion | Rows |
|---|---:|
| real (recorded) | 2,296 |
| linear_push | 8,687 |
| hold_release | 8,568 |
| sinusoidal | 6,426 |
| random_walk | 2,261 |

Trajectory counts back-derived (rows / ~119 frames per synth traj) match the
documented accepted-motion counts from
[../tasks/experiment_explanation.md](../tasks/experiment_explanation.md):
linear_push 73, hold_release 72, sinusoidal 54, random_walk 19. Consistent.

### Action magnitude (m/frame, ‖Δctrl‖ for real groups)

| Material | p95 | p99 | p99.9 |
|---|---:|---:|---:|
| cloth | 0.0145 | 0.0223 | 0.0304 |
| rope | 0.0027 | 0.0045 | 0.0080 |
| sloth | 0.0056 | 0.0099 | 0.0122 |
| toy | 0.0108 | 0.0129 | 0.0151 |

All well under the 0.1 m/frame sanity bound. Rope is the slowest mover —
expected (the recorded rope manipulations are gentle pushes).

### Force-goal − force-now residual (N, per frame)

| Material | p50 | p95 | p99 |
|---|---:|---:|---:|
| cloth | 567 | 13,522 | 28,143 |
| rope | 96 | 4,640 | 10,666 |
| sloth | 4,446 | 63,364 | 102,473 |
| **toy** | 4,160 | **2,827,715** | **5,891,855** |

**Toy is broken.** p95 and p99 are 100–1000× larger than any other
material. This matches the prior project's observation that
`double_lift_zebra` and `single_lift_dinosor` have numerical force
blow-ups. The synthetic donor list excludes them, but the real npz files
are still in `dataset_v2/`. Action item carried into Step 2 — either drop
toy from policy training, or apply per-case 99th-percentile force clipping
(same trick as v3.1).

### Group separation (double-control cases only)

- 67 double-control trajectories.
- Min inter-group centroid separation at frame 0: **0.309 m** (well above
  the 0.01 m sanity floor). KMeans clustering is recovering meaningful
  two-handed gripper layouts.

## Hiccups

### `material` vs `object_category` field confusion

First run printed `per material: {'cloth': 15294, 'real': 12944}`. Turns
out the `material` scalar in the npz is the **cfg_type** ("real" or
"cloth", which PhysTwin uses to pick `configs/real.yaml` vs
`configs/cloth.yaml`), not the material taxonomy. The real material
labels (rope / cloth / sloth / toy) live in **`object_category`**.

Fixed by preferring `object_category` in `load_trajectory()`. Logged in
[lessons.md](lessons.md) for future scripts that read these npz files.

## Acceptance criteria check

From [step1_plan.md](step1_plan.md):

- [x] `policy_dataset.npz` exists and loads cleanly.
- [x] All 8 validation checks pass; stats in
      `policy_dataset_summary.json`.
- [x] Row counts within ±5% of `Σ(T_case − 1)` — actually exact match.
- [x] Numbers documented here (this file).

## Carry-forward to Step 2

1. **Toy category needs outlier handling** before training. Options: drop
   entirely, or clip force_goal/force_now to per-case 99th percentile.
2. **Per-material force scale variance is large** (cloth median residual
   567 N vs sloth median 4,446 N). Reuse v3.1's per-category target
   scaler — single shared scaler will fail again.
3. **Trajectory imbalance**: cloth has ~3× more rows than rope, ~30× more
   than toy. Consider per-material loss weighting or balanced sampling.
4. **The `random_walk` slice is small** (2,261 rows). Per-motion-type
   eval breakdowns in Step 5 will have low statistical power for this
   slice — note in figure captions.
