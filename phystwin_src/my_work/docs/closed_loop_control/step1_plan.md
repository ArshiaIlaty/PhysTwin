# Step 1 plan — Build the policy dataset

Convert the existing per-trajectory npz files into a single flat
`(state, force_now, force_goal, action)` table used to train the policy.
**No PhysTwin involved.** Pure numpy + scikit-learn (for KMeans).

## Goal

For every consecutive frame pair `(t, t+1)` in every trajectory across
`dataset_v2/` and `dataset_synth_raw/`, emit one row containing what the
policy needs at training time.

Output file: `my_work/results/policy_dataset.npz`. Single flat file (not
per-case) for fast loading at training time.

## Inputs

Read from:

| Source | Files | Notes |
|---|---|---|
| Real | `my_work/results/dataset_v2/*.npz` | 21 cases, ~150 frames each |
| Synthetic | `my_work/results/dataset_synth_raw/*.npz` | 218 trajectories, ~120 frames each |

Both have identical schema (verified):
```
X              [T, 31]       float32
y_per_ctrl     [T, 2, 3]     float32   (padded: 2nd group zero for single-ctrl)
y_net          [T, 3]        float32
positions      [T, N, 3]     float32   (not used for policy training, leave on disk)
controller_pos [T, K, 3]     float32
case_name      scalar str
material       scalar str    {rope, cloth, sloth, toy}
n_ctrl_parts   scalar int    {1, 2}
feature_names  [31] str
# synthetic only:
source_donor   scalar str
motion_type    scalar str    {linear_push, sinusoidal, random_walk, hold_release}
```

## Exclusions

- **`single_push_sloth`** — 866 kN spring blow-up, same exclusion as v3.1
  training.
- **Any case where `controller_pos` has fewer than `n_ctrl_parts` distinct
  point clusters** at frame 0. (Defensive check; expected to be empty.)

## Output schema

`my_work/results/policy_dataset.npz`:

```
state         [R, 31]       float32   X[t]
force_now     [R, 2, 3]     float32   y_per_ctrl[t]
force_goal    [R, 2, 3]     float32   y_per_ctrl[t+1]
action        [R, 2, 3]     float32   per-group centroid Δ from t→t+1
action_mask   [R, 2]        float32   1.0 for real groups, 0.0 for padded
material      [R]           <U10      per-row material label (for per-cat scaler)
case_name     [R]           <U64      per-row case id (for random-block split)
n_ctrl_parts  [R]           int8      1 or 2
source        [R]           <U10      {"real", "synth"}
motion_type   [R]           <U16      {"real", "linear_push", "sinusoidal",
                                       "random_walk", "hold_release"}
                                       — "real" for dataset_v2 rows; the
                                       synthetic motion family for
                                       dataset_synth_raw rows. Used in
                                       Step 5 figures to break down policy
                                       performance by source motion.
group_ids     list of arrays [K_i] int  per-case group assignment (saved as
                                        object array, indexed by case_idx)
case_idx      [R]           int32     index into group_ids, per row
```

`R` = total rows ≈ Σ_cases (T_case − 1). With 21 + 218 trajectories at
~100–150 frames each, expect 25k–35k rows.

Why both `material` and `case_name` per row: trainer needs `material` for
the per-category target scaler (v3.1 trick); needs `case_name` for the
random-block split (which holds out 20% consecutive frames per case).

Why `action_mask`: single-control rows have a meaningless 2nd group; mask it
out of training loss.

Why `group_ids` as an object array: each case has a different `K`, so we
can't stack them. Keeping per-case group assignments lets the closed-loop
driver (Step 3) reuse them.

Why `motion_type` per row: lets Step 5 figures split policy performance
by motion family — e.g., "does the policy track ramp-like recorded motion
better than oscillatory hold-release synthetic motion?" Without this we'd
have to re-derive it from `case_name` substring parsing every time.

## Per-frame transformation logic

For one trajectory (`positions`, `controller_pos`, `y_per_ctrl`, X,
`n_ctrl_parts`):

### 1. Compute group assignment (once per trajectory)

```python
from sklearn.cluster import KMeans

ctrl0 = controller_pos[0]                                # [K, 3]
if n_ctrl_parts == 1:
    group_ids = np.zeros(ctrl0.shape[0], dtype=np.int8)
else:
    km = KMeans(n_clusters=n_ctrl_parts, n_init=10, random_state=0).fit(ctrl0)
    group_ids = km.labels_.astype(np.int8)              # [K] in {0, 1}
```

Group assignment is computed on frame 0 only. Each group moves rigidly
across frames in the recorded / synthetic data (verified by construction of
`generate_synthetic.py`), so the per-frame group membership is stable.

### 2. Compute per-frame per-group centroid

```python
T, K, _ = controller_pos.shape
centroids = np.zeros((T, 2, 3), dtype=np.float32)        # padded to 2 groups
for g in range(n_ctrl_parts):
    centroids[:, g] = controller_pos[:, group_ids == g].mean(axis=1)
# centroids[:, 1] left as zero if n_ctrl_parts == 1
```

### 3. Build per-frame-pair rows

```python
T_rows = T - 1
state       = X[:T_rows]                                 # [T-1, 31]
force_now   = y_per_ctrl[:T_rows]                        # [T-1, 2, 3]
force_goal  = y_per_ctrl[1:]                             # [T-1, 2, 3]
action      = centroids[1:] - centroids[:T_rows]         # [T-1, 2, 3]
action_mask = np.zeros((T_rows, 2), dtype=np.float32)
action_mask[:, :n_ctrl_parts] = 1.0

# motion_type tag: present in synthetic npz, absent in real
motion_type = str(traj["motion_type"]) if "motion_type" in traj.files else "real"
```

Broadcast `motion_type` (and the `material`, `case_name`, `n_ctrl_parts`,
`source` scalars) across all `T_rows` rows at concatenation time.

### 4. Append to global buffers, then save

After processing every trajectory, `np.concatenate` along the row axis and
`np.savez` the final dict.

## Validation / sanity checks (must pass before declaring Step 1 done)

These are the printable checks the script must do (and ideally log to a
small `policy_dataset_summary.json` next to the npz):

1. **Row count** matches expectation: `R == sum(T_case - 1)` across
   processed cases.
2. **Per-material row counts** (rope / cloth / sloth / toy) printed.
3. **Action magnitudes** are non-zero and within a sane range. Print 5th /
   50th / 95th / 99th / 99.9th percentile of `||action||` per material.
   Flag any case where 99th-percentile `||action||` > 0.1 m/frame as
   suspicious (real trajectories are slow).
4. **Force-goal − force-now residual.** Most frames should have small
   per-step force change. Print 50th / 95th percentile of
   `||force_goal − force_now||` per material. Large jumps indicate
   numerical instability frames (e.g. the `single_push_sloth` spike — should
   be absent after exclusion).
5. **Group assignment sanity.** For each double-control case, the two
   per-group centroids at frame 0 should be ≥ some minimum separation
   distance (use ½ bbox-diag as a rough check). Print min separation per
   double-control case.
6. **No NaN / Inf** in any of state / force_now / force_goal / action.
7. **Mask coverage.** Number of rows with `action_mask[:, 1] == 1` should
   equal Σ_{double-control cases} (T − 1). Number with == 0 should equal
   Σ_{single-control cases} (T − 1).
8. **Motion-type coverage.** Print row counts per `motion_type` value
   ({real, linear_push, sinusoidal, random_walk, hold_release}). Expected
   from the existing dataset: `real` ≈ ~2k rows (21 cases × ~100 frames),
   synth ≈ ~26k rows (218 trajectories × ~120 frames). Within synth,
   `linear_push` and `hold_release` should dominate per the
   `experiment_explanation.md` accepted-motion counts.

If any check fails, raise — don't write the file.

## Script structure

Where it lives: `my_work/code/build_policy_dataset.py`.

CLI shape (mirror existing my_work scripts):

```bash
python my_work/code/build_policy_dataset.py \
    --dataset_v2_dir   my_work/results/dataset_v2 \
    --synth_dir        my_work/results/dataset_synth_raw \
    --out              my_work/results/policy_dataset.npz \
    --exclude          single_push_sloth \
    --include_synth    true
```

All flags default to the values above so plain `python build_policy_dataset.py`
from the repo root does the right thing.

Code outline:

```python
def load_trajectory(npz_path):
    """Return dict with keys needed; handle missing optional fields."""

def compute_group_ids(controller_pos_t0, n_ctrl_parts):
    """KMeans on frame 0; returns [K] int8 in {0..n_ctrl_parts-1}."""

def trajectory_to_rows(traj):
    """Returns dict of per-frame-pair arrays for one trajectory."""

def main():
    # 1. enumerate dataset_v2/*.npz and dataset_synth_raw/*.npz
    # 2. apply exclusions
    # 3. load + convert each → list of row-dicts
    # 4. concatenate
    # 5. run validation checks (raise on failure)
    # 6. write npz + summary json
```

Keep it under ~200 LOC. No CUDA, no torch dependency — pure
numpy + sklearn + json.

## Out of scope for Step 1 (do later)

- Train/val split — done at training time in Step 2.
- Feature scaling — fit scalers at training time in Step 2.
- Loading `positions` (we don't need raw particles for the policy; just X).
- Including `source_donor` per row — useful for tracing a synthetic
  trajectory back to its calibrated origin, but `case_name` already encodes
  the donor as a prefix (`double_lift_cloth_1__synth_hold_release_0` →
  donor is everything before `__`). Skip the separate field; recover by
  string-split if ever needed.
- Validating that the recorded `controller_pos` motion is actually rigid
  per-group. (It is, by construction. If a future synthetic generator
  breaks this, the action vector still defines a valid centroid delta —
  rollout would just lose some shape info, which we don't need for the
  policy anyway.)

## Scope estimate

~1 hour: 30 min code + 15 min validation + 15 min iteration on edge cases.
No compute load — runs on the login node in seconds.

## What could go wrong

| Risk | Symptom | Fix |
|---|---|---|
| KMeans on frame 0 gives an empty cluster (degenerate ctrl point layout). | `group_ids` has only one unique value when `n_ctrl_parts == 2`. | Defensive check #5 catches this. Skip the case and log a warning. |
| `controller_pos` contains NaN frames (rare, e.g. simulator stall). | NaN propagates into action. | Validation check #6 catches; we either trim or skip the case. |
| Forces in `y_per_ctrl` saved as `[T, 1, 3]` for some single-control synthetic files instead of `[T, 2, 3]`. | Shape mismatch on concat. | Pad with zeros to `[T, 2, 3]` if needed (existing `pad_forces` helper in `extract_dataset.py` already does this for real data — verify synthetic matches). |
| Action magnitudes are huge for some synthetic trajectories (sanity-gate didn't catch). | 99th percentile action ≫ recorded extent. | Validation check #3 flags; trim those rows or drop the trajectory. |

---

## Acceptance criteria

Step 1 is done when:
1. `my_work/results/policy_dataset.npz` exists and loads cleanly.
2. All 7 validation checks pass and are recorded in
   `my_work/results/policy_dataset_summary.json`.
3. Row counts are within ±5% of `Σ(T_case − 1)` for non-excluded cases.
4. A 10-line review at the end of `tasks.md` (or a new `step1_review.md`)
   documents the actual numbers (total rows, per-material counts, action
   magnitude percentiles).
