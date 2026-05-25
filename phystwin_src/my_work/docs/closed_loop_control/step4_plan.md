# Step 4 plan — Closed-loop evaluation

Run the trained policy across a curated set of cases and target-force
profiles, collect quantitative tracking metrics, and produce the
per-material headline numbers for the demo.

This is the **systematic eval** — Step 3 was "does the loop close?"
Step 4 is "where does it work, where does it break, what are the numbers?"

## Scope (narrowed after Step 3 / Fix A findings)

Per [step3_review.md addendum](step3_review.md#addendum-2026-05-23--fix-a-goal-shaping-attempted-and-discarded),
the policy works in **replay mode** (track recorded force trajectories)
but fails on synthetic ramp/step/sinusoid targets (compounding-error
distribution shift, see lessons). Step 4 reflects this:

- **Primary**: replay-mode tracking across 3–4 cases per material.
  Generates the headline R²/RMS error per material.
- **Secondary**: ONE ramp rollout per material as a "limitation
  characterization" exhibit. Documents the failure mode for Step 5
  figures.
- **Out of scope**: step / sinusoid profiles; cross-case generalization
  claims; any new model training.

## Goal & outputs

End state:

```text
results/eval_closed_loop/
  <case>__policy_recorded_goal.npz    × 9-12 rollouts (3-4 per material)
  <case>__policy_ramp.npz             × 3 rollouts (1 per material)
  summary.json                        per-rollout + per-material aggregates
```

`summary.json` schema:
```json
{
  "per_rollout": {
    "<case>__<profile>": {
      "material": "rope|cloth|sloth",
      "T": 92,
      "mean_force_err_N": 3926.4,
      "mean_goal_mag_N": 15346.6,
      "force_err_ratio": 0.256,
      "peak_force_err_N": ...,
      "peak_goal_overshoot_ratio": ...,
      "final_drift_m": 0.06,
      "any_nan": false
    },
    ...
  },
  "per_material": {
    "rope": {
      "replay": {"mean_err_ratio": 0.28, "std": 0.05, "n_cases": 4},
      "ramp":   {"mean_err_ratio": 1.30, "std": 0.20, "n_cases": 1}
    },
    "cloth": ...,
    "sloth": ...
  }
}
```

The per-rollout npzs already include `forces`, `F_goal`, `positions`,
`controller_pos`, `recorded_*` — no new fields needed. Step 4 just
adds many more of them and the aggregation script.

## Case selection

| Material | Cases to evaluate (replay) | Ramp case |
|---|---|---|
| rope | `single_push_rope_1`, `single_push_rope`, `single_push_rope_4`, `single_lift_rope` | `single_push_rope_1` |
| cloth | `double_lift_cloth_3`, `double_lift_cloth_1`, `single_clift_cloth_1`, `single_lift_cloth` | `double_lift_cloth_3` |
| sloth | `double_stretch_sloth`, `double_lift_sloth`, `single_lift_sloth` | `double_stretch_sloth` |

3 + 4 + 4 = 11 replay rollouts + 3 ramp rollouts = **14 rollouts total**.

At ~30 s per rollout on A30, total compute ≈ 7–10 min. Comfortably fits
one sbatch job under 30 min wall time.

All cases above are in the policy's training data (within-case split).
Sloth excludes `single_push_sloth` (the 866 kN spike case, already
excluded from training).

## Architecture

One new script: `my_work/code/eval_closed_loop.py`. Does:

1. Walk a case list + profile list.
2. For each (case, profile), call `run_closed_loop.py`-equivalent logic
   in-process (don't shell out — set up trainer once per case, reuse
   across profiles for that case).
3. Compute per-rollout metrics (RMS force error, peak overshoot, drift,
   NaN check).
4. Aggregate per material.
5. Write all npzs + summary.json.

Alternatively: a thin wrapper that loops `run_closed_loop.py` as a
subprocess. Simpler but slower (trainer setup ~10 s per case × 11 cases
= 2 extra min). Tractable either way; choose subprocess for simplicity.

**Decision**: subprocess wrapper. Cost in time is ~2 min vs ~1 hr of
extra code complexity for the in-process refactor.

SLURM script: `my_work/scripts/slurm/eval_closed_loop.sbatch`. Runs the
eval driver under the standard env.

## Per-rollout metrics (defined precisely)

For each rollout npz with arrays `forces[T, n_ctrl, 3]`, `F_goal[T, 2, 3]`:

```python
n = n_ctrl_parts                          # 1 or 2
F_a = forces                              # [T, n, 3]      achieved
F_g = F_goal[:, :n]                       # [T, n, 3]      goal (active groups only)

err_per_frame = np.linalg.norm(F_a - F_g, axis=-1)        # [T, n]
mag_a = np.linalg.norm(F_a, axis=-1)                       # [T, n]
mag_g = np.linalg.norm(F_g, axis=-1)                       # [T, n]

metrics = {
    "mean_force_err_N":   float(err_per_frame.mean()),
    "p95_force_err_N":    float(np.percentile(err_per_frame, 95)),
    "mean_goal_mag_N":    float(mag_g.mean()),
    "force_err_ratio":    float(err_per_frame.mean() / max(mag_g.mean(), 1e-6)),
    "peak_overshoot_ratio": float(mag_a.max() / max(mag_g.max(), 1e-6)),
    "final_force_N":      float(mag_a[-1].mean()),
    "final_drift_m":      float(np.linalg.norm(controller_pos[-1] - controller_pos[0], axis=-1).max()),
    "any_nan":            bool(np.isnan(forces).any() or np.isnan(positions).any()),
    "T":                  int(T),
}
```

These are the same metrics already printed at the end of
`run_closed_loop.py` plus a couple more (p95, peak overshoot).

## Per-material aggregates

For each (material, profile):
- mean and std of `force_err_ratio` across cases of that material
- mean and std of `peak_overshoot_ratio`
- count of cases with `any_nan == True` (sim instability count)
- count of cases above a generous error threshold (e.g.
  `force_err_ratio > 1.0` = "achieved-vs-goal error exceeds the goal
  magnitude itself, system uncontrolled")

## Verification gates

### G1 — Single-case smoke (~1 min compute)

```bash
python my_work/code/eval_closed_loop.py --cases single_push_rope_1 \
    --profiles policy_recorded_goal --out my_work/results/eval_smoke
```

What it does: full path through one case, write one npz + a tiny
summary.json with one entry.

**Pass criteria**:
- Script completes without crash.
- `eval_smoke/single_push_rope_1__policy_recorded_goal.npz` exists.
- Summary has the expected metric keys.
- Reproduces Step 3 G4's numbers within ±5% (sanity that nothing
  drifted since Step 3).

### G2 — Per-material smoke (~3 min compute)

```bash
python my_work/code/eval_closed_loop.py \
    --cases single_push_rope_1 double_lift_cloth_3 double_stretch_sloth \
    --profiles policy_recorded_goal --out my_work/results/eval_smoke
```

**Pass criteria**:
- All 3 cases complete (some may have high error — that's data, not bug).
- summary.json has per-material aggregates with 1 case each.
- No crash on the double-control cloth or sloth cases.

### G3 — Full sweep (~10 min compute)

```bash
python my_work/code/eval_closed_loop.py  # all default cases + profiles
```

**Pass criteria**:
- All 14 rollouts complete.
- summary.json has 14 per_rollout entries + per_material aggregates.
- For replay: per-material `mean force_err_ratio` ≤ 1.0 (achieved is
  in the ballpark of goal — sanity check that the policy is
  controlling, not random).
- For ramp: per-material `peak_overshoot_ratio` documented (no
  threshold; we expect overshoot per Step 3 findings).
- NaN count: documented per material (we expect 0 for replay; some
  ramp rollouts may have NaN — log and continue).

## Scope estimate

| Phase | Time |
|---|---|
| Write eval_closed_loop.py + aggregator | 1.5 hr |
| Write SLURM wrapper | 15 min |
| G1 smoke | 5 min |
| G2 per-material smoke | 5 min |
| G3 full sweep | 15 min compute + analysis |
| Write step4_review.md | 45 min |
| **Total** | **~3 hr** |

## What could go wrong

| Risk | Symptom | Fix |
|---|---|---|
| Subprocess invocation slow (10 s/case startup) | G3 takes 20 min instead of 10 | Acceptable; if much worse, refactor to in-process |
| Some test cases missing `best_*.pth` | Per-case crash | Validate case list against `experiments/<case>/train/` before launching |
| Cloth ramp produces NaN (per Step 3 finding) | NaN in summary | Catch + log; mark `any_nan=True`; continue to other cases |
| Per-material aggregation divides by zero | If only 1 case per material | Use `np.nanstd` with `ddof=0`; report n_cases |
| Sloth `single_push_sloth` accidentally included | Force scale blows up | Hardcoded exclusion in CASES list |
| `default policy_dir` points to the wrong seed | Wrong policy used | CLI arg with explicit default `models_policy/seed_1`; assert it exists at startup |

## Acceptance criteria

Step 4 is done when:

1. `results/eval_closed_loop/` contains all 14 rollout npzs.
2. `results/eval_closed_loop/summary.json` exists with per-rollout and
   per-material aggregates.
3. G1, G2, G3 all passed and outcomes recorded in
   `step4_review.md`.
4. Per-material replay tables in `step4_review.md` — these are the
   headline numbers for Step 5 figures.
5. Ramp-failure characterization documented (overshoot ratios, NaN
   count per material) — these are the "limitation" figures.
6. Any new lessons logged.

## Out of scope for Step 4

- Figures and videos — Step 5.
- Step / sinusoid profiles — narrowed scope.
- Cross-case (held-out trajectory) evaluation.
- Multi-seed sweep (use seed_1 throughout — the policy is the same model
  Step 3 used).
- Any model retraining or new fixes.
