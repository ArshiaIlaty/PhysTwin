# Step 2 review — Train MLP policy

Plan: [step2_plan.md](step2_plan.md). Script:
[../../code/train_policy.py](../../code/train_policy.py). Models:
[../../results/models_policy/](../../results/models_policy/).

## Status

✅ Done. All 5 verification gates passed. All 6 (material × group) val
metrics beat the plan thresholds.

## Verification gate outcomes

| Gate | Time | Result |
|---|---|---|
| G1 dataloader smoke | <1 s | PASS — 27,834 rows after dropping toy, 79.8/20.2 train/val, scaled mean abs ≤ 1e-5, scaled std mean = 1.0 |
| G2 tiny overfit (100 rows × 200 epochs) | ~16 s | PASS — final loss 5.5e-4 (threshold 1e-3) |
| G3 single seed × 20 epochs | ~7 s | PASS — all (material, group) val vec_R² ≥ +0.47, val loss monotonically decreasing |
| G4 inspect 10 val predictions | <1 s | PASS — magnitudes in expected mm scale; 2 findings logged below |
| G5 full 5 seeds × 100 epochs | ~80 s total | PASS — full table below |

## Headline results (G5, mean ± std over 5 seeds)

| Material | Group | n val rows | vec_R² | mag_R² | dir_cosine |
|---|---|---:|---:|---:|---:|
| cloth | 0 | ~3,000 | +0.54 ± 0.02 | +0.69 ± 0.03 | +0.59 ± 0.03 |
| cloth | 1 | ~400 | +0.65 ± 0.13 | +0.65 ± 0.14 | +0.75 ± 0.07 |
| **rope** | **0** | ~1,500 | **+0.82 ± 0.05** | **+0.81 ± 0.05** | **+0.82 ± 0.04** |
| rope | 1 | ~500 | +0.76 ± 0.12 | +0.71 ± 0.14 | +0.85 ± 0.05 |
| sloth | 0 | ~1,000 | +0.54 ± 0.07 | +0.57 ± 0.11 | +0.57 ± 0.02 |
| sloth | 1 | ~600 | +0.55 ± 0.07 | +0.58 ± 0.10 | +0.57 ± 0.04 |

All entries beat the plan thresholds (vec_R² ≥ 0.3, dir_cos ≥ 0.5).
Seed-to-seed std ≤ 0.15 across all entries — stable.

Most early-stopping happened around epochs 25–45, confirming 100 epochs
was generous.

### Per-material interpretation

- **Rope is easiest** (R² ≈ 0.82). Rope motions are gentle and largely
  axis-aligned; the inverse mapping is closest to linear.
- **Cloth group 0 is solid and rock-stable** (R² 0.54 ± 0.02, lowest std
  of any entry). Group 1 is more variable (std 0.13) because there are
  only ~400 val rows and they're concentrated in a few double-control
  cases.
- **Sloth is the noisiest** (R² 0.55, std 0.07). Force scales are 10×
  larger than rope but data is smaller (4,858 rows total). Expected.

## Hiccups & fixes

### F1 — `load_dataset()` missing `n_ctrl_parts`
**Symptom**: G4 inspect crashed with `KeyError: 'n_ctrl_parts'`.
**Root cause**: I filtered `n_ctrl_parts` out of the returned dict because
the training forward pass didn't use it, but the inspect code printed it.
**Fix**: Added `n_ctrl_parts` to the returned dict in `load_dataset`.
Logged as a lesson: dataset loaders default to including everything.

### F2 — masked outputs are not "near zero"
**Observation**: G4 showed `mean ‖pred_g1‖ = 1.126 mm` for single-control
val rows. The plan optimistically expected "near 0."
**Root cause (architectural, not a bug)**: The masked MSE loss never
backprops through group-2 outputs for single-ctrl rows, so the network
has no incentive to output zero there. Whatever the linear head produces
is fine from the loss's perspective.
**Fix**: This is a Step 3 contract, not a Step 2 fix. The closed-loop
driver MUST multiply policy outputs by an `n_ctrl_parts`-derived mask
before applying to the simulator. Added to lessons.md and todo.md.

### F3 — input dim arithmetic error in the plan
**Symptom**: First draft of `step2_plan.md` said `Input dim = 37` for
`31 + 6 + 6` (actually 43).
**Root cause**: Wrong arithmetic in prose.
**Fix**: Corrected to 43 in both the plan and the code; added
`assert features.shape[1] == 43` in `load_dataset` so any future schema
change will surface immediately.

## Artifacts

```
results/models_policy/
  seed_0/  policy.pt (~330 KB), feat_scaler.pkl, target_scalers.pkl, metrics.json, train_log.json
  seed_1/  ...
  seed_2/  ...
  seed_3/  ...
  seed_4/  ...
  summary.json   ← per-material per-group mean ± std across seeds
```

Plus a throwaway `models_policy_quick/` from G3, kept for reference.

## Acceptance criteria check

From [step2_plan.md](step2_plan.md):

- [x] `results/models_policy/seed_{0..4}/policy.pt` exist and load cleanly.
- [x] `summary.json` has mean ± std per material per group per metric.
- [x] All 5 G-gates passed (this file documents outcomes).
- [x] Per-material val action R² ≥ 0.3 for cloth — actually ≥ 0.54 for
      every (material, group), much stronger than the floor.
- [x] Lessons logged in [lessons.md](lessons.md) (3 new entries).

## Carry-forward to Step 3 (closed-loop driver)

1. **Mask policy outputs by `n_ctrl_parts` before applying to the
   simulator** (F2 above). For single-ctrl cases, ignore the group-2
   output of the policy. For double-ctrl, use both.
2. **Reuse `target_scalers.pkl` and `feat_scaler.pkl` at inference**.
   The driver must look up the case's material to pick the right target
   scaler. Recommendation: load all scalers at driver init, pass material
   into the per-step inference function.
3. **The per-step inference contract is**:
   ```
   features = [state(31), force_now(6), force_goal(6)]   # 43-dim
   features_scaled = (features - feat_scaler.mean) / feat_scaler.std
   pred_scaled = model(features_scaled)                  # 6-dim
   pred_action = (pred_scaled * target_scalers[material].std
                  + target_scalers[material].mean)
                 .reshape(2, 3)
   pred_action *= action_mask                            # F2 fix
   ```
4. **Pick a strong seed for the demo** — seeds 1 and 3 had the best rope
   R²; pick whichever has lowest val loss for the cleanest demo. Verify
   from `seed_*/metrics.json::best_val_loss`.
5. **Per-step inference cost is microseconds** on CPU — no GPU needed
   for the policy itself. Step 3's bottleneck will be the PhysTwin
   forward simulator, not the model.
