# Step 2 plan — Train the MLP policy

Train a frame-independent MLP that maps `(state, force_now, force_goal)` →
per-group `Δctrl`. Multi-seed, per-category scaler, random-block within-case
split — same protocol as v3.1.

## Goal

End-state artifact: `results/models_policy/seed_{0..4}/` directories with
checkpoints + scalers + per-seed metrics, plus
`results/models_policy/summary.json` with mean ± std across seeds.

A "good" model can predict held-out frame Δctrl well enough that **Step 3's
closed-loop rollout is worth running**. Targeted metric: per-material
action R² on val ≥ 0.3 (cloth/rope/sloth). Below that, return to Step 2
before sinking time into the closed-loop driver.

## Inputs

`results/policy_dataset.npz` from [step1_plan.md](step1_plan.md), 28238
rows. Required arrays: `state[R,31]`, `force_now[R,2,3]`, `force_goal[R,2,3]`,
`action[R,2,3]`, `action_mask[R,2]`, `material[R]`, `case_name[R]`,
`motion_type[R]`.

## Outputs

```
results/models_policy/
  seed_0/
    policy.pt              # torch state dict
    feat_scaler.pkl        # StandardScaler over (state + forces)
    target_scalers.pkl     # dict: material -> StandardScaler over actions
    metrics.json           # per-material R², MSE, direction cosine on val
    train_log.json         # per-epoch train/val loss for debugging
  seed_1/ ...
  seed_2/ ...
  ...
  summary.json             # mean ± std across seeds, per-material
```

## Locked decisions

1. **Drop `toy` from policy training.** 404 rows, force residuals up to
   6 MN (zebra/dinosaur numerical instabilities, same root cause that made
   the prior project exclude them as synthetic donors). Re-evaluate after
   the baseline works. Recorded as a CLI default `--exclude_materials toy`.
2. **Architecture**: 2 hidden layers × 256 units, ReLU, linear output head.
   Same family as v3.1; small enough to train on CPU in seconds per epoch.
3. **Input dim = 43** = 31 (state) + 6 (force_now flat) + 6 (force_goal
   flat). No mask or n_ctrl_parts in input — zero-padded forces already
   signal "no second gripper."
4. **Output dim = 6**, reshaped to `[B, 2, 3]` for masked loss.
5. **Loss = masked MSE in scaled target space**:
   `mask = action_mask.unsqueeze(-1).expand_as(action)`,
   `loss = (mask * (pred - target_scaled)**2).sum() / mask.sum().clamp(min=1)`.
6. **Feature scaler**: one `StandardScaler` fit on train rows only.
7. **Target scaler**: per-material `StandardScaler` fit on train rows' raw
   actions (separate scaler for rope, cloth, sloth). Apply with material
   lookup at training and inference time.
8. **Train/val split**: random-block within-case, 20% val per case. Seed
   chooses the contiguous starting frame per case. Matches v3.1
   `random_block` protocol.
9. **Optimizer**: Adam, lr=1e-3, batch size=256, 100 epochs, early stopping
   patience=15 on val loss.
10. **Seeds**: 5 (0..4). Reported metric is mean ± std.
11. **Compute**: CPU on login node. 28k rows × ~100 epochs × 37 dims is
    trivial — no GPU needed. Confirmed: total expected runtime ≤ 5 minutes
    for the full 5-seed run.

## Why these match v3.1

The prior project converged on per-category target scaler as the single
biggest fix (rope R² 0.36 ± 0.19 → 0.65 ± 0.02). Same trick applies here
because actions also have per-material magnitude variance (rope p99
0.005 m/frame vs cloth p99 0.022 m/frame). Single shared scaler would
under-emphasize rope and over-emphasize cloth.

## Verification gates (do in order, abort early if one fails)

The whole point is to catch dataloader / model / scaler bugs in the first
~20 min before committing to the full multi-seed run.

### G1 — Dataloader smoke (no training, <1 min)

```bash
python my_work/code/train_policy.py --check_only --seed 0
```

What it does: load the npz, drop toy, split into train/val by random-block,
fit feature + per-material target scalers, compute scaled tensor shapes.

**Pass criteria** (script prints + asserts):
- After dropping toy: ~27,834 rows.
- Train rows ≈ 80% of total, val ≈ 20% of total, off by ≤1%.
- Train and val each contain rope, cloth, sloth (no material missing).
- Scaled feature mean ≈ 0, std ≈ 1 (per axis, on train).
- Per-material scaled action mean ≈ 0, std ≈ 1 per axis.
- No NaN in scaled tensors.

**If it fails**: do not proceed. Bug is in loading / splitting / scaling.

### G2 — Tiny overfit (5 min)

```bash
python my_work/code/train_policy.py --overfit_tiny --epochs 200 --seed 0
```

What it does: pick 100 rows (random sample), train+val on the same 100,
train 200 epochs.

**Pass criteria**: train loss drops below 1e-3 (scaled space). If the model
can't overfit 100 rows, the model+loss+forward pass is broken. Likely
bugs caught: wrong dim, mask broadcast wrong, optimizer not stepping,
loss not actually using `pred` and `target` from the same rows.

**If it fails**: do not proceed. Print first 5 (pred, target) pairs to
debug.

### G3 — Single-seed quick run (5–10 min)

```bash
python my_work/code/train_policy.py --seed 0 --epochs 20 \
    --out results/models_policy_quick
```

What it does: full dataset, single seed, 20 epochs. Saves model + scalers
+ metrics into `models_policy_quick/seed_0/`.

**Pass criteria**:
- Train loss decreases monotonically (allow occasional bumps).
- Val loss decreases for at least 5 epochs then plateaus (some
  generalization gap is fine, train ≪ val is a red flag).
- Per-material val R² ≥ 0 for at least one material (proves the policy
  is learning something useful, not just predicting mean).

**If it fails**:
- All-negative R²: usually means target scaler is fit per-material on
  train but evaluated on val with the same scaler — verify val rows are
  scaled using the train-fit scaler, not refit.
- Val loss increases: likely overfitting, but 20 epochs shouldn't overfit
  this dataset. Look for batch-norm bugs / dropout misuse.
- Train loss doesn't decrease at all: likely optimizer step missing
  (caught by G2, but worth re-checking).

### G4 — Inspect a few predictions (5 min, no compute)

```bash
python my_work/code/train_policy.py --inspect_only \
    --model_dir results/models_policy_quick/seed_0 --n_examples 10
```

What it does: load the saved model, run on 10 random val rows, print
`(material, case_name, pred Δctrl, true Δctrl, |pred-true|)` per row.

**Pass criteria** (eyeball check):
- Predicted action magnitudes are in the right ballpark (mm-scale per
  frame for rope, cm-scale for cloth).
- No predictions are wildly out of range (e.g. predicting 1 m/frame
  when target is 1 mm).
- For single-ctrl rows, group-2 predictions should be near zero (the
  model learned the padding convention).

**If it fails**: same debugging as G3. Add a single-row debug print to
the forward pass.

### G5 — Full multi-seed run (10–15 min)

```bash
python my_work/code/train_policy.py --seeds 0,1,2,3,4 --epochs 100 \
    --out results/models_policy
```

What it does: 5-seed run, full epochs, full data. Writes per-seed dirs +
`summary.json` with mean ± std per material per metric.

**Pass criteria**:
- All 5 seeds finish without crash.
- Per-material val R² mean (over seeds) ≥ 0.3 for rope, cloth, sloth.
  Below this we should iterate on architecture / features before
  committing to Step 3.
- Per-material direction cosine ≥ 0.5 on val.
- Seed-to-seed std of val R² ≤ 0.15 (indicates the model is reasonably
  stable, not lottery-ticket sensitive).

## Metrics (computed per material, on val, in **original (unscaled)** action units)

| Metric | Definition |
|---|---|
| MSE per axis | `mean((pred - true)^2)` over active group axes |
| Action magnitude R² | `1 - var(‖pred‖ - ‖true‖) / var(‖true‖)` |
| Direction cosine | `mean(<pred, true> / (‖pred‖ ‖true‖))` for ‖true‖ > 1e-6 |
| Per-material loss | Final scaled MSE on val rows of that material |

All metrics computed only on rows where `action_mask[g] == 1` for the
respective group axis.

## Script structure

One main script: `my_work/code/train_policy.py`. Subcommands via flags:

```python
def load_and_split(npz_path, seed, exclude_materials):
    """Load, drop excluded materials, random-block within-case split."""

def fit_scalers(train_dict):
    """Returns feat_scaler (single) and target_scalers (dict material -> Scaler)."""

def apply_scalers(rows_dict, feat_scaler, target_scalers):
    """Returns scaled tensors ready for the model."""

class PolicyMLP(nn.Module):
    def __init__(self, in_dim=43, hidden=256, out_dim=6): ...

def masked_loss(pred, target_scaled, action_mask): ...

def train_one_seed(seed, args): ...

def evaluate(model, val_loader, feat_scaler, target_scalers, materials_per_row): ...

def main():
    # parse args; dispatch to G1 / G2 / G3 / G5 paths
    if args.check_only: do G1 and exit
    if args.overfit_tiny: do G2 and exit
    if args.inspect_only: do G4 and exit
    # else: do G3 (single seed) or G5 (multi seed) based on --seeds vs --seed
```

Aim for ≤ 400 LOC.

## What could go wrong

| Risk | Symptom | Fix |
|---|---|---|
| Target scaler refit on val instead of reuse | val R² near 0 across the board | Save fit scalers after train; apply only `.transform()` on val |
| Material lookup at scaler-apply time mismatches train materials | KeyError on val | Assert val materials ⊆ train materials before scaling |
| Mask broadcast wrong shape | RuntimeError or silently-broken loss (group 2 contributions for single-ctrl rows) | Unit-test shapes: assert `(mask.unsqueeze(-1) * x).sum()` matches a manual loop on 3 rows |
| Random-block split puts entire case in val (small cases) | Train missing a case → scaler doesn't see that material | Enforce: every material has ≥1 train block; assert before training |
| Per-axis StandardScaler underflow (action variance ~1e-5 for rope) | Scaled targets near zero, model learns identity | Print scaler.scale_ in G1; if any axis has scale < 1e-6, switch to RobustScaler for that material |
| `material` field from npz is bytes not str (numpy quirk) | `mat == "rope"` always False | Cast to str() at load |

## Scope estimate

| Phase | Time |
|---|---|
| Write train_policy.py | 1.5 hr |
| G1–G4 verification | 30 min |
| G5 full multi-seed run | 15 min |
| Write step2_review.md | 30 min |
| **Total** | **~3 hr** |

## Acceptance criteria

Step 2 is done when:
1. `results/models_policy/seed_{0..4}/policy.pt` exist and load cleanly.
2. `results/models_policy/summary.json` has mean ± std per material per
   metric.
3. All 5 G-gates passed (record outcomes in [step2_review.md](step2_review.md)).
4. Per-material val action R² ≥ 0.3 for at least cloth (the easy case).
   Rope and sloth can fall short; if so, document and proceed — the
   closed-loop test in Step 3 is the real arbiter.
5. Lessons learned during G1–G5 are logged in [lessons.md](lessons.md).

## Out of scope for Step 2

- GRU / 1D-conv temporal models (only if MLP plateaus *and* closed-loop
  fails).
- Toy category training (deferred — broken data).
- Cross-case generalization eval (not the goal here).
- Hyperparameter sweep — pick reasonable defaults, don't grid search.
- Action ranking losses, GANs, diffusion — overkill at this data scale.
