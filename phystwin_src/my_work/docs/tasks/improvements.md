# Result-improvement plan

**State (2026-05-21):** Within-case MLP-per-type R² = 0.56 (rope), 0.44 (cloth),
−0.02 (sloth). Cross-case R² uniformly negative. We have NOT yet exhausted the
data we already have — the next moves are about *using it better*, not just
collecting more.

---

## What's wrong with the current results (root causes)

1. **Throwing away the contact position.** `extract_dataset.py` stores
   `controller_pos [T, K, 3]` in the npz but the model never sees it. The
   model is asked "given object deformation, predict force" without knowing
   *where* the gripper is touching. That's missing information.
2. **Throwing away velocity.** We only encode displacement-from-rest. Spring
   forces depend on stretch AND damping × velocity (`dashpot_damping` is a
   tuned param in PhysTwin). Velocity-based features should add signal.
3. **Force-magnitude variance across cases of same material.** Rope cases
   range 2.5–17 kN; cloth 9–62 kN; sloth 27–114 kN. Single-shot regression
   to absolute force values is fighting this. Predicting *normalized*
   force, or log-magnitude + direction separately, sidesteps it.
4. **One sloth case (`single_push_sloth`) has 866 kN spike** — spring-mass
   numerical instability. Skews all sloth metrics. Outlier handling needed.
5. **17 cases is small.** Even with all the above fixes, generalization
   across material-internal trajectory variation needs more data. There are
   5 PhysTwin cases we haven't extracted (zebra ×3, dinosaur, package); plus
   PokeFlex (~hundreds of trajectories with real measured force) sits in the
   plan as Plan B.

---

## Improvements ranked by **ROI** (impact × ease)

Each row: **what** | **expected R² lift (gut estimate)** | **effort** | **risk**

### Tier 1 — do these first (cheap, high signal)

| # | Improvement | Δ R² | Effort | Risk |
|---|---|---|---|---|
| **T1.1** | **Add controller_pos to features** (centroid of K controller points; distance from controller to closest object point; relative motion of controller vs object) → adds ~6 features | **+0.10 to +0.20** | 30 min | low |
| **T1.2** | **Add velocity features** (per-frame Δposition: centroid, max, mean magnitude) → adds ~4 features | +0.05 to +0.15 | 20 min | low |
| **T1.3** | **Outlier handling**: clip per-frame force magnitudes to per-case 99th percentile before training | sloth R² −0.02 → ~+0.2 | 10 min | low |
| **T1.4** | **Predict log-magnitude + unit direction separately** instead of raw 3-vector. Two outputs: scalar `log(‖F‖+ε)` + 3D unit direction (with cosine loss). De-couples scale from direction. | +0.10 on rope/cloth, larger on sloth | 1 hr | medium |
| **T1.5** | **Multi-seed evaluation** — current is seed=42 only. Report mean ± std over 5 seeds. Lets us tell "real signal" from "lucky seed." | reveals noise | 15 min | low |

### Tier 2 — moderate effort, may unlock cross-case

| # | Improvement | Δ R² | Effort | Risk |
|---|---|---|---|---|
| **T2.1** | **Particle-count as feature** + **bbox absolute dimensions at rest** — gives the model a size prior for the object | small but useful for cross-case | 15 min | low |
| **T2.2** | **Per-case force normalization** at training time: divide y by case-mean `‖F‖`, predict relative force; require scale estimate at inference | could lift cross-case R² from negative to ~0.2 | 1 hr | medium (need scale at inference) |
| **T2.3** | **LSTM/Transformer over time** (8-frame context window). Currently model is frame-independent. Temporal context should help — forces in PhysTwin have inertia. | +0.10 on within-case | 2-3 hr | medium |
| **T2.4** | **DeepSets over particles** — replace summary features with a permutation-invariant set encoder. Captures spatial structure we're collapsing. | +0.10 to +0.20 within-case | 3-4 hr | medium-high (overfitting on 17 cases) |

### Tier 3 — more data (do after Tier 1, before Tier 2.4)

| # | Improvement | Δ R² | Effort | Risk |
|---|---|---|---|---|
| **T3.1** | **Extract the remaining PhysTwin cases** (zebra ×3, dinosaur ×1, package ×1) — 5 more trajectories, 2 new material categories ("toy", "dinosor", "package") | small per material if it's just 1 case; helps if we add toy as a 4th type with 3 cases | 30 min (re-run sbatch with extended CASES) | low |
| **T3.2** | **PokeFlex dataset** (arxiv 2409.17124) — robot arm pokes 18 deformable objects with measured force/torque at high rate. Likely hundreds of trajectories. Real measurements (no PhysTwin numerical artifacts). | unlocks honest cross-case generalization | 4-8 hr (download, parse, re-design extractor for PokeFlex schema) | low — pipeline transfers cleanly |
| **T3.3** | **Augment via temporal sub-sampling**: every 17-frame trajectory → ~3 stride-2 sub-trajectories. Doesn't add new physics but improves sample efficiency for time-context models. | small | 30 min | low |

### Tier 4 — speculative

- **Domain adaptation** between PhysTwin (sim) and PokeFlex (real) via a small alignment layer.
- **Pretrain on PokeFlex, fine-tune on PhysTwin** — get the deformation→force prior from a large dataset, specialize per case.
- **Active learning loop** — train MLP, find timesteps where MLP disagrees with PhysTwin, run more PhysTwin optimization on those frames.

---

## Recommended next session order

1. **T1.3 (outlier clipping) + T1.1 (controller features) + T1.2 (velocity features)** — all together, one rerun of extract_dataset.py + train_models.py. ~1 hour of editing + ~10 min of compute. This is the **single highest-leverage step**.
2. **T1.5 (multi-seed)** — wrap the existing train pipeline in a 5-seed loop. Confirms whether the Tier 1 lifts are real.
3. **T3.1 (more PhysTwin cases)** — cheap data expansion. 22 cases instead of 17, plus possibly a 4th material (toy/zebra).
4. **T1.4 (log-magnitude + direction)** — refactor `train_models.py` to two heads. Compare against vector regression baseline.
5. If we still need more: **T3.2 (PokeFlex)** is the big-ticket move.

---

## Specific code edits queued

Below are concrete diffs ready to drop in once we start.

### extract_dataset.py additions

```python
# In extract_case(), after loading positions:
# --- Tier 1.1: controller features ---
ctrl = controller_pos                                          # [T, K, 3]
ctrl_centroid = ctrl.mean(axis=1)                              # [T, 3]
ctrl_disp = ctrl_centroid - ctrl_centroid[0:1]                 # [T, 3]
# Distance from controller centroid to nearest object particle
obj_to_ctrl = np.linalg.norm(positions - ctrl_centroid[:, None, :], axis=-1)  # [T, N]
nearest_dist = obj_to_ctrl.min(axis=1)                         # [T]
mean_contact_dist = obj_to_ctrl.mean(axis=1)                   # [T]
# Relative motion ctrl vs centroid
obj_centroid = positions.mean(axis=1)                          # [T, 3]
rel_motion = ctrl_centroid - obj_centroid                      # [T, 3]

# --- Tier 1.2: velocity features ---
disp = positions - positions[0:1]                              # [T, N, 3]
vel = np.zeros_like(disp); vel[1:] = disp[1:] - disp[:-1]      # [T, N, 3]
vel_mag = np.linalg.norm(vel, axis=-1)                         # [T, N]
mean_vel_mag = vel_mag.mean(axis=1)                            # [T]
max_vel_mag = vel_mag.max(axis=1)                              # [T]
centroid_vel = vel.mean(axis=1)                                # [T, 3]
```

Stack into `X`: 18 (existing) + 3 (ctrl_disp) + 1 (nearest_dist) + 1 (mean_contact_dist) + 3 (rel_motion) + 1 (mean_vel_mag) + 1 (max_vel_mag) + 3 (centroid_vel) = **31 features**.

### train_models.py additions

```python
# Tier 1.3: outlier clipping
def clip_outliers(cases: dict, percentile: float = 99.0):
    for n, info in cases.items():
        mag = np.linalg.norm(info["y"], axis=-1) if info["y"].ndim == 2 else np.linalg.norm(info["y"].reshape(-1, 3), axis=-1)
        cap = np.percentile(mag, percentile)
        scale = np.where(mag > cap, cap / np.maximum(mag, 1e-6), 1.0)
        info["y"] = info["y"] * scale[:, None]
    return cases

# Tier 1.4: log-mag + direction heads (model)
class ForceMLP_v2(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                              nn.Linear(hidden, hidden), nn.ReLU())
        self.trunk = trunk
        self.mag_head = nn.Linear(hidden, 1)   # log(‖F‖+1)
        self.dir_head = nn.Linear(hidden, 3)   # unit direction
    def forward(self, x):
        h = self.trunk(x)
        return self.mag_head(h), torch.nn.functional.normalize(self.dir_head(h), dim=-1)

def loss_v2(pred_mag, pred_dir, y):
    mag_true = torch.norm(y, dim=-1, keepdim=True)
    log_mag_true = torch.log1p(mag_true)
    dir_true = torch.nn.functional.normalize(y + 1e-6, dim=-1)
    return ((pred_mag - log_mag_true) ** 2).mean() + (1 - (pred_dir * dir_true).sum(-1)).mean()
```

### Multi-seed wrapper

```python
# Tier 1.5
def run_seeds(seeds=(0, 1, 2, 3, 4), **kw):
    rows = []
    for s in seeds:
        set_seed(s)
        m = run_pipeline(seed=s, **kw)
        for model in m["results"]:
            for cat in m["results"][model]:
                rows.append({"seed": s, "model": model, "cat": cat,
                             "r2": m["results"][model][cat]["r2"]})
    return pd.DataFrame(rows).groupby(["model","cat"]).agg(["mean","std"])
```

---

## Do we need more data?

Short answer: **not yet**. Tier 1 should be done first — there's signal
sitting on disk we're discarding (controller position, velocity). Tier 3.1
(remaining 5 PhysTwin cases) is essentially free; do it concurrently.

We'd really need PokeFlex (T3.2) only if Tier 1 + T3.1 still leave cross-case
R² negative. The PhysTwin authors themselves don't have many more cases —
22 total is the dataset. So scaling up means going to a different data
source, which is the PokeFlex move.
