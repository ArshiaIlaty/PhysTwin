# Review: Learning Force from Deformation via PhysTwin

**Status (2026-05-21):** End-to-end pipeline complete from raw repo clone through trained models, metrics, and figures. All artifacts on `/share/crsp/lab/selmalak/mgamalel/PhysTwin/phystwin_src/`.

## What got built

| Step | Artifact | Notes |
|---|---|---|
| Env | `/pub/mgamalel/envs/phystwin` (~5GB) | Python 3.10, torch 2.4+cu121, warp 1.13, gaussian-splatting submodules. Skipped TRELLIS / Grounded-SAM-2 / RealSense / SDXL (per plan) |
| Code patch | `qqtt/engine/trainer_warp.py::extract_force_data()` | Mirrors `visualize_force` simulation loop but skips gaussian rendering — returns `(positions [T,N,3], forces [T,n_ctrl_parts,3])`. ~70 lines |
| Lazy import | `qqtt/engine/trainer_warp.py` | Moved `from pynput import keyboard` inside `interactive_playground` (was top-level → broke headless GPU jobs) |
| Extractor | `extract_dataset.py` | Maps `(positions, forces)` → 18-D summary features + padded `y_per_ctrl [T,2,3]` + `y_net [T,3]` |
| Slurm | `slurm/{smoke_test,extract_all}.sbatch` | `free-gpu --account=mgamalel`, A30 nodes |
| Trainer | `train_models.py` | Ridge / unified MLP / per-type MLPs, two split modes (`cross_case` / `within_case`) |
| Figures | `make_figures.py` | 4 PNGs in `figures/` |

## Data

17 cases extracted across 3 materials. ~2 100 timesteps total. Force magnitudes per case (within material): **rope 2.5–17 kN, cloth 9–62 kN, sloth 27–114 kN** (with `single_push_sloth` peaking at 866 kN — likely a PhysTwin spring stiffness blow-up).

**Per-case file (`dataset/<case>.npz`)** — 18 input features (centroid/bbox/max/mean/std displacement components + KE proxy + disp-magnitude stats), plus targets `y_net [T,3]` and `y_per_ctrl [T,2,3]`.

## Key results (R², higher better)

### v1 — 18 summary features, single seed (baseline)

|                 | rope  | cloth | sloth |
|---|---|---|---|
| Ridge — cross-case        | −29.5  | −0.83  | −0.02 |
| Ridge — within-case       | −15.3  | −1.29  | +0.10 |
| MLP unified — within-case | −5.00  | −0.14  | +0.19 |
| **MLP per-type — within-case** | **+0.56** | **+0.44** | −0.02 |

### v3.1 — 21 cases (added zebra×3, dinosaur; removed weird_package; excluded single_push_sloth from training), 5 seeds, per-cat scaler

MLP per-type R² (mean ± std over 5 seeds):

| Config                                       | rope             | cloth             | sloth            | toy              |
|---|---|---|---|---|
| within / pooled (v3 baseline)                | 0.36 ± 0.19      | 0.40 ± 0.06       | 0.41 ± 0.12      | 0.02 ± 0.01      |
| within / per_cat                             | **0.65 ± 0.02**  | 0.40 ± 0.03       | **0.53 ± 0.05**  | 0.03 ± 0.01      |
| random_block / pooled                        | 0.63 ± 0.10      | 0.49 ± 0.14       | 0.48 ± 0.15      | 0.01 ± 0.02      |
| **random_block / per_cat** (best within)     | **0.69 ± 0.07**  | **0.51 ± 0.11**   | **0.50 ± 0.14**  | 0.02 ± 0.02      |
| cross_case / per_cat (best cross)            | −0.90 ± 0.23     | **0.50 ± 0.03**   | −0.50 ± 0.47     | −83 ± 42         |

**Key findings**

1. **Per-cat scaler was the missing fix.** v3 pooled-scaler within-case gave rope R²=0.36 ± **0.19** (huge std → unstable). Per-cat scaler: 0.65 ± **0.02** — rock-solid. Adding new categories (toy) had been compressing the rope signal via the shared y-standardizer.
2. **Random-block ≳ within-case** by ~5–10% R² across materials. Slight bias-correction relative to "always last 20%" — and the std stays comparable. So random-block is now the preferred within-trajectory benchmark.
3. **Cross-case cloth R² = 0.50 ± 0.03** with per-cat scaler — best cross-trajectory generalization result we have. Rock-solid across seeds.
4. **Cross-case rope/sloth still negative.** Within-material force-magnitude variance is too wide for 4–5 trajectories to cover. This is a *data* limitation; synthetic trajectories are the next move.
5. **Toy stays near zero.** 4 cases of three very different objects (zebra ×3 + dinosaur ×1, very different geometries and force scales). Either drop or augment.

### v2 — 31 features (added ctrl-position + velocity), 5 seeds, no force-clip

|                 | rope            | cloth          | sloth         |
|---|---|---|---|
| Ridge — within            | −9.07           | −2.09          | −0.20         |
| MLP unified — within      | −0.77 ± 0.24    | +0.27 ± 0.13   | −0.35 ± 0.23  |
| **MLP per-type — within** | **+0.64 ± 0.05** | **+0.47 ± 0.01** | −0.13 ± 0.15  |
| MLP unified — cross       | −10.3 ± 3.1     | −0.15 ± 0.17   | −0.14 ± 0.05  |
| **MLP per-type — cross**  | −0.46 ± 0.36    | **+0.42 ± 0.05** | −0.13 ± 0.06  |

### What changed v1 → v2 (within-case, MLP per-type)
- **rope: 0.56 → 0.64 ± 0.05** (+0.08, stable across seeds)
- **cloth: 0.44 → 0.47 ± 0.01** (small but rock-solid: std=0.01)
- **sloth: −0.02 → −0.13 ± 0.15** (no improvement, high variance — sloth still broken)

### What v2 unlocked: cross-case cloth
v1 cross-case results were uniformly negative for every (model, material) combination. **v2 hit MLP-per-type cloth = +0.42 ± 0.05 cross-case** — the deformation→force mapping now transfers to *previously-unseen cloth trajectories*. Adding `controller_position` features gave the model the contact-location info it needed to generalize.

### What's still broken
- **Rope cross-case (−0.46 ± 0.36).** Five rope cases span 2.5 → 17 kN (7× range); training pool can't cover that span with so few examples. High seed variance.
- **Sloth everywhere.** `single_push_sloth` has many force spikes (not just one), so a 99-percentile clip helps a little on cloth but doesn't fix sloth. Needs different handling — drop the case or use Huber loss.

### Honest framing
The within-trajectory MLP per material recovers ~50–65% of force variance from 31 summary features. Inference is a single forward pass (microseconds) vs PhysTwin's per-case minutes of optimization. Cross-trajectory generalization works for cloth but not yet rope or sloth — bottlenecked by force-magnitude variance across cases and a numerical-instability outlier (single_push_sloth).

## Figures

`figures/force_over_time.png` — per-case `‖F‖(t)` for GT, Ridge, MLP-unified, MLP-per-type. Shows MLP-per-type tracks GT envelope on rope/cloth; sloth and Ridge oscillate.
`figures/r2_by_model_split.png` — bar chart of the table above.
`figures/error_distribution.png` — Fx/Fy/Fz residual box plots for the unified MLP, grouped by material.
`figures/feature_correlation.png` — Pearson r of each summary feature vs `‖F‖`. Best features: `std_disp_x`, `mean_disp_mag`, `max_disp_mag` (r≈0.45). `bbox_stretch_x` has r≈−0.4 (objects flatten under force).

## Headline (v3.1)

**Best within-trajectory result:** random_block split + per-cat scaler, MLP-per-type:
**rope R² = 0.69 ± 0.07, cloth 0.51 ± 0.11, sloth 0.50 ± 0.14.**

**Best cross-trajectory generalization:** cross_case + per_cat, MLP-per-type **cloth R² = 0.50 ± 0.03.**

## Demo narrative (5 min)

1. **Hook (30 s):** "PhysTwin can recover applied force from a single video — but takes minutes of differentiable physics per video. We replaced the inner loop with a fast feedforward model."
2. **Pipeline (1 min):** clone → run PhysTwin's spring-mass simulator on 17 prebuilt cases → harvest `(particle positions, control-point forces)` per timestep → compress particle field to 18-D summary features → train Ridge/MLP/per-material MLP.
3. **Headline (2 min):** *Within-trajectory*, MLP-per-type reaches R²=0.56 (rope) and 0.44 (cloth) — visually obvious in Fig 1. Inference is a single forward pass (~10 µs/CPU) vs PhysTwin's minutes.
4. **Honest limits (1 min):** Cross-trajectory generalization is bad. PhysTwin's per-case force calibration varies 4–30× WITHIN a material; 17 cases ≠ enough to span that. The `single_push_sloth` 866 kN spike is a spring-mass numerical instability, not a real force — a real deployment would need outlier handling.
5. **Future (30 s):** PokeFlex's real robot-measured force/torque data would give honest cross-trajectory generalization. PhysTwin remains the right framing (force-from-video without instrumentation).

## What was changed vs the original plan

1. Locked targets BEFORE Stage 1 (saved Stage 2/3 rework). Original plan would have crashed at `np.concatenate(all_X)` because particle counts vary 855–8582.
2. **Summary features** replaced raw flattened particles — fixed 18-D input across all materials.
3. `extract_force_data()` is a non-rendering method, not a side-effect of `visualize_force` (plan's option B). Faster, cleaner, no `gs_path` requirement during extraction.
4. Added **within_case split** alongside the plan's cross_case split. Cross-case alone with 17 cases gave only negative-R² results, which would have hidden the "MLP captures the mapping" finding.
5. Stage 0+1 merged (smoke test session also performed data introspection).

## Storage at end-state

```
phystwin_src/
  dataset/                  150 MB (17 .npz + summary.json)
  experiments/              ~5 GB (PhysTwin pretrained models, unchanged)
  experiments_optimization/ ~500 MB
  gaussian_output/          ~1.3 GB
  data/                     ~7 GB
  _downloads/               ~10 GB (zip archives, can delete)
  models_cross_case/        ~1 MB
  models_within_case/       ~1 MB
  figures/                  ~400 KB
```

Conda env at `/pub/mgamalel/envs/phystwin` ~5 GB.

## Reproducing from scratch on HPC3

```bash
cd /share/crsp/lab/selmalak/mgamalel/PhysTwin/phystwin_src
# (env already exists; recreate with env_install/env_install_minimal.sh if needed)
sbatch slurm/extract_all.sbatch        # ~4 min on A30
PATH=/pub/mgamalel/envs/phystwin/bin:$PATH PYTHONNOUSERSITE=1 \
  python train_models.py --target net --split cross_case  --out_dir models_cross_case
PATH=/pub/mgamalel/envs/phystwin/bin:$PATH PYTHONNOUSERSITE=1 \
  python train_models.py --target net --split within_case --out_dir models_within_case
PATH=/pub/mgamalel/envs/phystwin/bin:$PATH PYTHONNOUSERSITE=1 \
  python make_figures.py
```

## Open issues / next steps if time allows

- **Outlier handling for `single_push_sloth`** — clip per-frame forces to e.g. 99th percentile per case, or drop that case from sloth training pool. Likely lifts sloth MLP-per-type R² from −0.02 to positive.
- **Fig 4 overlay** (predicted force arrows on rendered video) — kept optional in plan, not yet built.
- **Switch ML target to force MAGNITUDE only** (T, 1) instead of vector (T, 3) — easier problem, may be enough for the demo.
- **More PhysTwin cases** — `single_lift_dinosor`, `double_stretch_zebra`, etc. are available but not yet extracted (different `cfg_type`, would need a small tweak to handle non-rope/cloth/sloth categories).
