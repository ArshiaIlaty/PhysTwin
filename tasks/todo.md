# PhysTwin force-from-deformation — implementation tracker

Source plan: `experiment_plan.md`. Project rules: `AGENTS.md`.

## 2026-05-21: Explain current experiment process and results
- [x] Read `tasks/lessons.md` and `tasks/review.md`.
- [x] Verify target arrays and feature definitions from the actual code.
- [x] Write a plain-English Markdown explanation covering pipeline, R², target choice, and feature meanings.
- [x] Review the generated Markdown for accuracy against `tasks/review.md` and source code.

Review: Added `tasks/experiment_explanation.md`. Verified from source that
reported v3.1 metrics use `dataset_v2`, `target=net`, `scaler_mode=per_cat`,
and `exclude_cases=single_push_sloth`; verified feature definitions from
`extract_dataset.py` and `augment_dataset.py`.

## 2026-05-21: Add Ridge and `extract_force_data()` details
- [x] Add a plain-English explanation of Ridge regression.
- [x] Add step-by-step details for what `extract_force_data()` does.
- [x] Verify the added writeup against `trainer_warp.py`.

Review: Expanded `tasks/experiment_explanation.md` with Ridge regression and
the detailed `extract_force_data()` pipeline requested by the user.

## 2026-05-21: Add synthetic trajectory generation explanation
- [x] Read `phystwin_src/generate_synthetic.py`.
- [x] Check current `dataset_synth_raw` artifacts.
- [x] Add synthetic trajectory generation section to `tasks/experiment_explanation.md`.

Review: Documented donor cases, synthetic controller motion patterns, sanity
gates, saved schema, and why synthetic trajectories address cross-case data
scarcity.

## Locked decisions (2026-05-20)
- **ML target**: net wrench at control points `[T, n_ctrl_parts, 3]`.
- **Features**: fixed-dim summary statistics over particles (centroid disp, bbox deformation, max/mean per-axis disp, kinetic-energy-like terms). Generalizes across rope/cloth/sloth despite varying `N_particles`.
- **Data hook**: add non-rendering `extract_force_data()` to `qqtt/engine/trainer_warp.py`.
- **Stage 0 + Stage 1 are combined** (smoke test + data introspection in same session).
- **Compute**: SLURM `free-gpu --account=mgamalel`.

## Critical analysis (what changed vs original plan)
1. Original plan's `np.concatenate(all_X, axis=0)` would crash because particle counts vary across cases. Resolved by switching to summary features.
2. Plan ambiguously says "force" — locked to net control-point wrench (not per-particle internal forces).
3. Plan suggested patching `visualize_force` as a side effect; cleaner separation via a dedicated extraction method.
4. Per-type MLP for sloth (1 case) is overfit-by-construction. Frame as "case-specific signature" demo, not generalization claim.
5. CUDA version is 12.2 (cluster module), not 12.1 (plan). Verify Warp/torch wheel compatibility before committing.

## Stage 0+1: Setup + smoke test + introspection
- [x] Survey cluster (CUDA modules, GPU partitions, slurm account)
- [x] Lock plan decisions with user
- [ ] Clone PhysTwin repo
- [ ] Read `env_install/env_install.sh` and figure out the minimal subset
- [ ] Create conda env at `/pub/mgamalel/envs/phystwin` with python 3.10
- [ ] Install inference-only deps (skip Trellis / Grounding-SAM-2 / RealSense / SDXL per README)
- [ ] Download 4 zips (data, experiments_optimization, experiments, gaussian_output) → `/share/crsp/lab/selmalak/mgamalel/PhysTwin/`
- [ ] Submit smoke-test sbatch job: `visualize_force.py --case_name single_push_rope_1 --n_ctrl_parts 1`
- [ ] Document shapes of `final_data.pkl`, `optimal_params.pkl`, and the force tensor in `trainer_warp.py`
- [ ] Note exact lines where ctrl-point forces are computed

## Stage 2: Data extraction
- [ ] Add `extract_force_data(self, best_model_path, n_ctrl_parts) -> (positions, forces)` to `trainer_warp.py`
- [ ] Write `extract_dataset.py` driving the extractor across all target cases
- [ ] Implement summary-feature transformer (centroid disp, bbox stretch, max/mean disp, KE proxy) — fixed dim, e.g. 16-32 features
- [ ] Save per-case `.npz` with `X` (summary features), `y` (control-point wrench), metadata
- [ ] Sanity check: forces non-zero and time-varying

## Stage 3: Dataset assembly
- [ ] Load all `.npz` → unified `(X, y, labels)` arrays
- [ ] Plot force-magnitude distribution per material
- [ ] Plot force vs time per case (physical plausibility check)
- [ ] Train/test split BY CASE, not by timestep
- [ ] Fit + pickle StandardScalers

## Stage 4: Training
- [ ] Ridge baseline (linear)
- [ ] Unified MLP
- [ ] Per-type MLPs (rope, cloth; sloth → single-case fit, report as such)
- [ ] Fill the metrics table

## Stage 5: Demo
- [ ] Figure 1: GT vs predicted force-magnitude over time, one rep per type
- [ ] Figure 2: R² bar chart, model × type
- [ ] Figure 3: Per-component error box plots
- [ ] (Optional) Figure 4: predicted force vectors overlaid on PhysTwin video
- [ ] Rehearse 5-minute narrative

## Fallback (Plan B)
Trigger if Stage 2 extraction hits >8h with no working pipeline → switch to PokeFlex (arxiv 2409.17124).
