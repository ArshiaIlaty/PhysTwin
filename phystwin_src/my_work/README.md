# my_work — force-from-deformation project

Everything in this folder is work I added on top of the upstream
[PhysTwin](https://github.com/Jianghanxiao/PhysTwin) source. Anything
outside `my_work/` (and outside `phystwin_src/qqtt/engine/trainer_warp.py`
— see [notes/upstream_changes.md](notes/upstream_changes.md)) is upstream.

## Layout

```
my_work/
├── code/         Python scripts (data extraction, training, evaluation, figures)
├── scripts/      Shell + Slurm orchestrators
│   └── slurm/    HPC batch submission scripts
├── results/      All generated artifacts (gitignored — datasets, models, figures, logs)
├── docs/         Experiment plan + working notes from the project
└── notes/        Notes about the upstream code I had to modify
```

## code/

| File | What it does |
| --- | --- |
| [extract_dataset.py](code/extract_dataset.py) | Drive PhysTwin inference across a list of cases → `results/dataset/*.npz` |
| [augment_dataset.py](code/augment_dataset.py) | Extend per-frame features (18 → 31) → `results/dataset_v2/*.npz` |
| [generate_synthetic.py](code/generate_synthetic.py) | Drive the calibrated simulator with new controller motions → `results/dataset_synth_raw/` |
| [inspect_data.py](code/inspect_data.py) | Print shapes / keys for `.pkl` / `.npz` files |
| [train_models.py](code/train_models.py) | Train Ridge + unified-MLP + per-material-MLP, write metrics & checkpoints |
| [evaluate_models.py](code/evaluate_models.py) | Generate demo figures (force-over-time, R² bars, error box plots) |
| [make_figures.py](code/make_figures.py) | v1 figure set (single-seed) |
| [make_figures_v2.py](code/make_figures_v2.py) | v2 figure set (extended features + multi-seed sweeps + v1↔v2 comparison) |

All Python scripts compute their default output paths relative to their own
location (`my_work/results/...`), so they work from any cwd as long as `qqtt`
imports resolve — typically run from the `phystwin_src/` repo root.

## scripts/

| File | What it does |
| --- | --- |
| [run_pipeline.sh](scripts/run_pipeline.sh) | End-to-end: extract → inspect → train → figures |
| [env_install_minimal.sh](scripts/env_install_minimal.sh) | Pip-only install path for inference + force visualization (skips TRELLIS, Grounded-SAM-2, RealSense, SDXL) |
| [slurm/extract_all.sbatch](scripts/slurm/extract_all.sbatch) | Slurm: extract per-case datasets |
| [slurm/smoke_test.sbatch](scripts/slurm/smoke_test.sbatch) | Slurm: smoke-test extraction on one case |
| [slurm/smoke_debug.sbatch](scripts/slurm/smoke_debug.sbatch) | Slurm: verbose variant for env / module debugging |
| [slurm/synth_gen.sbatch](scripts/slurm/synth_gen.sbatch) | Slurm: synthetic-trajectory generation |

## results/

Gitignored. Holds everything regeneratable:

- `dataset/`, `dataset_v2/`, `dataset_smoke/`, `dataset_synth_raw/`
- `models_*/` — checkpoints, scalers, metrics JSONs for each training variant
- `figures/`
- `slurm_logs/`

## docs/

- [experiment_plan.md](docs/experiment_plan.md) — high-level project plan
- [tasks/](docs/tasks/) — working notes (`improvements.md`, `lessons.md`, `review.md`, `todo.md`, etc.)

## notes/

- [upstream_changes.md](notes/upstream_changes.md) — what I modified inside the upstream PhysTwin source (currently `qqtt/engine/trainer_warp.py`)
