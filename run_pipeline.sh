#!/bin/bash
# End-to-end orchestrator for the force-from-deformation pipeline.
# Run from repo root after downloads are unzipped and the conda env is active.
set -euo pipefail

echo "[$(date)] Stage 2 — Extract per-case datasets"
python extract_dataset.py

echo "[$(date)] Stage 3 — Inspect extracted dataset"
python inspect_data.py --dataset_dir dataset

echo "[$(date)] Stage 4 — Train models"
python train_models.py --target per_ctrl --out_dir models

echo "[$(date)] Stage 5 — Generate figures"
python evaluate_models.py --models_dir models --fig_dir figures

echo "[$(date)] DONE."
ls -la figures/
cat models/metrics.json | python -m json.tool | head -80
