#!/bin/bash
# End-to-end orchestrator for the force-from-deformation pipeline.
# Run from the phystwin_src/ repo root after downloads are unzipped and the
# conda env is active. Scripts default to writing into my_work/results/.
set -euo pipefail

CODE=my_work/code
RESULTS=my_work/results

echo "[$(date)] Stage 2 — Extract per-case datasets"
python "$CODE/extract_dataset.py"

echo "[$(date)] Stage 3 — Inspect extracted dataset"
python "$CODE/inspect_data.py" --dataset_dir "$RESULTS/dataset"

echo "[$(date)] Stage 4 — Train models"
python "$CODE/train_models.py" --target per_ctrl --out_dir "$RESULTS/models"

echo "[$(date)] Stage 5 — Generate figures"
python "$CODE/evaluate_models.py" --models_dir "$RESULTS/models" --fig_dir "$RESULTS/figures"

echo "[$(date)] DONE."
ls -la "$RESULTS/figures/"
cat "$RESULTS/models/metrics.json" | python -m json.tool | head -80
