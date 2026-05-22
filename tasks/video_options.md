# Demo video options (to build AFTER results plateau)

We're parking this until the modeling is tightened. Order is increasing effort.

## Option A — raw input video (zero work)
`data/different_types/<case>/final_data.mp4` is already on disk for every case
(downloaded with `data.zip`). Pick 3 representative cases (one per material)
and drop them straight into the slides as the "input side" of the story.

Suggested picks:
- rope: `single_push_rope_1`
- cloth: `single_lift_cloth_3`
- sloth: `double_stretch_sloth`

## Option B — PhysTwin's force-visualization video (~5 min/case on GPU)
`visualize_force.py` already produces `experiments/<case>/force_visualization.mp4`
— gaussian-rendered object on the real frame with **red ground-truth force
arrows** at each control point.

Smoke test sbatch already calls extract_dataset only; uncomment the
`visualize_force.py` line in `slurm/smoke_test.sbatch` (was tested earlier).
Run on 3 representative cases.

## Option C — predicted-vs-GT side-by-side overlay (~1 hour)
Build `visualize_predicted_force.py`:
1. Run `extract_force_data` to harvest positions per frame.
2. Compute the 18 summary features at each frame, forward-pass the trained
   MLP-per-type to get predicted `[T, 3]` net force.
3. Re-use PhysTwin's `getArrowMesh` + offscreen open3d renderer (already
   used in `visualize_force`) to draw TWO arrows per frame at the
   control-point centroid: **black GT, blue prediction**.
4. Write mp4 either as overlay or as side-by-side pane.

This is Figure 4 in the original `experiment_plan.md`. High demo impact:
visually shows the MLP tracking GT closely on rope, lagging on sloth.

## Build order when we come back
1. Option A first (just file copies) → already-usable B-roll.
2. Option B for the slide that shows "this is what PhysTwin recovers" → run
   on 3 cases via existing sbatch, ~15 min total.
3. Option C for the headline slide → ~1 hour development + ~15 min GPU.
