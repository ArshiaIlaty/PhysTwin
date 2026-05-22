# Modifications to upstream PhysTwin code

These edits live inside the upstream tree (not under `my_work/`) because
they touch files imported by the rest of the package.

## `qqtt/engine/trainer_warp.py`

1. **Lazy `pynput.keyboard` import** — the original top-level import
   `from pynput import keyboard` fails on headless nodes (no X display),
   which broke every training / batch-extraction job. The import is now
   inside `InvPhyTrainerWarp.interactive_playground`, the only method that
   uses it.

2. **`InvPhyTrainerWarp.extract_force_data(model_path, n_ctrl_parts=1)`** —
   new method (added after `interactive_playground_with_force_visualization`,
   around line 1757). Runs the inference simulation without rendering and
   returns per-timestep arrays:

   - `positions`      `[T, N_all_points, 3]` — simulated object particles
   - `forces`         `[T, n_ctrl_parts, 3]` — net wrench at each ctrl group
   - `controller_pos` `[T, N_ctrl, 3]`
   - `meta`           `dict(num_all_points, n_ctrl_parts, frame_len)`

   Used by [extract_dataset.py](../code/extract_dataset.py).
