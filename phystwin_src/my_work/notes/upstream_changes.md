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

3. **`InvPhyTrainerWarp.run_policy(model_path, n_ctrl_parts, policy_fn, F_goal,
   feature_fn, max_frames=None)`** — new method added right after
   `extract_force_data` (around line 1879). The closed-loop driver for the
   policy extension project ([../docs/closed_loop_control/](../docs/closed_loop_control/)).

   Mirrors `extract_force_data`'s setup (load checkpoint, push spring /
   collide params, group controller points via KMeans, build per-group
   force-spring bookkeeping). The difference is the per-frame loop: instead
   of reading the recorded controller trajectory via
   `set_controller_target(i)`, it calls
   `policy_fn(state31, force_now_pad, force_goal_pad, frame_idx)` for a
   per-group centroid Δ, rigid-translates each group's K controller points
   by that Δ, then advances the simulator via
   `set_controller_interactive(prev_target, curr_target)` and
   `wp.capture_launch(forward_graph)`.

   Per-step force computation uses the **driven** controller positions
   (not `self.simulator.controller_points[t]` which holds the recorded
   trajectory), via `get_force_vector(...)`.

   Returns the same array schema as `extract_force_data` plus
   `meta["K"]` and `meta["group_ids"]` for downstream group-aware analysis.

   Used by [run_closed_loop.py](../code/run_closed_loop.py).
