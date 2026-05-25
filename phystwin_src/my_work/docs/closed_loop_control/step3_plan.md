# Step 3 plan — Closed-loop driver

Add a method to PhysTwin's trainer that drives the simulator frame by frame,
asking a learned policy for the next controller motion at each step instead
of reading the recorded trajectory.

## Goal

End state: `run_policy(model_path, n_ctrl_parts, policy_fn, F_goal,
material)` method on `InvPhyTrainerWarp`, sibling to `extract_force_data`.
Takes a callable policy + a desired force trajectory, returns the actual
positions / forces / controller positions PhysTwin produced under
policy-driven control.

Plus a small CLI driver `my_work/code/run_closed_loop.py` that loads a
trained policy, builds the callable, and invokes `run_policy()` on a chosen
case.

This step is the **first step that actually closes the loop**. The point
is not to evaluate the policy (Step 4 / Step 5) — only to prove the
machinery works without crashing or producing nonsense.

## Inputs

- `experiments/<case>/train/best_*.pth` — same checkpoint
  `extract_force_data` consumes.
- `experiments_optimization/<case>/optimal_params.pkl` + standard PhysTwin
  per-case files (calibrate, metadata, final_data).
- `results/models_policy/seed_*/policy.pt` + `feat_scaler.pkl` +
  `target_scalers.pkl` from Step 2.
- `F_goal`: `[T, 2, 3]` numpy array — desired per-group force per frame.
  For G3/G4 below, this is the case's recorded `y_per_ctrl`.

## Per-step inference contract (locked, from Step 2 carry-forward)

```text
features = concat([state(31), force_now.flatten()(6), force_goal.flatten()(6)])  # 43
features_scaled = (features - feat_scaler.mean) / feat_scaler.std
pred_scaled = model(features_scaled)                                              # 6
pred_action = pred_scaled * target_scalers[material].std
              + target_scalers[material].mean
pred_action = pred_action.reshape(2, 3)
pred_action[n_ctrl_parts:] = 0          # F2 finding from Step 2: mask unused group
return pred_action                       # per-group centroid Δ in meters
```

The driver applies the per-group centroid Δ to all K controller points in
that group as a **rigid translation** (every point in group g gets the
same Δ). This matches how recorded controller motion behaves and how
`generate_synthetic.py` constructs synthetic data.

## Outputs

```text
positions       [T, N_particles, 3]    float32   simulated particle positions
forces          [T, n_ctrl_parts, 3]   float32   achieved per-group force
controller_pos  [T, K, 3]              float32   driven controller positions
meta            dict                             frame_len, n_ctrl_parts,
                                                 case_name, material, etc.
```

Same schema as `extract_force_data()` so the same plotting / video pipeline
in Step 4 / 5 will accept either source.

For the CLI driver, save to
`results/closed_loop_rollouts/{case}__{policy_seed}__{profile_name}.npz`.

## Architecture

### Method on `InvPhyTrainerWarp` (in `qqtt/engine/trainer_warp.py`)

```python
def run_policy(self, model_path, n_ctrl_parts, policy_fn, F_goal, material,
               max_frames=None):
    # 1. Setup (mirrors extract_force_data lines, factored if possible):
    #      - load checkpoint, push spring_Y / collide params into simulator
    #      - cluster controller points into n_ctrl_parts groups (KMeans)
    #      - allocate output buffers
    #      - reset simulator to frame 0
    #
    # 2. Record initial state (frame 0): positions, controller_pos, force_now.
    #
    # 3. Rollout loop t = 1..T-1:
    #      state31 = summary_features_31(positions[:t], controller_pos[:t])
    #      force_now_pad = pad_to_2(forces[t-1])             # [2, 3]
    #      delta_per_group = policy_fn(state31, force_now_pad,
    #                                  F_goal[t], material)  # [2, 3]
    #      delta_per_group[n_ctrl_parts:] = 0                # mask
    #      new_ctrl = controller_pos[t-1].copy()
    #      for g in range(n_ctrl_parts):
    #          new_ctrl[group_ids == g] += delta_per_group[g]
    #      simulator.set_controller_interactive(prev_target, new_ctrl)
    #      wp.capture_launch(simulator.forward_graph)
    #      positions[t]      = wp.to_torch(states[-1].wp_x).cpu().numpy()
    #      forces[t]         = _ctrl_forces(positions[t])
    #      controller_pos[t] = new_ctrl
    #
    # 4. Return (positions, forces, controller_pos, meta).
```

Place it next to `extract_force_data()` (~line 1757 in `trainer_warp.py`).
Document the edit in [../../notes/upstream_changes.md](../../notes/upstream_changes.md).

### Feature computation — REUSE, don't reimplement

The 31-D feature vector is computed in two pieces in the existing codebase:
- 18-D base summary features: `summary_features()` in
  [../../code/extract_dataset.py](../../code/extract_dataset.py)
- 13-D controller/velocity features added in
  [../../code/augment_dataset.py](../../code/augment_dataset.py)

The Step 3 driver MUST reuse these exact functions. Reimplementing risks a
silent feature mismatch (training on one definition, running on a slightly
different one → out-of-distribution input at every step → policy outputs
garbage).

Plan: factor `compute_31d_features(positions_so_far, controller_pos_so_far)`
into a small helper module (or import directly from `extract_dataset.py` +
`augment_dataset.py`) and call it from both the dataset builder and the
closed-loop driver.

### Policy callable

The CLI driver builds the callable as a closure over (model, feat_scaler,
target_scalers). `run_policy()` itself just calls it — no torch import in
`trainer_warp.py`'s new method, which keeps the upstream-touched code small
and decoupled.

```python
# in my_work/code/run_closed_loop.py
def make_policy_fn(model, feat_scaler, target_scalers):
    def policy_fn(state31, force_now_pad, force_goal_pad, material):
        feats = np.concatenate(
            [state31, force_now_pad.flatten(), force_goal_pad.flatten()]
        )
        x = (feats - feat_scaler["mean"]) / feat_scaler["std"]
        with torch.no_grad():
            y = model(torch.from_numpy(x).float().unsqueeze(0)).numpy()[0]
        ts = target_scalers[material]
        delta = (y * ts["std"] + ts["mean"]).reshape(2, 3)
        return delta
    return policy_fn
```

### CLI driver `my_work/code/run_closed_loop.py`

```text
python run_closed_loop.py
    --case_name        single_push_rope_1
    --policy_dir       results/models_policy/seed_1
    --profile          replay        # one of: random, zero, replay_action,
                                     #         policy_recorded_goal, policy_ramp
    --max_frames       150
    --out              results/closed_loop_rollouts/
```

The profile flag dispatches to one of 5 policy callables (see G1–G5 below).
This single script is reused across all gates.

## Verification gates

Each gate exercises one layer. Earlier gates don't depend on the policy —
they isolate the simulator-driving logic from the policy itself. **Cost
per gate is ~1–2 minutes** of rollout time.

### G1 — Random-policy smoke (1–2 min)

```bash
python run_closed_loop.py --case_name single_push_rope_1 --profile random
```

policy_fn returns a small random Δ per group per frame (uniform in
[-1e-3, +1e-3] m). No model loaded.

**Pass criteria:**
- 150 frames complete without crash.
- No NaN in positions or forces.
- Final position bounded: ‖positions[-1] - positions[0]‖_∞ < 1.0 m per
  particle (object hasn't exploded).

**If it fails**: the tensor layouts of `set_controller_interactive` are
wrong, or the KMeans group assignment is degenerate, or the per-step
recording is broken. Fix the driver — none of this depends on the policy.

### G2 — Zero-policy stationarity (1–2 min)

```bash
python run_closed_loop.py --case_name single_push_rope_1 --profile zero
```

policy_fn returns Δ = 0 every frame.

**Pass criteria:**
- All controller positions equal initial position: `‖controller_pos[t] -
  controller_pos[0]‖_∞ < 1e-6 m` for all t.
- Particle drift is small. For a hanging cloth case, frame 0 → 150
  particle drift bounded by ~10 cm under gravity (no controller motion
  means the object falls). For pushed rope, drift bounded by ~1 cm
  (rope is at rest if untouched).

**If it fails**: the driver is applying a non-zero translation when given
Δ=0. Probably a mask bug or accumulation bug. Fix before G3.

### G3 — Replay-action sanity (2 min) — **most important gate**

```bash
python run_closed_loop.py --case_name single_push_rope_1 --profile replay_action
```

policy_fn returns Δ = recorded `controller_centroid[t+1] -
controller_centroid[t]` (per group). No model — just read recorded actions
and feed them in via the closed-loop driver.

**Pass criteria:**
- `‖positions(driver) - positions(extract_force_data)‖_∞ < 1e-3 m` per
  particle per frame. Means: the closed-loop driver, fed the same actions
  as the recorded trajectory, produces the same simulation result.
- `‖forces(driver) - forces(extract_force_data)‖_∞ / ‖forces(recorded)‖_∞
  < 1e-2`. (Force is a derived quantity, can drift slightly with float32
  roundoff.)

**If it fails**: the per-group centroid → per-controller-point translation
isn't matching what `set_controller_target(i)` does. Likely candidates:
group assignment differs from extract_force_data's grouping, rigid-
translate convention is wrong, or `set_controller_interactive` accepts a
different tensor layout than expected.

**Why this matters**: G3 is the bridge between the recorded-data world
(extract_force_data) and the policy-driven world (run_policy). If it
passes, we've proven that the only difference between "PhysTwin running
the recorded trajectory" and "PhysTwin running an arbitrary trajectory"
is the action source. The downstream gates can then focus on the policy.

### G4 — Trained policy, recorded F_goal (2 min)

```bash
python run_closed_loop.py --case_name single_push_rope_1 \
    --policy_dir results/models_policy/seed_1 \
    --profile policy_recorded_goal
```

policy_fn = real trained model. F_goal = the case's recorded `y_per_ctrl`.
The policy is asked: "achieve the force that was actually achieved on this
case."

**Pass criteria:**
- Simulation completes without explosion (same boundedness checks as G1).
- Achieved force magnitude tracks the recorded force magnitude
  qualitatively. Quantitative target:
  `mean ‖F_achieved - F_recorded‖ / mean ‖F_recorded‖ < 0.5` (within 50%
  RMS, generous).
- Final controller position is "close to" final recorded controller
  position. Specifically, `‖controller_centroid[-1, 0] -
  recorded_controller_centroid[-1, 0]‖ < 0.1 m`. This is loose by design
  — small per-step errors compound and we expect the final gripper
  position to drift.

**If it fails (controller flies off):** likely the policy is producing
out-of-distribution actions at some frame, the simulator is amplifying
them, and runaway compounding kicks in. Check the per-frame max
‖pred_action‖ for spikes.

**If it fails (force tracking is poor but stable):** the policy isn't
good enough yet on this case. Try a different seed, or note that this
case's training data was particularly noisy. Move on to G5.

### G5 — Trained policy, simple ramp F_goal (2 min)

```bash
python run_closed_loop.py --case_name single_push_rope_1 \
    --policy_dir results/models_policy/seed_1 \
    --profile policy_ramp --target_peak_force 5.0
```

policy_fn = trained model. F_goal = linear ramp 0 → peak → 0 over 150
frames, applied to group 0 (group 1 = zero). Peak chosen relative to the
case's force range.

**Pass criteria** (loose, this is the diagnostic gate):
- Simulation completes.
- Achieved force has the right *shape* — rises, peaks roughly in the
  middle, falls. Not necessarily the right magnitude.
- Gripper trajectory is physically sensible (no jumps, monotonically-ish
  motion).

**G5 is the first real "novel target" eval.** Whether the policy can
hit the peak magnitude is a Step 4 question, not Step 3 acceptance.
Step 3 just needs to prove the loop doesn't break under novel targets.

## Cases to use for G3/G4/G5

Use **`single_push_rope_1`** as the primary test case:
- Rope had the best policy metrics (vec_R² 0.82).
- Single-control → simpler bookkeeping, less risk of group-assignment
  bugs.
- Recorded force is in the gentle 2.5–17 kN range.
- It was in training (80% of frames seen) → in-distribution test, the
  easiest setting per the user's request.

Secondary test for G4 only: **`double_lift_cloth_3`** to exercise the
double-control path before Step 4. Different gate, no acceptance impact
yet — just want to know if double-control even runs without crashing.

## Scope estimate

| Phase | Time |
|---|---|
| Factor `compute_31d_features` helper from existing scripts | 30 min |
| Write `run_policy()` in `trainer_warp.py` (mirror extract_force_data) | 2 hr |
| Write `run_closed_loop.py` CLI driver | 1 hr |
| G1 smoke | 5 min |
| G2 zero | 5 min |
| G3 replay (the bug-finding gate) | 30 min (may need to debug) |
| G4 trained policy | 15 min |
| G5 ramp | 15 min |
| Write step3_review.md | 30 min |
| Document upstream edit in `notes/upstream_changes.md` | 5 min |
| **Total** | **~5 hr** |

## Risks

| Risk | Symptom | Fix |
|---|---|---|
| `set_controller_interactive` expects a tensor layout we don't match | RuntimeError or silent wrong sim state | Read its impl in [spring_mass_warp.py:852](../../../qqtt/model/diff_simulator/spring_mass_warp.py#L852). Copy interactive_playground's call pattern verbatim. Caught by G3. |
| Group assignment at runtime ≠ group assignment from extract_force_data | G3 fails: positions drift even with replay actions | Use `policy_dataset.npz`'s saved `group_ids` for the case (Step 1 saved them per case). Otherwise re-KMeans with same seed=0. |
| Feature mismatch between training and rollout | Policy outputs garbage; G4 controller flies off | Share a single `compute_31d_features` function between extract/augment scripts and the driver. |
| Wrong units in F_goal | Policy gets force values 1000× off-scale; behavior nonsensical | Force units are Newtons throughout. Recorded `y_per_ctrl` is the source of truth. Assert F_goal scale before rollout (e.g. `‖F_goal‖.max() < 1e6`). |
| Rollout speed too slow | G3+G4+G5 take an hour each | Each step is a single `wp.capture_launch`, microseconds. 150 frames should be ~1–2 s. If it's much slower, something's wrong with the loop. |
| Material mismatch when looking up target_scalers | KeyError, or wrong scaler applied | The CLI driver must map the case name to its material (use `_object_category` from `extract_dataset.py`, the canonical lookup). Don't hardcode. |
| Simulator's internal "frame index" tracking breaks under interactive control | Force formula uses wrong rest lengths | Verify that `wp.capture_launch(forward_graph)` doesn't depend on a frame counter. If it does, manually reset. |
| Per-step inference float32 vs float64 mismatches with simulator state | Tiny numerical drift; G3 borderline fail | Cast policy output to float32 before sending into the simulator. |

## Acceptance criteria

Step 3 is done when:
1. `InvPhyTrainerWarp.run_policy()` exists, is documented in
   `notes/upstream_changes.md`, and is invoked successfully by
   `run_closed_loop.py`.
2. G1, G2, G3 all PASS. (G3 is the hard gate — if it fails we DON'T
   declare Step 3 done.)
3. G4 either PASSES or fails with a documented, characterized symptom
   (e.g. "achieved force RMS error 67%, gripper drifts to wrong final
   position"). Not strict R² gate yet — that's Step 4.
4. G5 simulation completes without explosion.
5. At least one rollout npz exists in `results/closed_loop_rollouts/`
   for use by Step 4.
6. New lessons recorded in [lessons.md](lessons.md).
7. `step3_review.md` written with gate outcomes.

## Out of scope for Step 3

- F_goal profile library (replay / ramp / step / sinusoid as a polished
  set) — that's Step 4.
- Quantitative tracking metrics — that's Step 4.
- Plots and videos — that's Step 5.
- Cross-case (held-out trajectories) — same case the policy trained on
  is fine for Step 3.
- Multi-seed eval — pick one seed for Step 3, save the per-seed sweep
  for Step 4.
- Model ensembling, MPPI on top of the policy, etc. — not now.
