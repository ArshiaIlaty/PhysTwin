# Closed-loop control — task breakdown

Project overview: [explanation.md](explanation.md). Active checklist:
[todo.md](todo.md). Detailed step plans: `stepN_plan.md`.

## Step 1 — Build policy dataset
**Goal.** Convert existing `dataset_v2/*.npz` and `dataset_synth_raw/*.npz`
into a single `(state, force_now, force_goal, action)` table for policy
training. No PhysTwin involved.

**Output.** `results/policy_dataset.npz` with row-major arrays. Schema
locked in [step1_plan.md](step1_plan.md).

**Scope.** ~1 hour. Pure numpy.

**Risk.** Low. Main thing to get right: KMeans group assignment for
double-control cases.

## Step 2 — Train MLP policy
**Goal.** Train a frame-independent MLP that maps `(state, force_now,
force_goal)` → per-group Δctrl. Multi-seed (5), per-category scaler,
random-block within-case split — same protocol as v3.1.

**Output.** `results/models_policy/seed_*/policy.pt`, scalers, metrics
json.

**Scope.** ~2 hours (mostly mirrors [train_models.py](../../code/train_models.py)).

**Risk.** Low. Architecture and training loop are well-validated.

## Step 3 — Closed-loop driver
**Goal.** Add `InvPhyTrainerWarp.run_policy(policy, F_goal)` to
[trainer_warp.py](../../../qqtt/engine/trainer_warp.py) — drives PhysTwin
frame by frame, calling the policy for each next controller target instead
of reading the recorded trajectory.

**Output.** New method in `trainer_warp.py` + a CLI driver script in
`my_work/code/run_closed_loop.py`.

**Scope.** ~4–6 hours. The only step with real engineering risk.

**Risk.** Medium. Must get controller-point tensor layout right (per-point
xyz, not per-group centroid). Reference: how
[interactive_playground @ trainer_warp.py:1149](../../../qqtt/engine/trainer_warp.py#L1149)
calls `set_controller_interactive`.

## Step 4 — Evaluation target-force profiles
**Goal.** Define a small library of `F_goal` trajectories and run the
closed-loop driver for each.

**Profiles.**
- **Replay:** `F_goal` = recorded force trajectory of held-out case.
- **Ramp:** linear 0 → peak → 0 (peak chosen per-case from case force range).
- **Step:** 0 → 10 N → hold → 0.
- **Sinusoid:** slow oscillation around 0.

**Output.** `results/eval_closed_loop/{case}__{profile}.npz` with
`F_goal`, `F_achieved`, `controller_traj`, `positions`.

**Scope.** ~2–3 hours including rollout time.

**Risk.** Low–medium. Mostly depends on policy quality from Step 2.

## Step 5 — Figures + videos
**Goal.** Two plot families per (case, profile), plus rendered rollout videos.

**Static plots.**
- **Force tracking:** `||F_achieved(t)||` overlaid on `||F_goal(t)||`.
- **Gripper trajectory:** 3D path of `controller_centroid(t)` for the
  policy-driven rollout, optionally overlaid on the recorded trajectory.
- **Summary bar chart**: per-profile mean tracking error per material.
- **Per-motion-type breakdown**: tracking error grouped by the rollout's
  source motion family (using the `motion_type` field from Step 1).

**Videos.** PhysTwin already has two render paths in
[trainer_warp.py:1438](../../../qqtt/engine/trainer_warp.py#L1438) and
[trainer_warp.py:1963](../../../qqtt/engine/trainer_warp.py#L1963), both
writing mp4 via `cv2.VideoWriter` using Gaussian-splatting rendering. Two
tiers, do them in this order:

1. **Lightweight (first).** Adapt `visualize_force`'s rendering loop to
   consume the policy-rollout arrays from Step 4 instead of the recorded
   trajectory. Renders the object + controller + force arrows per frame
   with the case's original camera intrinsics. Mp4 saved to
   `results/figures/closed_loop/videos/{case}__{profile}.mp4`. Needs only
   the `gaussian_output/` already on disk for that case.
2. **Photorealistic (optional, for the demo).** Same arrays, but route
   through `gs_render_dynamics.py` for full Gaussian-splatting render.
   Heavier compute; same shape of output.

For a side-by-side demo, add a script that stitches `goal-vs-achieved
force curve | rendered video` into a single frame so the viewer sees the
target force, the realized force, and the simulator's behavior all at
once.

**Output.**
- `results/figures/closed_loop/*.png` — static plots.
- `results/figures/closed_loop/videos/*.mp4` — rollout videos.

**Scope.** ~2 hours static plots + ~3–4 hours for the lightweight video
adapter + ~2 hours optional Gaussian render = total 4–8 hours depending on
how polished the video goes.

**Risk.** Low for static plots and the lightweight video path. Medium for
the Gaussian render path (heavier dependencies, per-case gaussian assets).

## Sequencing & dependencies

```
Step 1 ──→ Step 2 ──→ Step 3 ──→ Step 4 ──→ Step 5
              ↑          ↑
              └──────────┴── Step 3 can be scaffolded in parallel with Step 2
                             (driver works with a random policy for shape testing)
```

## Out of scope for the first cut

- GRU / 1D-conv temporal models (only if MLP plateaus).
- Real robot deployment / sim-to-real validation.
- Cross-case generalization claims.
- Multi-step lookahead (`F_goal[t+1..t+k]`).
- RL training — supervised inverse-dynamics first.
