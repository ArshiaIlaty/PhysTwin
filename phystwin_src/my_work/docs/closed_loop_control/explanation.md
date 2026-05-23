# Closed-Loop Force-Targeting Control with PhysTwin

This is an extension of the force-from-deformation project documented under
[../tasks/experiment_explanation.md](../tasks/experiment_explanation.md). The
prior project trained a model that *infers* applied force from observed
deformation. This project flips the arrow: instead of inferring force after
the fact, we *command* a desired force and let a learned policy drive the
gripper to achieve it.

## One-Sentence Summary

We train a small policy network that takes (current deformation state +
desired force) and outputs the next gripper motion, then close the loop in
PhysTwin to test whether the simulator actually follows the commanded force
profile.

## The Big Idea

| Project | Direction | Role of force |
|---|---|---|
| Original (force-from-deformation) | deformation → force | output / inferred |
| **This project (closed-loop control)** | **state + force_goal → gripper action** | **input / commanded** |

The simulator already accepts arbitrary controller targets every frame via
[spring_mass_warp.py:852](../../../qqtt/model/diff_simulator/spring_mass_warp.py#L852)
(`set_controller_interactive`). The interactive playground uses this to let a
human press keys. We replace the human with a learned policy.

## Why the Training Data Is Essentially Free

Every npz file in [results/dataset_v2/](../../results/dataset_v2/) and
[results/dataset_synth_raw/](../../results/dataset_synth_raw/) already stores
per-frame `X`, `controller_pos`, and `y_per_ctrl`. For any consecutive frame
pair `(t, t+1)` we can derive a training row:

| Field | Source | Shape |
|---|---|---|
| `state` | `X[t]` | 31 |
| `force_now` | `y_per_ctrl[t]` | 2 × 3 (padded) |
| `force_goal` | `y_per_ctrl[t+1]` | 2 × 3 (padded) |
| `action` | per-group centroid Δ from `controller_pos[t→t+1]` | 2 × 3 (padded) |

That's it. 21 real cases + 218 synthetic trajectories × ~100–150 frames each ≈
30k rows. No new PhysTwin rollouts needed for training.

## The Closed-Loop Test (the Actual Evaluation)

Held-out frame prediction (R² etc.) is only the training-time signal. The
real evaluation is to plug the policy into the simulator and watch what
happens:

```text
1. Reset PhysTwin to frame 0 of a test case.
2. Pick a target force trajectory F_goal[0..T]
     (e.g. ramp 0→10N→0, or sinusoid, or the recorded force itself).
3. For t = 0..T:
     observe X_t, force_now from the simulator state
     delta     = policy(X_t, force_now, F_goal[t+1])    # per-group centroid Δ
     new_ctrl  = rigid_translate(prev_ctrl, delta)      # apply same Δ to K points in each group
     sim.set_controller_interactive(prev_ctrl, new_ctrl)
     wp.capture_launch(sim.forward_graph)               # one sim step
     record positions, achieved forces
4. Plot achieved force(t) vs F_goal(t).
```

Two flavors:

- **Replay** — `F_goal` is the recorded force trajectory of the test case.
  We know an action sequence exists that achieves this (the recorded one).
  Sanity check.
- **Novel target** — `F_goal` is something synthetic (ramp / step / sinusoid).
  Tests whether the policy generalizes the inverse mapping to a goal it
  never saw paired with this exact initial state.

## Design Decisions (locked at planning time)

1. **MLP first.** Stateless, frame-independent. Same architecture family that
   reached v3.1 R² 0.5–0.7. Velocity features in `dataset_v2` already encode
   most temporal signal. Upgrade to small GRU / 1D-conv over an 8-frame
   window only if MLP plateaus.
2. **Per-group centroid action**, padded to 2 groups (3 × 2 = 6 outputs).
   Matches how `generate_synthetic.py` constructs synthetic motions and how
   PhysTwin clusters controller points via KMeans. Per-particle is silly
   (rigid gripper); single-centroid loses information for double-hand cases.
3. **Per-group `y_per_ctrl` for force I/O**, not `y_net`. Each gripper gets
   its own current and goal force. Padding convention matches the existing
   `y_per_ctrl [T, 2, 3]` shape — single-control trajectories have a
   zero-padded second group, masked out in loss + rollout.
4. **Random-block within-case split** for training/validation, same as v3.1.
   We're not making a cross-case generalization claim here; the closed-loop
   rollout is the deployment metric.
5. **5-seed multi-seed eval**, same wrapper as v3.1, to separate real signal
   from lucky seed.
6. **Rigid-translate during rollout.** When the policy outputs a per-group
   Δ, apply the same translation to all K controller points in that group.
   This matches how the recorded trajectories move (each KMeans cluster
   moves rigidly).

## Process Overview (5 steps)

Detailed step plans live in `stepN_plan.md` files in this folder.

| Step | What | Output |
|---|---|---|
| 1 | Build policy dataset from existing npz files | `results/policy_dataset.npz` with (state, force_now, force_goal, action) rows |
| 2 | Train MLP policy (multi-seed, per-cat scaler) | `results/models_policy/` with checkpoints + metrics |
| 3 | Build closed-loop driver (`run_policy()` in `trainer_warp.py`) | Method that drives PhysTwin with a learned policy |
| 4 | Define eval target-force profiles (replay / ramp / step / sinusoid) | `eval_policy.py` running rollouts |
| 5 | Make figures + rollout videos (achieved-vs-goal force, gripper trajectory overlay, mp4 render of the policy driving the simulator) | `results/figures/closed_loop/` |

## What This Demonstrates

The deliverable narrative is:

> PhysTwin's spring-mass simulator already lets us *infer* force from
> deformation. We added a learned policy that closes the loop the other way:
> commanded force → gripper motion → realized force. This turns PhysTwin from
> a passive force estimator into a force-aware deformable-object actuator
> inside the simulator.

This pairs naturally with the prior force-prediction work and lines up with
the official README's "RL policy for rope manipulation" pitch — same
direction (controlling the simulator), simpler method (supervised inverse
dynamics, not RL).

## Known Risks

| Risk | Mitigation |
|---|---|
| Distribution shift — policy was trained on actions in the synthetic + recorded distribution; novel `F_goal` may demand actions outside that. | Start with replay test (in-distribution). Use novel-target test as the harder story. Report the operating envelope honestly. |
| Per-group action requires knowing which controller points belong to which group. Not stored in npz. | KMeans on `controller_pos[0]` with `n_ctrl_parts` clusters; group assignment is stable across frames because each cluster moves rigidly. |
| Myopic one-step lookahead may overshoot on aggressive ramps. | If observed, condition on a force window `F_goal[t+1..t+k]` instead of one step. Out of scope for first cut. |
| Closed-loop rollout state must match policy training state exactly (same `compute_features` function, same units). | Reuse `summary_features()` from [extract_dataset.py](../../code/extract_dataset.py) inside `run_policy()` — don't reimplement. |
| `single_push_sloth` has 866 kN spikes (already excluded from prior training). | Exclude from policy training too. Document in `tasks.md`. |
