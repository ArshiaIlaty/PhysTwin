# Lessons learned — closed-loop control extension

Append a dated entry every time the user corrects an approach OR confirms a
non-obvious decision worked. Lead with the rule, then **Why** and **How to
apply**.

For the upstream project's lessons see [../tasks/lessons.md](../tasks/lessons.md).
SLURM defaults and HPC environment notes from that file still apply
(`--partition=free-gpu --account=mgamalel`, `/pub/mgamalel/envs/phystwin`,
build flags for CUDA submodules, etc.) — don't re-learn them here.

---

## (template)
## YYYY-MM-DD — short rule name
- **Rule**: …
- **Why**: …
- **How to apply**: …

## 2026-05-23 — In the dataset npz files, `material` is cfg_type, not the material taxonomy
- **Rule**: When reading per-trajectory npz files from `results/dataset_v2/`
  or `results/dataset_synth_raw/`, the field that holds the rope/cloth/sloth/
  toy taxonomy is **`object_category`**, not `material`. The `material` field
  holds PhysTwin's cfg_type, which is only `"real"` or `"cloth"` (used to
  select `configs/real.yaml` vs `configs/cloth.yaml` at extraction time).
- **Why**: `extract_dataset.py` saves `material=cfg_type` and
  `object_category=_object_category(case_name)`. Same convention in
  `generate_synthetic.py`. The names are misleading. First run of
  `build_policy_dataset.py` produced `per material: {cloth, real}` which is
  meaningless for a per-material scaler / per-material training.
- **How to apply**: New scripts should read `object_category` (with a
  fallback to `material`). Audit existing scripts before re-running — the
  v3.1 trainer reportedly does the right thing, but anything new should
  not assume `material` is what its name suggests.

## 2026-05-23 — Verify input dim arithmetic literally before coding
- **Rule**: Always compute the actual sum of feature dims and assert it
  matches the planned in_dim. Don't trust prose ("31 + 6 + 6 = 37").
- **Why**: First draft of step2_plan.md said `Input dim = 37` for
  `31 + 6 + 6`. The model code would have failed loudly at first forward
  pass, but the plan was wrong on paper for a while. Caught before
  implementation, but a tighter loop is to write `assert features.shape[1]
  == 43` next to the concat.
- **How to apply**: After every `np.concatenate` or `nn.Linear` dim choice,
  add an `assert` with the explicit expected number. Make the wrong number
  loud, not silent.

## 2026-05-23 — load_dataset() must expose every field downstream code uses
- **Rule**: When the source npz has N fields but load_dataset() returns
  only K of them, dependent code (inspect, eval, future steps) will
  KeyError at the worst possible moment. Mirror every npz field into the
  returned dict, even if "we don't need it for training."
- **Why**: G4 inspect crashed with `KeyError: 'n_ctrl_parts'`. I had
  filtered it out of `load_dataset` because the model didn't use it for
  forward pass, but the inspect code did need it to print per-row
  n_ctrl_parts. Adding it back was trivial; the lesson is the cost of
  finding out vs the cost of including the field.
- **How to apply**: In dataset loaders, default to including everything
  from the source. Explicitly drop only what's clearly redundant (e.g. a
  derived column already reconstructible from kept fields).

## 2026-05-24 — PPO + BC warm-start needs critic warmup or it destroys the policy
- **Rule**: When initializing PPO from a good behavior-cloning policy
  with a random-initialized critic, freeze the actor for the first 3-5
  PPO updates so the critic can learn sensible value estimates. Without
  this, the random critic produces noisy advantage estimates and the
  policy regresses dramatically in the early updates.
- **Why**: First Fix-RL smoke test (G3) used standard PPO defaults
  (lr_actor=3e-4, no critic warmup). The policy regressed from return
  -18 (BC warm-start) → -87 in 4 PPO updates, with KL divergence going
  to infinity. The BC actor was good; the critic was random; the
  computed advantages were garbage; the policy moved aggressively in
  the wrong direction.
- **How to apply**: For PPO with BC warm-start:
    1. Use much lower actor LR (3e-5 vs typical 3e-4). The actor is
       already near-optimal; small updates only.
    2. Warm up the critic alone for ~5 updates with actor frozen
       (`actor.parameters().requires_grad = False`). Re-enable actor
       updates after.
    3. Use `target_kl` early-stopping (e.g., 0.02). Stop the PPO update
       loop when KL exceeds 1.5× target.
    4. Clip the log-ratio before `exp()` (e.g., to ±10) to prevent
       numerical overflow in pathological updates.
    5. Lower entropy coefficient (0.005 vs typical 0.01) to discourage
       the policy from drifting away from BC noise.

## 2026-05-23 — Three goal-side fixes all failed the same way → release is state-distribution shift
- **Rule**: For a one-step inverse-dynamics policy that overshoots and
  can't release in closed loop, do NOT try goal-side fixes
  (inference-time goal shaping, training-time goal augmentation, or
  hierarchical sub-goal planners). All three will fail because they
  don't address the actual root cause: state-distribution shift under
  closed-loop rollout. The principled fixes are MPC (use the simulator
  directly) or RL fine-tuning (handles state shift naturally).
- **Why**: Documented systematically through three failed attempts:
    - Fix A (clip goal to ±500 N/frame): ramp release frac stayed at
      0.04; cloth simulation blew up to NaN.
    - Fix B (hindsight goal augmentation): ramp release 0.07 (barely
      changed); cloth replay regressed 0.22 → 1.01.
    - Fix C (hierarchical Model B + Model A): ramp release 0.02
      (WORSE); cloth replay catastrophically regressed 0.22 → 99.4
      with 2.4 m gripper drift.
  The shared failure mode: even when goal interpretation IS improved
  (Fix C had Model B at 80%+ R² on cloth sub-goal prediction),
  rollouts compound small action errors into states the model never
  saw at training, causing catastrophic drift.
- **How to apply**:
    1. If a behavior-cloned controller fails on novel targets, treat
       goal-side fixes as cheap experiments to RULE OUT
       goal-interpretation as the problem — not as expected solutions.
    2. Each goal-side fix has its own characteristic failure
       (averaging, hierarchical drift, etc.), but they all share the
       inability to handle out-of-distribution states.
    3. Skip to MPC or RL once goal-side has been verified to fail
       once or twice. Don't keep iterating goal-side variants.
    4. The "three failures, same root cause" framing is itself a
       valuable result for any writeup — strong negative evidence
       that the issue is state shift.

## 2026-05-23 — Hindsight goal augmentation lowers val loss but BREAKS closed-loop control
- **Rule**: For one-step inverse-dynamics policies, augmenting the
  training set with "future-frame goals" while keeping the action
  label as the recorded next-frame action will lower validation MSE
  but make closed-loop deployment substantially worse. Don't do it
  unless you have a way to relabel the actions too.
- **Why**: The augmented dataset puts multiple different
  `force_goal` values against the same `(state, force_now, action)`
  tuple — because all of them share the same recorded action. The
  MSE-minimizing prediction is the "average action" for the state,
  ignoring the goal entirely. So:
    - Val loss looks better (0.41 → 0.33 at 20 epochs) because the
      model is doing the easier "predict the mean" task.
    - Closed-loop tracking gets worse because the policy now ignores
      the user's goal trajectory. `double_lift_cloth_1` replay
      err_ratio went 0.22 → 1.01 (4.6× regression).
    - The release problem on novel ramps stays broken because the
      training-data action labels don't actually contain a clean
      "retract" signal — only 17% of rope release rows have
      retract-direction actions.
- **How to apply**:
    1. If you want to add goal-distribution diversity, you ALSO need
       to relabel the actions appropriately (e.g. via the simulator,
       inverse dynamics queries, or expert demonstrations under the
       new goal). Augmenting only the goal is silently destructive.
    2. **Never trust val loss as the sole metric for control
       policies.** Always include closed-loop rollout evaluation in
       the gate set — that's what catches the val-loss/deployment
       mismatch.
    3. The release problem in our project is state-distribution shift,
       not goal-distribution shift. Confirmed by Fix B failure. The
       next attempt should be MPC over the simulator (no learned
       policy → no distribution shift issue).

## 2026-05-23 — Inference-time goal shaping does NOT rescue a behavior-cloned policy
- **Rule**: When a one-step inverse-dynamics policy can't track a
  novel goal trajectory (e.g. fails to release after a rise), feeding
  it incremental sub-goals via an inference-time wrapper is unlikely
  to help and can actively destabilize the simulator.
- **Why**: We hypothesized the policy was failing because the user's
  goal was outside its trained distribution. Built a "goal pacer" that
  clipped per-frame goal change to ±500 N or ±1000 N. Result on the
  ramp test:
    - rope: equal or worse (end-force ROSE from 8.7 kN to 12.9 kN)
    - cloth: simulation blew up to 129 kN then NaN
    - sloth: NaN everywhere
  Root cause is deeper than goal scaling: (a) cloth/sloth dynamics
  are too compliant — even "small step" requests over many frames
  produce compounding gripper motion that the spring-mass sim can't
  absorb stably; (b) for rope, only 17% of release-row training
  labels actually retract (force magnitude is a noisy proxy for
  action direction in rope geometry); (c) this is the classic
  behavior-cloning distribution-shift problem.
- **How to apply**:
    1. Don't reach for inference-time goal shaping as a first fix for
       compounding-error failures. It's almost free to try (we did)
       but the failure modes are predictable from the underlying
       dynamics.
    2. If "policy can't release" or "policy overshoots" is the issue,
       the real fixes are at the model/data level: DAgger,
       multi-step lookahead, or replacing one-step IDM with
       MPC-over-the-simulator. Each is multi-day, not a 1-hour patch.
    3. ALWAYS check training-label consistency for the failure mode
       before assuming the architecture is at fault. In our case,
       the rope training labels themselves don't have a clean
       release-direction signal — no architecture could learn what
       isn't in the data.
    4. The `--goal_shaping` flag is retained in
       `run_closed_loop.py` for future re-attempts; default is
       `direct` (the wrapper is disabled).

## 2026-05-23 — Recorded controller motion is NOT rigid per group
- **Rule**: The recorded `controller_pos[T, K, 3]` in `dataset_v2/` and
  `dataset_synth_raw/` does NOT move as a rigid body per group. Per-point
  intra-group deviation in `single_push_rope_1` is median 1.4 mm/frame,
  max 9.2 mm/frame, with centroid Δ averaging only 1.6 mm/frame. So the
  per-point variation is comparable to the centroid motion. Any policy
  or driver that outputs per-group centroid Δ + rigid translates will
  diverge from a full-fidelity replay by ~3 mm/controller-point median,
  ~6 mm max, accumulating over a full trajectory.
- **Why**: PhysTwin's data processing pipeline tracks individual control
  particles from RGB-D video; it doesn't enforce a rigid-body constraint
  on the gripper. The K controller points can drift semi-independently
  (real hand deformation + tracking noise). Step 3 G3 caught this:
  rigid-replay produced 8 mm max particle error vs `extract_force_data`,
  not the 1 mm I had naively assumed.
- **How to apply**:
    1. Any "rigid-translate at the centroid Δ" contract (the policy
       action space and the closed-loop driver both use this) has an
       irreducible ~3-6 mm error vs full-fidelity replay. Don't set
       acceptance criteria below this floor unless you also refit the
       policy to output per-point actions.
    2. When comparing closed-loop rollout against recorded ground truth
       (Step 4/5 figures), report the rigid-replay deviation as a
       baseline — that's the floor the policy is judged against, not
       0.0 mm.
    3. If you ever want to reproduce the EXACT recorded simulation,
       use `extract_force_data` (full per-particle ctrl trajectory),
       not `run_policy` with replay_action profile.

## 2026-05-23 — Two `qqtt/` trees exist; new scripts MUST pin the phystwin_src copy
- **Rule**: This repo has two `qqtt/` packages:
  1. `/share/.../PhysTwin/qqtt/` (top-level)
  2. `/share/.../PhysTwin/phystwin_src/qqtt/` (where my upstream edits live —
     `extract_force_data`, `run_policy`, the lazy pynput import, etc.)
  Any new script that does `from qqtt import ...` from
  `my_work/code/<script>.py` MUST prepend `phystwin_src` to `sys.path`
  before the import, and SHOULD assert that
  `qqtt.__file__` starts with the phystwin_src path AND that the new
  method (e.g. `InvPhyTrainerWarp.run_policy`) is present. Without that,
  Python may resolve to the top-level copy and silently use a stale qqtt.
- **Why**: Step 3's first slurm run failed with
  `ModuleNotFoundError: No module named 'qqtt'` because the script was
  launched with `python my_work/code/run_closed_loop.py` from
  `--chdir=phystwin_src`. Python's default sys.path[0] is the SCRIPT's
  directory (`phystwin_src/my_work/code/`), not cwd, so neither qqtt
  copy was found. Adding `phystwin_src` to sys.path fixed it AND
  guaranteed we use the edited copy.
- **How to apply**: At the top of any new my_work/code script that needs
  qqtt:
  ```python
  SCRIPT_DIR = Path(__file__).resolve().parent
  REPO_ROOT = SCRIPT_DIR.parent.parent   # = phystwin_src
  sys.path.insert(0, str(REPO_ROOT))
  from qqtt import InvPhyTrainerWarp
  import qqtt as _q
  assert str(REPO_ROOT) in _q.__file__, f"wrong qqtt: {_q.__file__}"
  ```
  See `run_closed_loop.py` for the canonical pattern.

## 2026-05-23 — Masked outputs are unconstrained — never trust them at inference
- **Rule**: When a model has output dims that are masked out of the loss
  for some rows (e.g. group-2 predictions for single-ctrl rows), those
  outputs receive no gradient and will produce arbitrary values at
  inference. They are NOT learned to be zero.
- **Why**: G4 inspection showed single-ctrl rows had mean ‖pred_g1‖ =
  1.126 mm, despite the plan asserting it would be "near zero." The
  network has no incentive to output zero on those dims because the loss
  doesn't penalize them.
- **How to apply**:
    1. In any closed-loop or downstream consumer of the policy, ALWAYS
       multiply predictions by the action_mask (or n_ctrl_parts derived
       mask) before applying them. This is the contract.
    2. If "near zero on masked dims" is a hard requirement, either (a)
       add an L2 penalty on masked outputs as part of the loss, or (b)
       pass action_mask as an input feature so the model can learn the
       mapping. Don't expect it to emerge.
    3. The "near zero" check in G4 was overspecified in the plan; what
       actually matters is that the closed-loop driver masks the output.
       Step 3 must enforce this.
