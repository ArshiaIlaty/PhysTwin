# PhysTwin Experiment Explanation

This note explains the current force-from-deformation experiment summarized in
`tasks/review.md`. It focuses on the practical questions: what the pipeline is
doing, what R² means, whether the model predicts `y_net` or `y_per_ctrl`, and
what the extracted features mean in real-world terms.

## One-Sentence Summary

The experiment uses PhysTwin to generate supervision: for each video timestep,
it records how the object's simulated particles deform and the force applied at
the control point(s), then trains a small model to predict the net applied force
from compact deformation features.

## Current Pipeline

1. PhysTwin has already optimized physics parameters for each case.
   These live under `phystwin_src/experiments_optimization/<case>/optimal_params.pkl`.

2. `extract_force_data()` was added to
   `phystwin_src/qqtt/engine/trainer_warp.py`.
   It runs the PhysTwin simulator without rendering images. For every frame it
   returns:
   - `positions`: object particle positions, shape `[T, N_particles, 3]`
   - `forces`: force at each control group, shape `[T, n_ctrl_parts, 3]`
   - `controller_pos`: tracked controller positions, shape `[T, N_ctrl, 3]`

3. `extract_dataset.py` converts those raw simulator outputs into ML examples.
   Each timestep becomes one row:
   - input `X`: compact summary features describing deformation
   - target `y_net`: one 3D force vector
   - target `y_per_ctrl`: up to two 3D force vectors, one per control group

4. `augment_dataset.py` creates `dataset_v2/`.
   It keeps the original 18 deformation features and adds 13 controller and
   velocity features, so the current main feature vector is 31-dimensional.

5. `train_models.py` trains three model families:
   - Ridge regression: simple linear baseline
   - Unified MLP: one neural net across all object categories
   - Per-category MLP: separate neural net for rope, cloth, sloth, and toy

6. The best current setting in `tasks/review.md` is v3.1:
   - data: `dataset_v2`
   - target: `net`
   - model: per-category MLP
   - scaler: per-category target scaler
   - `single_push_sloth` excluded from training because of unstable force spikes
   - `weird_package` excluded because it is a singleton category

## What Ridge Regression Means

Ridge regression is the simple baseline model in this experiment.

Plain linear regression learns a weighted sum of the input features:

```text
predicted_force = w1 * feature1 + w2 * feature2 + ... + bias
```

Ridge regression does the same thing, but adds a penalty that discourages huge
weights. That penalty is called L2 regularization. In plain terms, Ridge says:

> "Fit the data, but keep the formula smooth and don't let any one feature get
> an absurdly large coefficient."

Why use it here:

- It is a sanity-check baseline.
- It tells us how much force information can be recovered with only a linear
  relationship between features and force.
- If Ridge works well, the problem is mostly linear/simple.
- If Ridge fails but the MLP works, the deformation-to-force mapping likely
  needs nonlinear relationships.

In the current results, Ridge is usually bad, often with negative R². That means
the force cannot be recovered well by a simple weighted sum of the 31 features.
The MLP does better because it can learn nonlinear combinations, for example:
"when the controller is near this part of the object and the bbox stretches this
way while particles move quickly, force tends to increase."

## Are We Predicting `y_net` or `y_per_ctrl`?

The reported results in `tasks/review.md` are predicting `y_net`.

The dataset stores both targets:

```text
y_per_ctrl: [T, 2, 3]
y_net:      [T, 3]
```

`y_per_ctrl` means "force per control group." For example:
- single-hand push/lift: one real control group plus one zero-padded slot
- double-hand lift/stretch: two real control groups

It is padded to `[T, 2, 3]` so every case has the same shape.

`y_net` is computed as:

```python
y_net = forces.sum(axis=1)
```

So if a frame has two hands/controllers, their 3D force vectors are summed into
one total applied force vector. If a frame has one controller, `y_net` is just
that controller's force.

Important: older notes say "net wrench", but the code target here is a 3D force
vector `[Fx, Fy, Fz]`, not a 6D force-plus-torque wrench.

## What R² Means

R² is a score for "how much of the target variation did the model explain?"

The formula is:

```text
R² = 1 - model_error / baseline_error
```

The baseline is a very dumb predictor: always predict the average force in the
test set. So R² compares the model against "just guess the mean force every
time."

How to read it:

| R² value | Meaning |
|---|---|
| 1.00 | Perfect prediction |
| 0.70 | Model explains about 70% of the force variation |
| 0.50 | Model explains about half the force variation |
| 0.00 | No better than predicting the average test force |
| Negative | Worse than predicting the average test force |

So when the review says rope R² = 0.69, that means the model captures about 69%
of the frame-to-frame force variation for that evaluation setup. When cross-case
rope R² is negative, it means the model does not generalize to held-out rope
cases yet; it would have been better to predict the average held-out force.

Negative R² is not a software error by itself. It is a warning that the model is
missing the test distribution. In this project that mainly happens when training
on a few cases cannot cover the force scale of a different held-out case.

## Current Results in Plain English

The best within-trajectory result is:

| Split / model | Rope | Cloth | Sloth | Toy |
|---|---:|---:|---:|---:|
| random-block, per-category MLP, per-category scaler | 0.69 ± 0.07 | 0.51 ± 0.11 | 0.50 ± 0.14 | 0.02 ± 0.02 |

This means the model can learn a useful force-from-deformation mapping when it
sees part of the same trajectory during training and predicts another held-out
block of that trajectory.

The best cross-case result is:

| Split / model | Rope | Cloth | Sloth | Toy |
|---|---:|---:|---:|---:|
| cross-case, per-category MLP, per-category scaler | -0.90 ± 0.23 | 0.50 ± 0.03 | -0.50 ± 0.47 | -83 ± 42 |

This means cloth generalizes best to unseen trajectories. Rope and sloth do not
yet generalize reliably across cases. Toy is essentially not solved because the
"toy" category mixes very different objects, mostly zebra cases plus one
dinosaur case, with too little data.

## Why Per-Category Scaling Helped

Forces differ a lot by object category and by case. A shared target scaler made
one category's force scale affect the numeric scale of another category. That
was especially bad after adding toy cases.

The per-category scaler fixes that by standardizing targets separately for rope,
cloth, sloth, and toy. In the review, rope became much more stable:

```text
within-case pooled scaler:   rope R² = 0.36 ± 0.19
within-case per-cat scaler:  rope R² = 0.65 ± 0.02
```

That is a real improvement: not just a higher average, but much lower seed-to-seed
variance.

## What Features Were Extracted?

Each frame starts with many particles: a rope may have around 1k-2k particles,
cloth around 5k-8k particles, and sloth/toy several thousand particles. Raw
particle coordinates cannot be concatenated directly because every case has a
different particle count. The solution is to summarize the particle cloud with
fixed-length features.

All deformation features are measured relative to frame 0, the rest/reference
frame.

### Original 18 Deformation Features

| Feature group | Dimensions | Real-world meaning |
|---|---:|---|
| `centroid_disp_x/y/z` | 3 | How far the object's average position moved from rest. Think: overall translation of the object. |
| `bbox_stretch_x/y/z` | 3 | How the object's bounding box changed size along each axis. Think: did it stretch, compress, or flatten? |
| `max_abs_x/y/z` | 3 | Largest absolute particle displacement along each axis. Think: the most extreme point motion in x, y, z. |
| `mean_abs_x/y/z` | 3 | Average absolute particle displacement along each axis. Think: typical amount of motion in each direction. |
| `std_disp_x/y/z` | 3 | Spread of particle displacement along each axis. Think: whether the object deforms unevenly instead of moving rigidly. |
| `ke_proxy` | 1 | Mean squared displacement. It is not true kinetic energy because it uses displacement, not mass and velocity, but it acts like an "overall deformation energy" signal. |
| `mean_disp_mag` | 1 | Average 3D displacement magnitude across particles. Think: average how-far-did-particles-move. |
| `max_disp_mag` | 1 | Largest 3D displacement magnitude. Think: most displaced point on the object. |

These 18 features answer: "What shape change did the object undergo?"

### Added 13 v2 Features

`augment_dataset.py` adds controller and velocity features, creating the current
31-feature `dataset_v2`.

| Feature group | Dimensions | Real-world meaning |
|---|---:|---|
| `ctrl_centroid_disp_x/y/z` | 3 | How far the controller/hand centroid moved from its starting position. |
| `nearest_dist` | 1 | Distance from controller centroid to the closest object particle. Think: rough contact/proximity cue. |
| `mean_contact_dist` | 1 | Average distance from controller centroid to all object particles. Think: where the controller is relative to the object as a whole. |
| `rel_motion_x/y/z` | 3 | Controller centroid minus object centroid. Think: where the hand/controller is relative to the object's center. |
| `mean_vel_mag` | 1 | Average per-frame particle speed magnitude. Computed from frame-to-frame particle motion. |
| `max_vel_mag` | 1 | Fastest particle's per-frame speed magnitude. |
| `centroid_vel_x/y/z` | 3 | Velocity of the object's average position. Think: overall object motion direction and speed. |

These 13 features answer: "Where is the hand/controller, and how fast is the
object moving?"

That controller information is important. The review notes that v2 unlocked
cross-case cloth generalization because the model finally had contact-location
information, not only deformation shape.

## How We Are Making More Trajectories

The file `phystwin_src/generate_synthetic.py` creates additional synthetic
training trajectories. The motivation is simple: cross-case generalization is
weak because the dataset has too few force-diverse examples per material. Rather
than hand-recording many new videos, we reuse PhysTwin's already-calibrated
simulators and drive them with new controller motions.

This is not creating arbitrary fake labels with a separate formula. It is still
using PhysTwin's spring-mass simulator and the same force extraction path.

### Core Idea

For a stable real case, called a donor case:

1. Load that donor's calibrated PhysTwin simulator.
2. Keep its optimized material/contact parameters.
3. Replace the recorded controller trajectory with a new synthetic controller
   trajectory.
4. Run the simulator forward with `extract_force_data()`.
5. Save the resulting particle positions, controller positions, and forces as a
   new `.npz` dataset case.

So if `rope_double_hand` is the donor, the script can create files like:

```text
rope_double_hand__synth_linear_push_0.npz
rope_double_hand__synth_sinusoidal_3.npz
rope_double_hand__synth_hold_release_5.npz
```

Each saved file has the same schema as `dataset_v2`:

```text
X:              [T, 31]
y_net:          [T, 3]
y_per_ctrl:     [T, 2, 3]
positions:      [T, N_particles, 3]
controller_pos: [T, K, 3]
feature_names:  31 feature names
source_donor:   original real case
motion_type:    synthetic controller motion type
```

### Donor Cases

The script uses multiple donor cases per material, but it intentionally excludes
cases with obvious numerical force blow-ups.

| Material | Donors used | Excluded examples |
|---|---|---|
| rope | `rope_double_hand`, `single_push_rope_1`, `single_push_rope_4`, `single_lift_rope`, `single_push_rope` | none in the donor list |
| cloth | `single_lift_cloth`, `single_lift_cloth_4`, `single_lift_cloth_3`, `double_lift_cloth_3`, `double_lift_cloth_1`, `single_lift_cloth_1`, `single_clift_cloth_3`, `single_clift_cloth_1` | none in the donor list |
| sloth | `double_stretch_sloth`, `single_lift_sloth`, `double_lift_sloth` | `single_push_sloth`, because it has an 866 kN spring blow-up |
| toy | `double_stretch_zebra`, `single_lift_zebra` | `double_lift_zebra` and `single_lift_dinosor`, because their recorded fits have huge numerical force issues |

### Synthetic Motion Types

The script creates four controller-motion families:

| Motion type | Real-world meaning |
|---|---|
| `linear_push` | The controller moves from rest to a random offset, then returns to rest. |
| `sinusoidal` | The controller oscillates around rest along a random direction. |
| `random_walk` | The controller takes small random steps with a soft pull back toward rest. |
| `hold_release` | The controller pushes to an offset, holds there, then releases back to rest. |

All motions start from the donor's original frame-0 controller pose. That is
important because PhysTwin's spring rest lengths are calibrated around that
initial pose.

The motion amplitude is scaled by the donor's original controller motion extent.
By default, synthetic offsets are sampled between 10% and 40% of that donor
extent. This keeps the new motions near the regime PhysTwin was calibrated for.

### Sanity Gates

Every generated trajectory is checked before saving. Rejected trajectories are
not written to disk.

The checks are:

| Gate | Purpose |
|---|---|
| No NaN/Inf in positions or forces | Reject numerical failures. |
| `max(|F|) <= 1.5 * donor_recorded_max_force` | Reject force spikes far outside the donor's calibrated regime. |
| Particle excursion within `2.0 * donor_bbox_diag` | Reject simulations where particles fly away. |

These gates are important because synthetic motions can push the simulator
outside what PhysTwin calibrated from the original video. The goal is more
trajectory diversity, not unphysical explosions.

### How Features and Targets Are Built

For every accepted synthetic trajectory, the script rebuilds the same 31
features used by `dataset_v2`:

1. It calls `summary_features(pos)` to compute the original 18 deformation
   features.
2. It computes the 13 controller/velocity features inline.
3. It concatenates them into `X_full`, shape `[T, 31]`.
4. It saves:

```python
y_per_ctrl = pad_forces(f, max_ctrl_parts=2)
y_net = f.sum(axis=1)
```

So synthetic data uses the same target convention as the main experiment:
headline training still predicts `y_net`, while `y_per_ctrl` is kept available.

### Current Observed Synthetic Artifacts

At inspection time, `phystwin_src/dataset_synth_raw/` contains 218 accepted
synthetic `.npz` trajectories. The saved files are currently from rope, cloth,
and sloth donor cases. I did not find `_synth_summary.json` in the current tree,
even though the script is written to produce it at the end of a completed run.

Observed accepted motion counts:

| Motion type | Count |
|---|---:|
| `linear_push` | 73 |
| `hold_release` | 72 |
| `sinusoidal` | 54 |
| `random_walk` | 19 |

The lower `random_walk` count makes sense: random walks are more likely to push
the simulator into unstable or high-force regions, so more of them get rejected
by the sanity gates.

### Why This Matters

The current real dataset has too few cases to cover the force range within each
material. That is why within-trajectory R² can be good while cross-case R² is
still negative for rope and sloth.

Synthetic trajectories are meant to increase coverage:

- more controller paths per material
- more force magnitudes
- more contact geometries
- more examples without needing new videos

The caveat is that these are still PhysTwin-generated labels, not real robot
force-sensor labels. They are useful for improving the learned PhysTwin
surrogate, but they do not replace real measured force data for final validation.

## How the Force Labels Are Computed

The force labels are not measured by a real force sensor. They are extracted
from PhysTwin's optimized spring-mass simulation.

In `extract_force_data()`, PhysTwin groups controller points into one or two
control groups. For each group, it finds the control springs that connect the
controller to object particles. For each frame, it computes spring force from:

```text
spring stiffness * spring stretch/compression direction
```

Then it sums those spring forces for the control group. That gives a force
vector `[Fx, Fy, Fz]` per control group per frame.

So the target is best understood as:

> "According to PhysTwin's optimized simulator, what external force did the
> controller apply to the object at this frame?"

## What `extract_force_data()` Actually Does

`extract_force_data()` is a thin hook into PhysTwin. It does not invent new
physics. It runs PhysTwin's already-fit simulator frame by frame and exposes
the particle positions and control-point forces that were already being computed
inside visualization code.

For one case, it does this:

### Step 1: Load PhysTwin's Converged Calibration

It loads the best trained PhysTwin checkpoint:

```python
checkpoint = torch.load("experiments/<case>/train/best_*.pth")
spring_Y = checkpoint["spring_Y"]
collide_elas = checkpoint["collide_elas"]
collide_fric = checkpoint["collide_fric"]
collide_object_elas = checkpoint["collide_object_elas"]
collide_object_fric = checkpoint["collide_object_fric"]
num_object_springs = checkpoint["num_object_springs"]
```

These values are PhysTwin's optimized physics parameters:

| Parameter | Meaning |
|---|---|
| `spring_Y` | Per-spring stiffness after optimization. |
| `collide_elas` | Ground/contact elasticity. |
| `collide_fric` | Ground/contact friction. |
| `collide_object_elas` | Object-object contact elasticity. |
| `collide_object_fric` | Object-object contact friction. |
| `num_object_springs` | Number of internal object springs before the controller springs begin. |

These are outputs of PhysTwin's optimization, not learned by our MLP.

### Step 2: Push Those Parameters Into the Simulator

The function applies the optimized parameters to the active simulator:

```python
self.simulator.set_spring_Y(torch.log(spring_Y))
self.simulator.set_collide(collide_elas, collide_fric)
self.simulator.set_collide_object(collide_object_elas, collide_object_fric)
```

PhysTwin stores spring stiffness internally in log form, so `spring_Y` is passed
through `torch.log(...)`.

### Step 3: Group Controller Springs by Control Part

Each case may have one or two control parts:

- one control part: single-hand push/lift
- two control parts: double-hand lift/stretch

For one control part, all controller points are treated as one group. For two
control parts, the function clusters controller points into two groups with
KMeans. Then, for each group, it finds the springs that connect that controller
group to object particles:

```text
force_springs[i]       = controller-object spring pairs for group i
force_rest_lengths[i]  = rest length of each spring
force_spring_Y[i]      = stiffness of each spring
```

This is bookkeeping around PhysTwin's spring data structure. It mirrors what
`visualize_force` needs in order to draw force arrows.

### Step 4: Forward-Simulate Frame by Frame

The function resets the simulator to the initial state, then loops through the
video frames:

```python
for i in range(1, frame_len):
    self.simulator.set_controller_target(i, pure_inference=True)
    wp.capture_launch(self.simulator.forward_graph)
    x = wp.to_torch(self.simulator.wp_states[-1].wp_x)
    positions[i] = x.cpu().numpy()
    forces[i] = _ctrl_forces(x, i)
```

Real-world meaning:

- `set_controller_target(i, pure_inference=True)` tells PhysTwin where the
  gripper/controller is at frame `i`.
- `wp.capture_launch(...)` runs one step of the Warp GPU spring-mass simulator.
- `x` is the object particle cloud at that frame, shape `[N_particles, 3]`.
- `positions[i]` stores the object state.
- `forces[i]` stores the control force computed from the controller springs.

### Step 5: Compute Net Force at Each Control Group

For each control group, `extract_force_data()` calls PhysTwin's
`get_force_vector(...)`.

The core calculation is Hooke's law over all controller-object springs:

```python
x1 = controller_points[springs[:, 0]]
x2 = x[springs[:, 1]]
dis = x2 - x1
dis_len = torch.norm(dis, dim=1)
direction = dis / dis_len
spring_forces = spring_Y * (dis_len / rest_lengths - 1.0) * direction
total_force = -spring_forces.sum(dim=0)
```

In real-world terms:

- each controller spring has a controller-side endpoint and an object-side
  endpoint
- if the spring is stretched or compressed away from its rest length, it exerts
  force
- the force is proportional to stiffness times strain
- all spring forces for that control group are summed into one 3D vector
- the sign is flipped so the reported vector is the force applied by the
  controller to the object

So for each frame and each control group, we get:

```text
[Fx, Fy, Fz]
```

### Step 6: Return Arrays for Dataset Creation

The function returns:

```text
positions:      [T, N_particles, 3]
forces:         [T, n_ctrl_parts, 3]
controller_pos: [T, N_ctrl, 3]
meta:           frame count, particle count, number of control parts
```

Then `extract_dataset.py` converts those into ML-ready fields:

```python
X = summary_features(positions)
y_per_ctrl = pad_forces(forces, max_ctrl_parts=2)
y_net = forces.sum(axis=1)
```

In short: `extract_force_data()` drives PhysTwin's forward simulator from
outside, using PhysTwin's already-fit physics, and reads off `(positions,
forces)` at every frame. The new code is mainly the loop and array packaging.
The spring topology, optimized physics, GPU simulation, and force formula are
all PhysTwin's.

## Most Important Takeaways

The experiment does work as a within-trajectory force estimator: for rope,
cloth, and sloth, the per-category MLP explains about 50-70% of force variation
from compact deformation/controller features.

The strongest generalization result is cloth cross-case R² around 0.50. That is
the cleanest evidence that the learned mapping can transfer to a trajectory the
model did not train on.

Rope and sloth cross-case results are still negative. The likely cause is not
the model architecture alone; the dataset is very small and force magnitudes
vary widely between cases. `single_push_sloth` also has unrealistic force spikes,
so it was excluded from the v3.1 training runs.

Toy is not solved. The category is too heterogeneous and too small.

For presenting this work, the clean framing is:

> PhysTwin provides slow but rich force supervision. A lightweight MLP can learn
> to approximate net applied force from per-frame deformation summaries. It
> works well within trajectories and transfers for cloth, but robust cross-case
> generalization needs more and cleaner force-diverse data.
