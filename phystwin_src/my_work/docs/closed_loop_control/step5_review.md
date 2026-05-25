# Step 5 review — Preliminary figures

Plan: [step5_plan.md](step5_plan.md). Script:
[../../code/make_closed_loop_figures.py](../../code/make_closed_loop_figures.py).
Output: [../../results/figures/closed_loop/](../../results/figures/closed_loop/).

## Status

✅ Done — preliminary static figure set generated. Video deliverables
**deferred** to follow-up (see Carry-forward).

## Artifacts

```
results/figures/closed_loop/
  01_per_material_bars.png         54 KB   headline bar chart
  02_force_tracking_grid.png      347 KB   9 controlled replay cases, 3×3 grid
  03_limitations_panel.png        190 KB   3 failure modes
  04_highlight_cloth.png          148 KB   double_lift_cloth_1 per-axis
  05_ramp_failure.png             155 KB   3 ramp comparisons
```

All rendered on the login node in ~5 seconds total. No GPU needed.

## What each figure shows (and what to say about it in the demo)

### 01 — Per-material bars
Bar chart of mean force-error ratio per (material × profile). Red dotted
line at ratio = 1 marks "control breakdown" (mean error = mean goal).
Annotations above each bar show how many cases were controlled vs failed.

**Demo line**: "Closed-loop control works in *replay* mode on every
material — cloth best with 4/4 controlled cases, rope and sloth have one
outlier each. Synthetic *ramp* targets blow up on cloth and sloth, as
expected from the one-step inverse-dynamics architecture."

### 02 — Force tracking grid (the meat of the result)
9 controlled rollouts arranged 3×3. Per panel: black-dashed goal force
magnitude vs solid-colored achieved force magnitude over the trajectory.

**Demo line**: "Achieved force tracks the recorded goal across all
working cases. Magnitude profiles match closely — small per-step noise,
but the trajectory envelope is preserved." Strongest individual panels
to point at: `double_lift_cloth_1` (err 0.22) and `single_lift_sloth`
(err 0.29).

### 03 — Limitations panel
3 representative failures: rope outlier (single_push_rope_4, gentle
case where the policy over-applies), sloth long-trajectory drift
(double_stretch_sloth, 192 frames, errors compound), cloth ramp blow-up
(novel target → unstable sim).

**Demo line**: "Here's what failure looks like, by failure mode. (1) On
very low-force cases the policy is mis-calibrated. (2) On long
trajectories errors compound. (3) On novel targets the simulator
escapes the trained regime."

### 04 — Highlight (cloth best)
`double_lift_cloth_1` per-axis decomposition. Four panels: Fx, Fy, Fz,
‖F‖. Shows the achieved force tracks the goal across all axes, not
just by accident in magnitude.

**Demo line**: "Picking the best case — cloth, two-hand lift. The
achieved force tracks the recorded goal across all three Cartesian axes
plus magnitude. Initial 5-frame transient settles fast." Worth showing
in a video later (best visual candidate).

### 05 — Ramp comparison
3 panels side-by-side. Rope ramp shows the policy rising to ~half the
peak and not releasing (the "one-directional" finding from Step 3 G5).
Cloth + sloth ramps show the achieved force diverging upward and never
following the goal-shape — the simulator gets pushed outside its
trained regime.

**Demo line**: "Asymmetric capability. Rising-edge tracking works;
release-to-zero doesn't. The fix is multi-step planning (MPC over the
simulator), not goal scheduling — we tried that already and recorded
the result."

## Verification gates

| Gate | Result |
|---|---|
| G1 script runs without error | PASS — all 5 PNGs written |
| G2 visual sanity | PASS — values match summary.json; axes/labels/titles correct; cloth bar lowest as expected; ramp bars tower; tracking curves show goal-dashed/achieved-solid as designed |

## Addendum 2026-05-23 — Video adapter done

[../../code/render_rollout_video.py](../../code/render_rollout_video.py)
generates side-by-side mp4 from any closed-loop rollout npz. Layout:
3D particle cloud + per-group force arrows on the left; goal-vs-achieved
force-magnitude curve with a moving "now" cursor on the right. Legend
in the top-left of the 3D panel labels every visual element (object
particles / gripper 1 / gripper 2 / achieved force / goal force).
Implementation uses OpenCV `VideoWriter` with the `mp4v` codec (no
ffmpeg needed in the env).

**Four videos rendered**, full trajectories at 15 fps × 1600×720:

| File | Case | Material | Frames | Duration | Size |
|---|---|---|---|---|---|
| `double_lift_cloth_1.mp4` | best cloth (err 0.22) | cloth | 116 | 7.7 s | 2.08 MB |
| `single_push_rope_1.mp4` | best rope (err 0.26) | rope | 92 | 6.1 s | 1.99 MB |
| `single_lift_sloth.mp4` | best sloth (err 0.29) | sloth | 85 | 5.7 s | 1.70 MB |
| `double_lift_cloth_3__ramp_FAILURE.mp4` | cloth ramp blow-up (err 6.55) | cloth | 118 | 7.9 s | 1.91 MB |

The first three demonstrate working closed-loop control on each
material. The fourth is the limitations exhibit — viewer sees the
achieved force diverge from the goal ramp.

Iteration notes (Step 5 video pass):
1. Initial v1 was 58 frames because of `--stride 2`; defaulted to full
   stride for v2.
2. Initial v1 had the 3D plot's z-axis labels bleeding into the right
   2D panel. Fixed by replacing `gridspec + tight_layout` with explicit
   `fig.add_axes([left, bottom, w, h])` positioning, giving the two
   panels a hard horizontal gap. Also shrank 3D tick label fonts and
   reduced label padding.
3. CLI `--width` and `--height` defaults bumped to 1600×720 (was
   1280×640) for the cleaner aspect ratio.
4. Added an explicit 3D legend (proxy artists for the scatters + arrow
   line styles) so viewers can identify the red dots as "gripper 1"
   without needing prose explanation. Same legend works for both
   single-control (only "gripper 1" entry shown) and double-control
   cases.

## Side-finding 2026-05-23 — PhysTwin's "controller" is a 30-point cluster, not a single point

The rope video's apparently-scattered red dots prompted a useful
investigation. For `single_push_rope_1`:

- K = 30 controller particles per group.
- At frame 0 they span a **21 cm bounding box** (median pairwise
  distance 7.2 cm).
- The rope itself is 48 cm long, so the controller cluster spans
  ~44% of the rope's spatial extent.

These 30 points are real tracked hand-contact particles from
PhysTwin's RGB-D data processing pipeline — the human's palm/fingers/
wrist as visible in the original video, not a synthetic geometry.

Connects to the Step 3 G3 finding (controller motion is not rigid per
group, median 1.4 mm/frame intra-group deviation). Our policy outputs
a centroid Δ per group and the closed-loop driver rigid-translates the
30 points by that Δ — that's the source of the ~8 mm rigid-replay
error we measured.

Worth a one-liner in the demo: "the gripper is a 30-particle cluster
from the original video tracking, not a single point; the policy
commands the cluster centroid."

Reusable: `python render_rollout_video.py --rollout <npz> --out <mp4>`
on any of the 14 rollouts.

## Carry-forward — what's still owed

1. **More videos if time permits.** Highest next candidates:
   `single_lift_sloth.mp4` (sloth's best, err 0.29), one limitation
   video (e.g. `double_lift_cloth_3__policy_ramp.mp4` to show the
   blow-up).
2. **Improvements pass.** Per user request, after this preliminary set
   we go back to the open questions:
   - **Fix B** (retrain with hindsight goal augmentation) — most
     likely to help cloth/sloth ramp failures and may improve
     long-trajectory drift.
   - **MPC over the simulator** (Direction 3 from the original
     project options) — the principled answer to the release problem.
   - **Outlier handling** — single_push_rope_4 specifically; the
     policy's force-scale calibration could be improved by reweighting
     training rows by case force range.

## Headline numbers for the demo (cheat sheet)

| Material | Profile | n | Mean err_ratio | Controlled |
|---|---|---|---|---|
| **cloth** | **replay** | **4** | **0.36** | **4/4** |
| rope | replay | 4 | 0.75 (0.39 excl. outlier) | 3/4 |
| sloth | replay | 3 | 0.87 (0.59 excl. outlier) | 2/3 |
| rope | ramp | 1 | 0.66 | 1/1 |
| cloth | ramp | 1 | 6.55 | 0/1 |
| sloth | ramp | 1 | 5.06 | 0/1 |

**One-sentence summary**: "We turned PhysTwin's spring-mass simulator
into a closed-loop force-aware actuator for deformable objects;
in-distribution force tracking works (cloth 36% RMS error), out-of-
distribution targets don't, and the failure modes are characterized."
