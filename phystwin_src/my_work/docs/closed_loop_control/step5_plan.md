# Step 5 plan — Preliminary figures (and optional video)

**Posture**: this is the preliminary deliverable so we have *something*
to show if time runs out before improvements (Fix B / MPC / etc.) land.
Prioritize cheap-but-clear static figures. Defer videos to a follow-up
unless the static plots come together in <1 hour.

## Inputs

- `results/eval_closed_loop/{case}__{profile}.npz` × 14 (from Step 4)
- `results/eval_closed_loop/summary.json` with per-material aggregates

## Outputs

```
results/figures/closed_loop/
  01_per_material_bars.png      headline bar chart
  02_force_tracking_grid.png    9 controlled replay cases, F_achieved vs F_goal
  03_limitations_panel.png      3 uncontrolled cases (rope_4 outlier, sloth_stretch,
                                cloth ramp blow-up)
  04_highlight_cloth.png        single best case (double_lift_cloth_1, err=0.22)
                                with annotated phases
  05_ramp_failure.png           3 ramp rollouts side-by-side (rope OK, cloth/sloth break)
```

## Figure specs

### 01 — Per-material bars (headline)
- X: material × profile (cloth/replay, rope/replay, sloth/replay, then ramps)
- Y: mean err_ratio with std error bars
- Annotate "uncontrolled" count above each bar
- Horizontal line at y=1.0 (the "control breakdown" threshold)

### 02 — Force tracking grid (the meat)
- 9 controlled replay rollouts arranged 3×3
- Per subplot: time-axis with `‖F_achieved(t)‖` (solid) overlaid on
  `‖F_goal(t)‖` (dashed)
- Title: case name + err_ratio
- Common y-axis scale within each material

### 03 — Limitations panel
- 3 rollouts: single_push_rope_4 (rope outlier), double_stretch_sloth
  (sloth outlier), double_lift_cloth_3 ramp (blow-up)
- Same overlay style; highlight where things go wrong

### 04 — Highlight (demo prominence)
- One controlled case (double_lift_cloth_1, err 0.22)
- Larger axes; per-axis force decomposition (Fx, Fy, Fz separate panels)
- Achieved vs goal for each axis

### 05 — Ramp comparison
- 3 ramps side-by-side: rope (borderline), cloth (blow-up), sloth (blow-up)
- Shows the policy's directional asymmetry

## Architecture

One script `my_work/code/make_closed_loop_figures.py`. Loads npzs, walks
the figure list, writes PNGs. No CUDA, no PhysTwin imports — pure
numpy + matplotlib. Runs on login node in seconds.

## Verification gates

Cheap because no compute beyond plotting.

| Gate | Check | Pass criterion |
|---|---|---|
| G1 | Script runs without error | All 5 PNGs written |
| G2 | Quick visual sanity | Plots have correct titles, labels, axes; values match summary.json |

That's it — figures are direct projections of the npz data. No new
analysis to verify.

## Out of scope for first cut

- Videos (matplotlib 3D animation OR Open3D offscreen render). If time
  permits after static figures, add a single 1-case mp4 (cloth highlight).
- Gaussian-splatting render — too heavy for preliminary deliverable.
- Per-motion-type breakdown (eval was replay-mode only — no synthetic
  motion analysis yet).

## Scope estimate

| Phase | Time |
|---|---|
| Write `make_closed_loop_figures.py` | 1 hr |
| Generate + visually inspect | 15 min |
| Write step5_review.md | 30 min |
| **Total (static figures only)** | **~2 hr** |
| Optional video adapter (1 case) | +1.5 hr |

## Acceptance criteria

1. 5 PNGs under `results/figures/closed_loop/`.
2. Numbers on the plots match `summary.json` (no transcription error).
3. step5_review.md written with brief commentary on each figure.
4. Demo-ready static set.
