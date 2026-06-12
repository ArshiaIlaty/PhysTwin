#!/usr/bin/env python3
"""
render_rollout_video.py — Side-by-side mp4 of a closed-loop rollout

Left panel  (3D): particle cloud + controller points + force vectors per group.
Right panel (2D): goal vs achieved force magnitude over time with vertical
                  cursor at current frame.

Writes mp4 via OpenCV (mp4v codec — no ffmpeg needed).

Plan: my_work/docs/closed_loop_control/step5_plan.md (deferred video item).

Usage:
  python my_work/code/render_rollout_video.py \\
      --rollout my_work/results/eval_closed_loop/double_lift_cloth_1__policy_recorded_goal.npz \\
      --out my_work/results/figures/closed_loop/videos/double_lift_cloth_1.mp4
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

logger = logging.getLogger("render_rollout")


MATERIAL_COLORS = {"rope": "#1f77b4", "cloth": "#ff7f0e",
                   "sloth": "#2ca02c", "toy": "#9467bd"}


class Arrow3D(FancyArrowPatch):
    """3D quiver arrow that updates correctly under matplotlib's 3D projection."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (xs, ys, zs)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._xyz
        xs, ys, _ = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(np.asarray(zs3d))


def fig_to_bgr(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = buf.reshape((h, w, 4))[:, :, [1, 2, 3]]  # ARGB -> RGB
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return bgr


def render_video(rollout_path: Path, out_path: Path, fps: int = 15,
                  stride: int = 1, width: int = 1600, height: int = 720,
                  particle_size: float = 4.0, arrow_scale_m: float = 0.15,
                  particle_only: bool = False):
    d = np.load(rollout_path, allow_pickle=True)
    positions = d["positions"].astype(np.float32)       # [T, N, 3]
    ctrl_pos = d["controller_pos"].astype(np.float32)   # [T, K, 3]
    forces = d["forces"].astype(np.float32)             # [T, n_ctrl, 3]
    F_goal = d["F_goal"].astype(np.float32)             # [T, 2, 3]
    case_name = str(d["case_name"])
    material = str(d["material"])
    n_ctrl = int(d["n_ctrl_parts"])
    group_ids = np.asarray(d["group_ids"], dtype=np.int8)  # [K]

    T = positions.shape[0]
    frames = list(range(0, T, stride))
    logger.info("rendering %d frames (stride=%d) for %s",
                len(frames), stride, case_name)

    # Pre-compute 3D axis limits with a small pad
    all_pts = np.concatenate([positions.reshape(-1, 3),
                              ctrl_pos.reshape(-1, 3)], axis=0)
    mn, mx = all_pts.min(axis=0), all_pts.max(axis=0)
    pad = 0.1 * (mx - mn).max()
    xlim = (mn[0] - pad, mx[0] + pad)
    ylim = (mn[1] - pad, mx[1] + pad)
    zlim = (mn[2] - pad, mx[2] + pad)

    # Force magnitudes (per-frame total, summed across groups)
    F_a_total = np.linalg.norm(forces.sum(axis=1), axis=-1)         # [T]
    F_g_total = np.linalg.norm(F_goal[:, :n_ctrl].sum(axis=1), axis=-1)
    t_arr = np.arange(T)
    F_a_kN = F_a_total / 1000.0
    F_g_kN = F_g_total / 1000.0
    y_max = max(F_a_kN.max(), F_g_kN.max()) * 1.1
    y_min = 0.0

    # Arrow scale: max achieved force magnitude → arrow_scale_m meters
    max_force = max(float(np.linalg.norm(forces, axis=-1).max()), 1e-3)
    scale = arrow_scale_m / max_force                          # m per N
    max_goal_force = max(float(np.linalg.norm(F_goal[:, :n_ctrl], axis=-1).max()), 1e-3)
    scale_g = arrow_scale_m / max_goal_force

    # Pick a consistent material color for achieved force
    col_a = MATERIAL_COLORS.get(material, "tab:blue")
    col_g = "black"

    # ---- one figure, redrawn per frame -------------------------------
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    # Manually position the two panels with a clear horizontal gap so the
    # 3D axis tick labels / z-axis don't bleed into the right (2D) panel.
    # Coordinates are (left, bottom, width, height) in figure fractions.
    if particle_only:
        ax3d = fig.add_axes([0.04, 0.04, 0.92, 0.88], projection="3d")
        ax2d = None
    else:
        ax3d = fig.add_axes([0.02, 0.08, 0.45, 0.82], projection="3d")
        ax2d = fig.add_axes([0.58, 0.10, 0.39, 0.78])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not vw.isOpened():
        raise RuntimeError(f"VideoWriter failed to open: {out_path}")

    for i, t in enumerate(frames):
        ax3d.clear()
        if ax2d is not None:
            ax2d.clear()

        # --- 3D panel ---
        ax3d.scatter(positions[t, :, 0], positions[t, :, 1], positions[t, :, 2],
                      s=particle_size, c="lightsteelblue", alpha=0.35,
                      edgecolors="none", label="object particles")
        # Per-group controller points
        ctrl_colors = ["red", "darkmagenta"]
        ctrl_labels = ["gripper 1", "gripper 2"]
        for g in range(n_ctrl):
            mask = group_ids == g
            ax3d.scatter(ctrl_pos[t, mask, 0], ctrl_pos[t, mask, 1],
                          ctrl_pos[t, mask, 2],
                          s=40, c=ctrl_colors[g],
                          edgecolors="black", linewidths=0.5,
                          label=ctrl_labels[g])
            # Force arrows from this group's centroid
            cen = ctrl_pos[t, mask].mean(axis=0)
            f_a = forces[t, g]
            f_g = F_goal[t, g]
            # Achieved (solid colored)
            tip_a = cen + f_a * scale
            arr_a = Arrow3D(
                [cen[0], tip_a[0]], [cen[1], tip_a[1]], [cen[2], tip_a[2]],
                mutation_scale=15, arrowstyle="-|>", color=col_a,
                linewidth=2.5,
            )
            ax3d.add_artist(arr_a)
            # Goal (dashed black, dashed via linestyle)
            tip_g = cen + f_g * scale_g
            arr_g = Arrow3D(
                [cen[0], tip_g[0]], [cen[1], tip_g[1]], [cen[2], tip_g[2]],
                mutation_scale=12, arrowstyle="-|>", color=col_g,
                linewidth=1.2, linestyle="dashed", alpha=0.7,
            )
            ax3d.add_artist(arr_g)
        ax3d.set_xlim(xlim); ax3d.set_ylim(ylim); ax3d.set_zlim(zlim)
        ax3d.set_xlabel("x (m)", fontsize=9, labelpad=-2)
        ax3d.set_ylabel("y (m)", fontsize=9, labelpad=-2)
        ax3d.set_zlabel("z (m)", fontsize=9, labelpad=-2)
        ax3d.tick_params(axis="both", labelsize=7)
        ax3d.set_title(f"{case_name}  ({material})  ·  frame {t} / {T - 1}",
                        fontsize=11, pad=8)
        # Fixed view angle (looks reasonable for hanging cloth lifts)
        ax3d.view_init(elev=20, azim=-60)

        # Legend for the 3D panel (scatter labels + arrow proxies).
        from matplotlib.lines import Line2D
        legend_entries = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="lightsteelblue", markersize=8,
                   label="object particles"),
        ]
        for g in range(n_ctrl):
            legend_entries.append(Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=ctrl_colors[g], markeredgecolor="black",
                markersize=9, label=ctrl_labels[g],
            ))
        legend_entries.append(Line2D(
            [0], [0], color=col_a, linewidth=2.5,
            label="achieved force (arrow)",
        ))
        legend_entries.append(Line2D(
            [0], [0], color="black", linewidth=1.2, linestyle="dashed",
            label="goal force (arrow)",
        ))
        ax3d.legend(handles=legend_entries, loc="upper left", fontsize=7,
                     framealpha=0.85)

        # --- 2D panel (force tracking) ---
        if ax2d is not None:
            ax2d.plot(t_arr, F_g_kN, "k--", linewidth=1.5, label="goal", alpha=0.8)
            ax2d.plot(t_arr, F_a_kN, color=col_a, linewidth=2.0, label="achieved")
            ax2d.axvline(t, color="red", linewidth=1.5, alpha=0.7, label="now")
            # marker at current time
            ax2d.scatter([t], [F_a_kN[t]], color=col_a, s=50, zorder=5)
            ax2d.scatter([t], [F_g_kN[t]], facecolors="none",
                          edgecolors="black", s=50, zorder=5)
            ax2d.set_xlabel("frame")
            ax2d.set_ylabel("‖F‖  (kN)")
            ax2d.set_xlim(0, T - 1)
            ax2d.set_ylim(y_min, y_max)
            ax2d.set_title(
                f"force tracking  ·  achieved = {F_a_kN[t]:.1f} kN  |  goal = {F_g_kN[t]:.1f} kN",
                fontsize=10,
            )
            ax2d.grid(True, alpha=0.3)
            ax2d.legend(loc="upper left", fontsize=8)

        if not particle_only:
            fig.suptitle(
                f"PhysTwin force tracking  ·  achieved vs goal\n"
                f"colored arrow = achieved force (per controller group),  "
                f"dashed black arrow = goal force",
                fontsize=10, y=0.98,
            )
        # Don't call tight_layout — we use explicit add_axes positions.

        bgr = fig_to_bgr(fig)
        # Resize if matplotlib gave us a slightly different size
        if bgr.shape[1] != width or bgr.shape[0] != height:
            bgr = cv2.resize(bgr, (width, height))
        vw.write(bgr)
        if (i + 1) % 20 == 0:
            logger.info("  rendered %d / %d frames", i + 1, len(frames))

    vw.release()
    plt.close(fig)
    size_mb = out_path.stat().st_size / 1e6
    logger.info("wrote %s (%.2f MB, %d frames @ %d fps)",
                out_path, size_mb, len(frames), fps)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout", type=Path, required=True,
                    help="Path to a closed-loop rollout npz from eval_closed_loop/")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output .mp4 path")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--stride", type=int, default=1,
                    help="Render every Nth frame (1=all)")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--particle_only", action="store_true",
                    help="Render only the 3D particle + force-arrow panel (no 2D force plot)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    render_video(args.rollout, args.out, fps=args.fps, stride=args.stride,
                  width=args.width, height=args.height,
                  particle_only=args.particle_only)


if __name__ == "__main__":
    main()
