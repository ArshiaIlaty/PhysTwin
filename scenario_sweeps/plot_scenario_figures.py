"""plot_advisor_figures.py — visualizations addressing advisor feedback.

Generates:
  - ground_truth_pipeline.png   — where force labels come from
  - feature_descriptor.png        — 35-dim input breakdown
  - data_force_vs_features.png    — look at the data (force + key features over time)
  - split_random_block.png        — what random_block means (NOT physics perturbation)
  - split_cross_case.png          — train vs test cases per material
  - prior_work_comparison.png     — novelty vs VLA / PhysTwin / GS-Dynamics

Usage:
  python arshia_work/plot_advisor_figures.py
  python arshia_work/plot_advisor_figures.py --out_dir docs/assets/arshia
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from train_models import load_dataset, random_block_split

OUT_DEFAULT = "arshia_work/forward_force_results/figures"

TYPE_NAMES = ["rope", "cloth", "sloth", "toy"]

SUMMARY_FEATURE_NAMES = (
    [f"centroid_disp_{a}" for a in "xyz"]
    + [f"bbox_stretch_{a}" for a in "xyz"]
    + [f"max_abs_{a}" for a in "xyz"]
    + [f"mean_abs_{a}" for a in "xyz"]
    + [f"std_disp_{a}" for a in "xyz"]
    + ["ke_proxy", "mean_disp_mag", "max_disp_mag"]
)

NEW_FEATURE_NAMES = [
    "ctrl_centroid_disp_x", "ctrl_centroid_disp_y", "ctrl_centroid_disp_z",
    "nearest_dist", "mean_contact_dist",
    "rel_motion_x", "rel_motion_y", "rel_motion_z",
    "mean_vel_mag", "max_vel_mag",
    "centroid_vel_x", "centroid_vel_y", "centroid_vel_z",
]


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")


def plot_ground_truth_pipeline(out_path: str) -> None:
    """Diagram: video → PhysTwin sim → spring forces → y_net label."""
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    boxes = [
        (0.2, 1.0, 1.8, 1.2, "Real video\n(deformable object\n+ gripper)", "#dbeafe"),
        (2.4, 1.0, 1.8, 1.2, "PhysTwin\noptimization\n(spring-mass fit)", "#e0e7ff"),
        (4.6, 1.0, 1.8, 1.2, "Simulator\nrollout\n(particle positions)", "#fef3c7"),
        (6.8, 1.0, 1.8, 1.2, "Controller–object\nspring forces\n(per gripper group)", "#fce7f3"),
        (9.0, 1.0, 1.8, 1.2, "ML label\ny_net = Σ F_group\n(3D wrench, Newtons)", "#d1fae5"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="#374151", lw=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    for x0, x1 in [(2.0, 2.4), (4.2, 4.6), (6.4, 6.8), (8.6, 9.0)]:
        ax.annotate("", xy=(x1, 1.6), xytext=(x0, 1.6),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#374151"))

    ax.text(5.5, 2.55,
            "Ground truth is NOT from a physical force sensor — it is computed inside the fitted simulator",
            ha="center", fontsize=10, fontweight="bold")
    ax.text(5.5, 0.35,
            "extract_force_data() in trainer_warp.py  →  dataset_v2/*.npz  →  train MLP",
            ha="center", fontsize=9, color="#4b5563")
    _save(fig, out_path)


def plot_feature_descriptor(out_path: str) -> None:
    """35-dim input vector: 31 deformation stats + 4 material one-hot."""
    groups = [
        ("Object deformation (18)", SUMMARY_FEATURE_NAMES, "#059669"),
        ("Controller & contact (7)", NEW_FEATURE_NAMES[:7], "#2563eb"),
        ("Velocity (6)", NEW_FEATURE_NAMES[7:], "#7c3aed"),
        ("Material tag (4)", [f"type_{c}" for c in TYPE_NAMES], "#d97706"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = 0
    yticks, ylabels = [], []
    for gname, names, color in groups:
        for i, name in enumerate(names):
            ax.barh(y, 1, color=color, alpha=0.85, height=0.7)
            ax.text(1.02, y, name, va="center", fontsize=8)
            yticks.append(y)
            ylabels.append(str(y + 1))
            y += 1
        y += 0.35

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlim(0, 1.5)
    ax.set_xlabel("Feature index (conceptual)")
    ax.set_title("Input vector X (35-d): fixed-size summary over variable # of particles", fontsize=11)
    legend = [
        mpatches.Patch(color="#059669", label="How much did the object stretch/move?"),
        mpatches.Patch(color="#2563eb", label="Where is the gripper relative to the object?"),
        mpatches.Patch(color="#7c3aed", label="How fast are particles moving?"),
        mpatches.Patch(color="#d97706", label="Which material category? (one-hot)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8)
    ax.text(0.02, -0.08,
            "Intuition: collapse thousands of particle positions into a compact 'deformation fingerprint' each frame.",
            transform=ax.transAxes, fontsize=9, style="italic")
    ax.invert_yaxis()
    _save(fig, out_path)


def plot_data_force_vs_features(
    dataset_dir: str,
    case_name: str,
    out_path: str,
) -> None:
    """Force magnitude vs interpretable features over time — 'look at the data'."""
    npz_path = os.path.join(dataset_dir, f"{case_name}.npz")
    if not os.path.isfile(npz_path):
        print(f"  skip data plot — missing {npz_path}")
        return

    d = np.load(npz_path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y_net = d["y_net"].astype(np.float32)
    names = list(d["feature_names"])
    T = len(y_net)
    t = np.arange(T)
    force_mag = np.linalg.norm(y_net, axis=1)

    # Pick readable feature indices
    idx = {}
    for key, candidates in [
        ("centroid_y", ["centroid_disp_y"]),
        ("bbox_y", ["bbox_stretch_y"]),
        ("nearest", ["nearest_dist"]),
        ("vel", ["mean_vel_mag"]),
    ]:
        for c in candidates:
            if c in names:
                idx[key] = names.index(c)
                break

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    ax = axes[0]
    ax.plot(t, force_mag, color="#dc2626", lw=2)
    ax.set_ylabel("|F| (N)")
    ax.set_title(f"Data sample: {case_name} — sim force label vs deformation features")
    ax.grid(True, alpha=0.3)
    ax.text(0.99, 0.92, "Label = sim spring force (ground truth)", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#dc2626")

    specs = [
        ("centroid_y", "centroid_disp_y", "Object centroid displacement"),
        ("bbox_y", "bbox_stretch_y", "Bounding-box stretch"),
        ("nearest", "nearest_dist", "Nearest particle–gripper distance"),
        ("vel", "mean_vel_mag", "Mean particle speed"),
    ]
    colors = ["#059669", "#2563eb", "#7c3aed", "#d97706"]
    for ax, (key, _, title), color in zip(axes[1:], specs, colors):
        if key in idx:
            ax.plot(t, X[:, idx[key]], color=color, lw=1.5)
        ax.set_ylabel(title.split()[0], fontsize=8)
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame index (timestep in trajectory)")
    fig.tight_layout()
    _save(fig, out_path)


def plot_random_block_split(
    dataset_dir: str,
    category: str,
    seed: int,
    out_path: str,
) -> None:
    """Timeline diagram: contiguous held-out block per trajectory."""
    by_cat = load_dataset(
        dataset_dir, target_key="net", clip_percentile=99.0,
        exclude_cases=("single_push_sloth",), exclude_categories=("toy",),
        add_type_feature=True,
    )
    cases = by_cat.get(category, [])
    if not cases:
        print(f"  skip random_block plot — no {category} cases")
        return

    rng = np.random.RandomState(seed)
    # Match train_models: one rng iterates all categories in sorted order
    test_info = []
    for cat, cat_cases in by_cat.items():
        _, test_cases = random_block_split(cat_cases, test_ratio=0.2, rng=rng)
        if cat == category:
            for tc in test_cases:
                # parse case_name__test[start:end]
                base = tc["case_name"].split("__test[")[0]
                bracket = tc["case_name"].split("__test[")[1].rstrip("]")
                start, end = map(int, bracket.split(":"))
                orig = next(c for c in cat_cases if c["case_name"] == base)
                test_info.append((base, len(orig["X"]), start, end))

    n = len(test_info)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(2.5, 1.2 * n)), squeeze=False)
    for ax, (name, T, start, end) in zip(axes[:, 0], test_info):
        train_mask = np.ones(T, dtype=bool)
        train_mask[start:end] = False
        for i in range(T):
            color = "#fca5a5" if not train_mask[i] else "#86efac"
            ax.bar(i, 1, color=color, width=1.0, edgecolor="none")
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.set_ylabel(name.replace("_", "\n"), fontsize=7, rotation=0, ha="right", va="center")
        ax.text(start + (end - start) / 2, 0.5, f"TEST\n[{start}:{end}]",
                ha="center", va="center", fontsize=7, fontweight="bold", color="#991b1b")

    fig.suptitle(
        f"random_block split (seed={seed}) — contiguous held-out frames, NOT physics perturbations",
        fontsize=11, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "Green = train frames  |  Red = test frames (~20% block per trajectory). "
             "Question: can the MLP interpolate force within a seen manipulation?",
             ha="center", fontsize=9, transform=fig.transFigure)
    legend = [mpatches.Patch(color="#86efac", label="Train"),
              mpatches.Patch(color="#fca5a5", label="Test (contiguous block)")]
    fig.legend(handles=legend, loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)


def plot_cross_case_split(out_path: str) -> None:
    """Table figure: which cases train vs test per category."""
    splits = {
        "cloth": {
            "train": ["double_lift_cloth_1", "double_lift_cloth_3", "single_clift_cloth_3"],
            "test": ["single_clift_cloth_1"],
        },
        "rope": {
            "train": ["single_push_rope_1", "single_push_rope_4"],
            "test": ["rope_double_hand"],
        },
        "sloth": {
            "train": ["double_stretch_sloth"],
            "test": ["double_lift_sloth"],
        },
    }
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    rows = []
    for cat, sp in splits.items():
        rows.append([cat.upper(), ", ".join(sp["train"]), ", ".join(sp["test"])])
    table = ax.table(
        cellText=rows,
        colLabels=["Material", "Train cases (entire trajectories)", "Test cases (held out)"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e5e7eb")
            cell.set_text_props(fontweight="bold")
        elif c == 2:
            cell.set_facecolor("#fef3c7")
    ax.set_title(
        "cross_case split — question: does the model generalize to a new manipulation?",
        fontsize=11, fontweight="bold", pad=20)
    ax.text(0.5, -0.08,
            "Example: rope trains on single-hand push, tests on two-hand — different regime → poor R².",
            ha="center", transform=ax.transAxes, fontsize=9, style="italic")
    _save(fig, out_path)


def plot_prior_work_comparison(out_path: str) -> None:
    """Novelty vs VLA, PhysTwin, GS-Dynamics."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")

    headers = ["", "Typical VLA", "GS-Dynamics / Spring-Gaussian", "PhysTwin (base)", "Our extension"]
    rows = [
        ["Input", "RGB + language", "Images / Gaussians", "Video → particle sim", "Sim deformation summaries"],
        ["Output", "Robot actions", "Future Gaussians", "Interactive sim", "Force wrench OR gripper Δ"],
        ["Supervision", "Human demos", "Video reconstruction", "Rendering + physics fit", "Sim spring forces"],
        ["Force labels", "None (implicit)", "None", "Visualization only", "Explicit y_net per frame"],
        ["Our question", "—", "—", "Reconstruct & simulate", "Learn force ↔ deformation mappings"],
    ]

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 2.0)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#374151")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 4:
            cell.set_facecolor("#d1fae5")
        elif c == 0:
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(fontweight="bold")

    ax.set_title("Why this is not a VLA — comparison to recent prior work", fontsize=12, fontweight="bold", y=0.98)
    _save(fig, out_path)


def plot_open_loop_teacher_forcing(out_path: str) -> None:
    """Open-loop = teacher forcing: replay fixed motion, no feedback."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    ax.text(5, 3.15, "Open-loop scenario sweep = teacher forcing (recorded motion replayed every time)",
            ha="center", fontsize=11, fontweight="bold")

    ax.add_patch(mpatches.FancyBboxPatch((0.3, 1.2), 2.2, 1.3, boxstyle="round", fc="#dbeafe", ec="#374151"))
    ax.text(1.4, 1.85, "Fixed motion\nscript", ha="center", va="center", fontsize=10)

    ax.add_patch(mpatches.FancyBboxPatch((3.2, 1.2), 2.2, 1.3, boxstyle="round", fc="#fef3c7", ec="#374151"))
    ax.text(4.3, 1.85, "PhysTwin sim\n(no feedback)", ha="center", va="center", fontsize=10)

    ax.add_patch(mpatches.FancyBboxPatch((6.1, 1.2), 2.2, 1.3, boxstyle="round", fc="#fce7f3", ec="#374151"))
    ax.text(7.2, 1.85, "Measure\npeak force", ha="center", va="center", fontsize=10)

    ax.annotate("", xy=(3.2, 1.85), xytext=(2.5, 1.85), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(6.1, 1.85), xytext=(5.4, 1.85), arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.text(5, 0.45,
            "Knobs changed: stiffness ×{0.5,1,2} and surface proxy  |  "
            "Finding: stiffer object → higher peak force (+40–90%) for the same motion.",
            ha="center", fontsize=9, wrap=True)
    _save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="dataset_v2")
    parser.add_argument("--out_dir", default=OUT_DEFAULT)
    parser.add_argument("--case_name", default="single_clift_cloth_3")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Writing advisor figures to {args.out_dir}/")
    plot_ground_truth_pipeline(os.path.join(args.out_dir, "ground_truth_pipeline.png"))
    plot_feature_descriptor(os.path.join(args.out_dir, "feature_descriptor.png"))
    plot_data_force_vs_features(args.dataset_dir, args.case_name,
                                os.path.join(args.out_dir, "data_force_vs_features.png"))
    plot_random_block_split(args.dataset_dir, "cloth", args.seed,
                            os.path.join(args.out_dir, "split_random_block.png"))
    plot_cross_case_split(os.path.join(args.out_dir, "split_cross_case.png"))
    plot_prior_work_comparison(os.path.join(args.out_dir, "prior_work_comparison.png"))
    plot_open_loop_teacher_forcing(os.path.join(args.out_dir, "open_loop_teacher_forcing.png"))
    print("Done.")


if __name__ == "__main__":
    main()
