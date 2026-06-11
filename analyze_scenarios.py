"""analyze_scenarios.py — plot force sensitivity from dataset_scenarios/*.npz."""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_scenarios(scenario_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(scenario_dir, "*.npz"))):
        d = np.load(path, allow_pickle=True)
        rows.append(
            {
                "path": path,
                "case_name": str(d["case_name"]),
                "object_category": str(d["object_category"]),
                "source_donor": str(d.get("source_donor", "")),
                "motion_type": str(d.get("motion_type", "")),
                "stiffness_scale": float(d["stiffness_scale"]),
                "surface_preset": str(d["surface_preset"]),
                "collide_elas": float(d["collide_elas"]),
                "collide_fric": float(d["collide_fric"]),
                "peak_net_force": float(d["peak_net_force"]),
                "mean_net_force": float(d["mean_net_force"]),
                "y_net": d["y_net"].astype(np.float32),
            }
        )
    return rows


def plot_peak_vs_stiffness(rows: list[dict], out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = sorted({r["object_category"] for r in rows})
    surfaces = sorted({r["surface_preset"] for r in rows})
    for cat in cats:
        for surf in surfaces:
            sub = [r for r in rows if r["object_category"] == cat and r["surface_preset"] == surf]
            if not sub:
                continue
            sub = sorted(sub, key=lambda r: r["stiffness_scale"])
            xs = [r["stiffness_scale"] for r in sub]
            ys = [r["peak_net_force"] for r in sub]
            ax.plot(xs, ys, marker="o", label=f"{cat} / {surf}")
    ax.set_xlabel("Object stiffness scale")
    ax.set_ylabel("Peak net force (N)")
    ax.set_title("Scenario sweep: peak force vs stiffness")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "peak_force_vs_stiffness.png"), dpi=150)
    plt.close(fig)


def plot_peak_vs_surface(rows: list[dict], out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = sorted({r["object_category"] for r in rows})
    stiffness_vals = sorted({r["stiffness_scale"] for r in rows})
    x_labels = sorted({r["surface_preset"] for r in rows})
    x_pos = np.arange(len(x_labels))
    width = 0.25
    for i, k in enumerate(stiffness_vals):
        means = []
        for surf in x_labels:
            vals = [
                r["peak_net_force"]
                for r in rows
                if r["stiffness_scale"] == k and r["surface_preset"] == surf
            ]
            means.append(np.mean(vals) if vals else 0.0)
        ax.bar(x_pos + i * width, means, width=width, label=f"stiffness×{k:g}")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Peak net force (N)")
    ax.set_title("Scenario sweep: peak force vs surface (mean over materials)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "peak_force_vs_surface.png"), dpi=150)
    plt.close(fig)


def plot_force_trajectories(rows: list[dict], out_dir: str) -> None:
    """One panel per material: force magnitude over time for baseline stiffness on concrete."""
    cats = sorted({r["object_category"] for r in rows})
    fig, axes = plt.subplots(len(cats), 1, figsize=(9, 3 * len(cats)), sharex=True)
    if len(cats) == 1:
        axes = [axes]
    for ax, cat in zip(axes, cats):
        for surf in sorted({r["surface_preset"] for r in rows}):
            sub = [
                r for r in rows
                if r["object_category"] == cat
                and r["surface_preset"] == surf
                and abs(r["stiffness_scale"] - 1.0) < 1e-6
            ]
            if not sub:
                continue
            y = sub[0]["y_net"]
            mag = np.linalg.norm(y, axis=-1)
            ax.plot(mag, label=surf)
        ax.set_ylabel(f"{cat}\n|F| (N)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame")
    fig.suptitle("Net force magnitude over time (stiffness×1.0)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "force_trajectories.png"), dpi=150)
    plt.close(fig)


def write_summary_table(rows: list[dict], out_dir: str) -> None:
    table = []
    for r in rows:
        table.append(
            {
                "case": r["case_name"],
                "material": r["object_category"],
                "donor": r["source_donor"],
                "stiffness_scale": r["stiffness_scale"],
                "surface": r["surface_preset"],
                "peak_net_force_N": r["peak_net_force"],
                "mean_net_force_N": r["mean_net_force"],
            }
        )
    with open(os.path.join(out_dir, "scenario_table.json"), "w") as f:
        json.dump(table, f, indent=2)
    print(f"Wrote {len(table)} rows to {out_dir}/scenario_table.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_dir", type=str, default="dataset_scenarios")
    parser.add_argument("--out_dir", type=str, default="figures/scenarios")
    args = parser.parse_args()

    rows = load_scenarios(args.scenario_dir)
    if not rows:
        raise SystemExit(f"No scenario npz files in {args.scenario_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    plot_peak_vs_stiffness(rows, args.out_dir)
    plot_peak_vs_surface(rows, args.out_dir)
    plot_force_trajectories(rows, args.out_dir)
    write_summary_table(rows, args.out_dir)
    print(f"Figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
