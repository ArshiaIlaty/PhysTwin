"""
stiffness_force_analysis.py
────────────────────────────────────────────────────────────────────────────
Extracts per-object spring stiffness statistics from PhysTwin reconstruction
outputs, loads Malak's force-prediction results, and produces:

  1. A correlation scatter plot  (force_error vs stiffness)
  2. A per-material bar chart    (mean force error per object type)
  3. A per-material violin plot  (error distribution)
  4. A printed summary table     (ready to paste into a paper)
  5. A CSV of all episode stats  (for sharing with Malak)

Usage
─────
    python stiffness_force_analysis.py \
        --phystwin_root  /path/to/PhysTwin \
        --force_results  /path/to/malak_force_predictions.csv \
        --output_dir     ./analysis_results

Expected file layout (PhysTwin repo after training)
─────────────────────────────────────────────────────
PhysTwin/
  experiments/
    double_stretch_sloth/          ← one folder per case
      params.pkl                   ← optimised α including spring stiffness
      spring_params.pkl            ← (alt) first-order optimised stiffness
      config.json                  ← object metadata incl. type label
    single_push_rope_1/
      params.pkl
      ...
  experiments_optimization/        ← zero-order stage outputs
    double_stretch_sloth/
      best_params.pkl

Expected force-predictions CSV columns (Malak's output)
─────────────────────────────────────────────────────────
  case_name        : str  e.g. "double_stretch_sloth"
  gt_force         : float  ground-truth force magnitude (N)
  pred_force       : float  predicted force magnitude (N)
  step             : int    timestep index (optional)

If Malak uses a different format, set FORCE_CSV_COLS below.
"""

import os
import pickle
import json
import glob
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless-safe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# ── column aliases ── adjust if Malak's CSV uses different names ─────────────
FORCE_CSV_COLS = {
    "case":   "case_name",
    "gt":     "gt_force",
    "pred":   "pred_force",
    "step":   "step",          # optional
}

# ── object-type mapping ── covers PhysTwin's known case prefixes ─────────────
OBJECT_TYPES = {
    "rope":     ["rope", "push_rope", "stretch_rope", "lift_rope"],
    "cloth":    ["cloth", "lift_cloth", "fold_cloth"],
    "stuffed":  ["sloth", "stuffed", "bunny", "animal", "toy"],
    "package":  ["package", "box", "delivery", "cardboard"],
}

PALETTE = {
    "rope":    "#1D9E75",
    "cloth":   "#378ADD",
    "stuffed": "#F5A623",
    "package": "#E05A4E",
    "unknown": "#888780",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Stiffness extraction
# ─────────────────────────────────────────────────────────────────────────────

def infer_object_type(case_name: str) -> str:
    name_lower = case_name.lower()
    for obj_type, keywords in OBJECT_TYPES.items():
        if any(kw in name_lower for kw in keywords):
            return obj_type
    return "unknown"


def load_params_pkl(pkl_path: str) -> Optional[dict]:
    """Load a PhysTwin params.pkl safely."""
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        warnings.warn(f"Could not load {pkl_path}: {e}")
        return None


def extract_stiffness_from_params(params: dict) -> Optional[np.ndarray]:
    """
    PhysTwin stores spring stiffness under various keys depending on
    which optimisation stage produced the file.  We try common names.

    After script_train.py (first-order):
        params["spring_stiffness"]  → shape (n_springs,)
        params["stiffness"]         → shape (n_springs,)
        params["k"]                 → shape (n_springs,)
        params["alpha"]["k"]        → same

    After script_optimize.py (zero-order):
        params["homogeneous_stiffness"]  → scalar or (1,)
        params["k_homo"]                 → scalar

    Returns a 1-D numpy array of stiffness values, or None.
    """
    candidates = [
        # first-order dense stiffness
        lambda p: np.array(p["spring_stiffness"]).ravel(),
        lambda p: np.array(p["stiffness"]).ravel(),
        lambda p: np.array(p["k"]).ravel(),
        lambda p: np.array(p["alpha"]["k"]).ravel(),
        lambda p: np.array(p["params"]["spring_stiffness"]).ravel(),
        # zero-order homogeneous (Malak's optimal_params.pkl uses this key)
        lambda p: np.array([p["global_spring_Y"]]).ravel(),
        lambda p: np.array([p["homogeneous_stiffness"]]).ravel(),
        lambda p: np.array([p["k_homo"]]).ravel(),
        lambda p: np.array([p["alpha"]["k_homo"]]).ravel(),
    ]
    for fn in candidates:
        try:
            vals = fn(params)
            if vals is not None and len(vals) > 0:
                return vals.astype(np.float32)
        except (KeyError, TypeError):
            continue
    return None


def extract_stiffness_stats(phystwin_root: str) -> pd.DataFrame:
    """
    Walk the experiments/ directory tree and extract stiffness statistics
    for every reconstructed case.

    Returns a DataFrame with columns:
        case_name, obj_type,
        k_mean, k_std, k_median, k_min, k_max,
        n_springs, source_file
    """
    root = Path(phystwin_root)
    records = []

    # Priority order for which pkl to load
    search_patterns = [
        "experiments/*/params.pkl",
        "experiments/*/spring_params.pkl",
        "experiments_optimization/*/optimal_params.pkl",
        "experiments_optimization/*/best_params.pkl",
        "experiments_optimization/*/params.pkl",
    ]

    visited = set()
    for pattern in search_patterns:
        for pkl_path in sorted(root.glob(pattern)):
            case_name = pkl_path.parent.name
            if case_name in visited:
                continue

            params = load_params_pkl(str(pkl_path))
            if params is None:
                continue

            k_vals = extract_stiffness_from_params(params)
            if k_vals is None or len(k_vals) == 0:
                print(f"  [warn] no stiffness found in {pkl_path}")
                continue

            visited.add(case_name)
            records.append({
                "case_name":   case_name,
                "obj_type":    infer_object_type(case_name),
                "k_mean":      float(np.mean(k_vals)),
                "k_std":       float(np.std(k_vals)),
                "k_median":    float(np.median(k_vals)),
                "k_min":       float(np.min(k_vals)),
                "k_max":       float(np.max(k_vals)),
                "n_springs":   len(k_vals),
                "source_file": str(pkl_path.relative_to(root)),
            })
            print(f"  ✓ {case_name:40s}  "
                  f"k_mean={np.mean(k_vals):.1f}  "
                  f"n_springs={len(k_vals)}  "
                  f"type={infer_object_type(case_name)}")

    if not records:
        raise FileNotFoundError(
            "No params.pkl files found under experiments/ or "
            "experiments_optimization/.  Run script_optimize.py and "
            "script_train.py first.")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load and process force predictions
# ─────────────────────────────────────────────────────────────────────────────

def load_force_results(csv_path: str) -> pd.DataFrame:
    """
    Load Malak's force prediction CSV.  Normalises column names.
    Computes per-step absolute and relative force error.
    """
    df = pd.read_csv(csv_path)

    # Rename columns to standard names
    rename = {}
    for std_name, col_name in FORCE_CSV_COLS.items():
        if col_name in df.columns:
            rename[col_name] = std_name
        else:
            # fuzzy match: find first column containing the key
            matches = [c for c in df.columns
                       if std_name in c.lower() or col_name in c.lower()]
            if matches:
                rename[matches[0]] = std_name
    df = df.rename(columns=rename)

    # Validate required columns
    required = {"case", "gt", "pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Force CSV missing columns: {missing}. "
            f"Available columns: {list(df.columns)}\n"
            f"Adjust FORCE_CSV_COLS at the top of this script.")

    df["abs_error"]  = (df["pred"] - df["gt"]).abs()
    df["rel_error"]  = df["abs_error"] / (df["gt"].abs() + 1e-8)
    df["error_sign"] = df["pred"] - df["gt"]   # positive = over-prediction

    return df


def aggregate_force_per_case(force_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-step force predictions to per-case summary statistics.
    """
    agg = force_df.groupby("case", sort=False).agg(
        gt_mean     = ("gt",        "mean"),
        pred_mean   = ("pred",      "mean"),
        mae         = ("abs_error", "mean"),
        mae_std     = ("abs_error", "std"),
        mre         = ("rel_error", "mean"),
        n_steps     = ("abs_error", "count"),
        over_pred_frac = ("error_sign",
                          lambda x: (x > 0).mean()),
        rmse        = ("abs_error",
                       lambda x: np.sqrt((x ** 2).mean())),
    ).reset_index()
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 3. Merge stiffness + force and run correlation
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_correlate(
    stiff_df: pd.DataFrame,
    force_agg: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """
    Inner-join stiffness stats with force aggregation on case_name.
    Returns merged DataFrame + dict of correlation statistics.
    """
    merged = stiff_df.merge(
        force_agg,
        left_on="case_name",
        right_on="case",
        how="inner",
    )

    if len(merged) == 0:
        raise ValueError(
            "No matching cases between stiffness and force data. "
            "Check that case names in the force CSV match the folder names "
            "in experiments/.")

    print(f"\n  Matched {len(merged)} cases "
          f"({len(stiff_df)} stiffness, {len(force_agg)} force)")

    # Correlation tests on k_mean vs MAE
    x = merged["k_mean"].values
    y = merged["mae"].values

    pearson_r,  pearson_p  = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    slope, intercept, _, _, _ = stats.linregress(x, y)

    corr_stats = {
        "n":          len(merged),
        "pearson_r":  pearson_r,
        "pearson_p":  pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "slope":      slope,
        "intercept":  intercept,
    }
    return merged, corr_stats


# ─────────────────────────────────────────────────────────────────────────────
# 4. Plotting
# ─────────────────────────────────────────────────────────────────────────────

# Paper-quality style
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   10,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8,
    "axes.linewidth":   0.6,
    "lines.linewidth":  1.5,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})
ONE_COL = 3.5
TWO_COL = 7.16


def plot_correlation(merged: pd.DataFrame, corr: dict, out_dir: str):
    """
    Figure 1: scatter plot of k_mean vs MAE with regression line,
    coloured by object type.
    """
    fig, ax = plt.subplots(figsize=(ONE_COL * 1.6, ONE_COL * 1.2))

    for obj_type, grp in merged.groupby("obj_type"):
        color = PALETTE.get(obj_type, PALETTE["unknown"])
        ax.scatter(
            grp["k_mean"], grp["mae"],
            label=obj_type.capitalize(),
            color=color, s=60, zorder=3, edgecolors="white", linewidths=0.5,
        )
        # annotate each point with case name (short version)
        for _, row in grp.iterrows():
            short = row["case_name"].replace("_", " ")[:18]
            ax.annotate(short,
                        (row["k_mean"], row["mae"]),
                        fontsize=6, color="#555555",
                        xytext=(4, 2), textcoords="offset points")

    # Regression line
    x_range = np.linspace(merged["k_mean"].min(),
                          merged["k_mean"].max(), 100)
    y_line  = corr["slope"] * x_range + corr["intercept"]
    ax.plot(x_range, y_line, "--", color="#555555", linewidth=1.0,
            label=f"Linear fit (r={corr['pearson_r']:.2f})")

    ax.set_xlabel("Spring stiffness k (mean, N/m)")
    ax.set_ylabel("Mean absolute force error (N)")
    ax.set_title(
        f"Force error vs. material stiffness\n"
        f"Pearson r = {corr['pearson_r']:.3f}  "
        f"(p = {corr['pearson_p']:.3f}),  "
        f"n = {corr['n']}")
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="both", linewidth=0.3, alpha=0.5)

    fig.tight_layout(pad=0.8)
    path = os.path.join(out_dir, "fig1_stiffness_vs_force_error.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_per_material_bar(merged: pd.DataFrame, out_dir: str):
    """
    Figure 2: grouped bar chart — mean MAE and mean k per object type.
    """
    grp = merged.groupby("obj_type").agg(
        mae_mean  = ("mae",    "mean"),
        mae_sem   = ("mae",    lambda x: x.std() / np.sqrt(len(x))),
        k_mean    = ("k_mean", "mean"),
        n         = ("mae",    "count"),
    ).reset_index().sort_values("k_mean")

    obj_types = grp["obj_type"].tolist()
    x = np.arange(len(obj_types))
    colors = [PALETTE.get(t, PALETTE["unknown"]) for t in obj_types]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TWO_COL, 2.6))

    # Left: force error
    bars = ax1.bar(x, grp["mae_mean"], color=colors,
                   alpha=0.88, zorder=3)
    ax1.errorbar(x, grp["mae_mean"], yerr=grp["mae_sem"],
                 fmt="none", color="#444", capsize=3, linewidth=0.8, zorder=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels([t.capitalize() for t in obj_types])
    ax1.set_ylabel("Mean absolute force error (N)")
    ax1.set_title("Force error by material")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)
    ax1.set_axisbelow(True)
    for bar, n in zip(bars, grp["n"]):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"n={n}", ha="center", va="bottom", fontsize=7)

    # Right: stiffness
    ax2.bar(x, grp["k_mean"], color=colors, alpha=0.88, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.capitalize() for t in obj_types])
    ax2.set_ylabel("Mean spring stiffness k (N/m)")
    ax2.set_title("Stiffness by material")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)
    ax2.set_axisbelow(True)

    fig.tight_layout(pad=0.8)
    path = os.path.join(out_dir, "fig2_per_material_bars.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_violin(merged: pd.DataFrame, out_dir: str):
    """
    Figure 3: violin plot of per-episode MAE distribution per object type,
    sorted by median stiffness so the trend is visible.
    """
    # Sort types by ascending k_mean
    order = (merged.groupby("obj_type")["k_mean"]
             .mean().sort_values().index.tolist())

    fig, ax = plt.subplots(figsize=(TWO_COL * 0.75, 2.8))

    data_by_type = [merged[merged["obj_type"] == t]["mae"].values
                    for t in order]
    data_by_type = [d for d in data_by_type if len(d) > 0]
    order        = [t for t in order
                    if len(merged[merged["obj_type"] == t]) > 0]

    parts = ax.violinplot(
        data_by_type,
        positions=range(len(order)),
        showmedians=True,
        showextrema=True,
    )
    for i, (pc, obj_type) in enumerate(
            zip(parts["bodies"], order)):
        pc.set_facecolor(PALETTE.get(obj_type, PALETTE["unknown"]))
        pc.set_alpha(0.75)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([t.capitalize() for t in order])
    ax.set_ylabel("Mean absolute force error (N)")
    ax.set_title("Force error distribution by material\n"
                 "(sorted by ascending stiffness →)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    legend_patches = [
        mpatches.Patch(color=PALETTE.get(t, PALETTE["unknown"]),
                       label=t.capitalize())
        for t in order
    ]
    ax.legend(handles=legend_patches, frameon=False, ncol=2, fontsize=7)
    fig.tight_layout(pad=0.8)
    path = os.path.join(out_dir, "fig3_violin_per_material.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_error_breakdown(merged: pd.DataFrame, out_dir: str):
    """
    Figure 4: over-prediction vs under-prediction fraction per material.
    Shows directional bias (does the policy systematically over-force
    stiff objects?).
    """
    grp = merged.groupby("obj_type").agg(
        over_frac  = ("over_pred_frac", "mean"),
        k_mean     = ("k_mean",          "mean"),
    ).reset_index().sort_values("k_mean")

    obj_types = grp["obj_type"].tolist()
    x = np.arange(len(obj_types))
    colors = [PALETTE.get(t, PALETTE["unknown"]) for t in obj_types]

    fig, ax = plt.subplots(figsize=(ONE_COL * 1.4, 2.6))
    ax.bar(x, grp["over_frac"], color=colors, alpha=0.88, zorder=3)
    ax.axhline(0.5, color="#555", linewidth=0.8, linestyle="--",
               label="No bias (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in obj_types])
    ax.set_ylabel("Fraction of steps with over-prediction")
    ax.set_ylim(0, 1)
    ax.set_title("Over-prediction bias by material")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.8)
    path = os.path.join(out_dir, "fig4_overpred_bias.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary table and CSV export
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(merged: pd.DataFrame, corr: dict):
    """Print a LaTeX-ready summary table to stdout."""
    print("\n" + "=" * 72)
    print("  SUMMARY TABLE  (paste into paper)")
    print("=" * 72)
    print(f"{'Object type':<14} {'k_mean':>9} {'k_std':>8} "
          f"{'MAE (N)':>9} {'RMSE':>8} {'MRE %':>8} {'n':>4}")
    print("-" * 72)
    for obj_type, grp in merged.groupby("obj_type", sort=False):
        print(f"{obj_type.capitalize():<14} "
              f"{grp['k_mean'].mean():>9.1f} "
              f"{grp['k_std'].mean():>8.1f} "
              f"{grp['mae'].mean():>9.4f} "
              f"{grp['rmse'].mean():>8.4f} "
              f"{grp['mre'].mean()*100:>8.1f} "
              f"{len(grp):>4d}")
    print("=" * 72)
    print(f"\nCorrelation (k_mean vs MAE):")
    print(f"  Pearson  r = {corr['pearson_r']:+.4f}  "
          f"p = {corr['pearson_p']:.4f}")
    print(f"  Spearman r = {corr['spearman_r']:+.4f}  "
          f"p = {corr['spearman_p']:.4f}")
    sig = "significant" if corr["pearson_p"] < 0.05 else "not significant"
    print(f"  Interpretation: correlation is {sig} at α=0.05")
    if corr["pearson_r"] > 0.4:
        print("  ✓ Positive correlation — higher stiffness → higher force error")
        print("    This is your key finding.")
    elif corr["pearson_r"] < -0.4:
        print("  Negative correlation — softer objects harder to control?")
    else:
        print("  Weak correlation — may need more data points or "
              "additional confounding factors.")

    print("\nLaTeX table row format:")
    for obj_type, grp in merged.groupby("obj_type"):
        print(f"  {obj_type.capitalize():12s} & "
              f"{grp['k_mean'].mean():.1f} & "
              f"{grp['mae'].mean():.4f} $\\pm$ "
              f"{grp['mae'].std():.4f} \\\\")


def save_csv(merged: pd.DataFrame, out_dir: str):
    cols = ["case_name", "obj_type",
            "k_mean", "k_std", "k_median", "n_springs",
            "mae", "rmse", "mre", "n_steps",
            "gt_mean", "pred_mean", "over_pred_frac"]
    cols = [c for c in cols if c in merged.columns]
    path = os.path.join(out_dir, "stiffness_force_summary.csv")
    merged[cols].sort_values("k_mean").to_csv(path, index=False, float_format="%.4f")
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Utility: inspect what's inside a params.pkl
# ─────────────────────────────────────────────────────────────────────────────

def inspect_pkl(pkl_path: str):
    """
    Helper to understand the structure of a params.pkl before running
    the full analysis.  Call this once to verify key names.

    Usage:
        python stiffness_force_analysis.py --inspect /path/to/params.pkl
    """
    params = load_params_pkl(pkl_path)
    if params is None:
        print("Failed to load file.")
        return

    def _describe(obj, prefix="", depth=0):
        if depth > 3:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                _describe(v, f"{prefix}.{k}" if prefix else str(k), depth+1)
        elif isinstance(obj, np.ndarray):
            print(f"  {prefix:50s}  ndarray {obj.shape}  "
                  f"dtype={obj.dtype}  "
                  f"min={obj.min():.4f}  max={obj.max():.4f}")
        elif isinstance(obj, (int, float)):
            print(f"  {prefix:50s}  scalar  {obj:.4f}")
        elif isinstance(obj, list) and len(obj) > 0:
            print(f"  {prefix:50s}  list[{len(obj)}]  "
                  f"type={type(obj[0]).__name__}")
        elif hasattr(obj, "shape"):       # torch tensor etc.
            import torch
            if isinstance(obj, torch.Tensor):
                arr = obj.detach().cpu().numpy()
                print(f"  {prefix:50s}  Tensor {arr.shape}  "
                      f"min={arr.min():.4f}  max={arr.max():.4f}")
        else:
            print(f"  {prefix:50s}  {type(obj).__name__}")

    print(f"\nContents of {pkl_path}:")
    _describe(params)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Correlate PhysTwin spring stiffness with force "
                    "prediction error across object materials.")
    parser.add_argument("--phystwin_root", required=False,
                        default=".",
                        help="Root directory of the PhysTwin repo "
                             "(contains experiments/ folder)")
    parser.add_argument("--force_results", required=False,
                        default=None,
                        help="Path to Malak's force prediction CSV")
    parser.add_argument("--output_dir", default="./analysis_results",
                        help="Where to save figures and CSV")
    parser.add_argument("--inspect", default=None,
                        help="Inspect a single params.pkl and exit")
    args = parser.parse_args()

    # ── inspection mode ───────────────────────────────────────────────────
    if args.inspect:
        inspect_pkl(args.inspect)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: stiffness extraction ─────────────────────────────────────
    print("\n[1/4] Extracting spring stiffness from PhysTwin outputs...")
    stiff_df = extract_stiffness_stats(args.phystwin_root)
    stiff_csv = os.path.join(args.output_dir, "stiffness_per_case.csv")
    stiff_df.to_csv(stiff_csv, index=False, float_format="%.4f")
    print(f"  Stiffness table saved → {stiff_csv}")
    print(f"  Cases found: {len(stiff_df)}  |  "
          f"Types: {stiff_df['obj_type'].value_counts().to_dict()}")

    # ── Step 2: load force results ────────────────────────────────────────
    if args.force_results is None:
        print("\n[2/4] No --force_results provided.")
        print("       Stiffness extraction complete.  Pass Malak's CSV to "
              "continue.\n")
        print("       Example usage:")
        print("         python stiffness_force_analysis.py \\")
        print("           --phystwin_root /path/to/PhysTwin \\")
        print(f"           --force_results malak_predictions.csv \\")
        print(f"           --output_dir {args.output_dir}\n")
        return

    print(f"\n[2/4] Loading force predictions from {args.force_results}...")
    force_df  = load_force_results(args.force_results)
    force_agg = aggregate_force_per_case(force_df)
    print(f"  Episodes: {len(force_agg)}  |  "
          f"Steps total: {int(force_df['abs_error'].count())}")

    # ── Step 3: merge and correlate ───────────────────────────────────────
    print("\n[3/4] Merging and computing correlations...")
    merged, corr = merge_and_correlate(stiff_df, force_agg)
    print_summary_table(merged, corr)
    save_csv(merged, args.output_dir)

    # ── Step 4: figures ───────────────────────────────────────────────────
    print("\n[4/4] Generating figures...")
    plot_correlation(merged, corr, args.output_dir)
    plot_per_material_bar(merged, args.output_dir)
    if len(merged) >= 4:       # violin needs enough data
        plot_violin(merged, args.output_dir)
    plot_error_breakdown(merged, args.output_dir)

    print(f"\n✅  All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
