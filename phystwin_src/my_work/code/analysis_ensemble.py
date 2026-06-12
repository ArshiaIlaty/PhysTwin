#!/usr/bin/env python3
"""
analysis_ensemble.py — Experiment 1: goal-aware policy ensemble.

We already have full 14-case closed-loop sweeps for several policies. Two of
them are complementary specialists:
  * the noise-injected transformer  — best in-distribution (replay) tracking
  * the 11-D FiLM physics policy     — best out-of-distribution (ramp) tracking

A single deployment policy has to compromise between the two. But at deployment
we always know our own *commanded* force goal F*(t), so we can route each
rollout to the right specialist BEFORE running it — no extra training, no
simulator calls. This script:

  1. loads the per-case err_ratio for every policy from its summary.json,
  2. computes a deployable routing feature from F*(t) alone (does the goal
     return to ~0 after its peak? = a release/ramp maneuver),
  3. builds two deployable selectors and a per-case oracle (upper bound),
  4. writes ensemble/ensemble_summary.json and figures/ensemble/01_ensemble_bars.png.

Pure numpy + matplotlib; runs on a login node.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ensemble")

SCRIPT_DIR = Path(__file__).resolve().parent
MY_WORK = SCRIPT_DIR.parent
RESULTS = MY_WORK / "results"
OUT_DIR = RESULTS / "ensemble"
FIG_DIR = RESULTS / "figures" / "ensemble"

# Policies to consider, with display names and the eval dir holding summary.json + npz.
POLICIES = {
    "BC+2D":             "eval_closed_loop_fixF",
    "RL+2D":             "eval_rl_fixF_multi",
    "Transformer-noise": "eval_closed_loop_v2_noise",
    "FiLM-11D":          "eval_closed_loop_v2_fixK_film",
}

GROUP_ORDER = [
    ("rope", "replay"), ("rope", "ramp"),
    ("cloth", "replay"), ("cloth", "ramp"),
    ("sloth", "replay"), ("sloth", "ramp"),
]

RAMP_TAIL_DROP_THRESH = 0.85  # goal falls >85% from its peak by the end -> release/ramp


def profile_bucket(profile: str) -> str:
    return "ramp" if "ramp" in profile else "replay"


def load_policy_table(eval_dir: str) -> dict:
    """case_key -> dict(material, profile, err_ratio)."""
    summ = json.load(open(RESULTS / eval_dir / "summary.json"))["per_rollout"]
    out = {}
    for k, v in summ.items():
        out[k] = {
            "material": v["material"],
            "profile": profile_bucket(v["profile"]),
            "err": v["force_err_ratio"],
        }
    return out


def goal_tail_drop(eval_dir: str, case_key: str) -> float:
    """Deployable routing feature computed from the commanded goal F*(t) only."""
    d = np.load(RESULTS / eval_dir / f"{case_key}.npz", allow_pickle=True)
    Fg = d["F_goal"]  # [T, K, 3]
    mag = np.linalg.norm(Fg.reshape(Fg.shape[0], -1), axis=1)
    if mag.max() < 1e-6:
        return 0.0
    peak = int(mag.argmax())
    return float((mag[peak] - mag[-1]) / (mag.max() + 1e-9))


def aggregate(per_case: dict) -> dict:
    """case->err  ->  {(material,profile): mean err}."""
    groups = defaultdict(list)
    for ck, err in per_case.items():
        groups[(META[ck]["material"], META[ck]["profile"])].append(err)
    return {g: float(np.mean(v)) for g, v in groups.items()}


# ---- load everything -------------------------------------------------------
tables = {name: load_policy_table(d) for name, d in POLICIES.items()}
CASES = sorted(next(iter(tables.values())).keys())
META = {ck: tables["RL+2D"][ck] for ck in CASES}  # material/profile per case

# routing feature (goal is identical across policies for a given case)
route_dir = POLICIES["Transformer-noise"]
ROUTE = {ck: goal_tail_drop(route_dir, ck) for ck in CASES}

logger.info("Routing check (tail_drop > %.2f => ramp specialist):", RAMP_TAIL_DROP_THRESH)
for ck in CASES:
    pred = "ramp" if ROUTE[ck] > RAMP_TAIL_DROP_THRESH else "replay"
    flag = "OK" if pred == META[ck]["profile"] else "MISROUTED"
    logger.info("  %-50s tail=%.2f -> %-6s (true %-6s) %s",
                ck, ROUTE[ck], pred, META[ck]["profile"], flag)

# ---- build selectors -------------------------------------------------------
# Selector A (goal-shape only, 2 policies): replay->Transformer-noise, ramp->FiLM-11D
# Selector B (goal-shape + material, 3 policies): cloth ramp stays with RL+2D,
#   because FiLM-11D regresses badly on cloth ramp (a measured exception).
sel_a, sel_b, oracle, oracle_pick = {}, {}, {}, {}
for ck in CASES:
    is_ramp = ROUTE[ck] > RAMP_TAIL_DROP_THRESH
    mat = META[ck]["material"]

    pa = "FiLM-11D" if is_ramp else "Transformer-noise"
    sel_a[ck] = (pa, tables[pa][ck]["err"])

    if is_ramp and mat == "cloth":
        pb = "RL+2D"
    elif is_ramp:
        pb = "FiLM-11D"
    else:
        pb = "Transformer-noise"
    sel_b[ck] = (pb, tables[pb][ck]["err"])

    best = min(POLICIES, key=lambda p: tables[p][ck]["err"])
    oracle[ck] = (best, tables[best][ck]["err"])
    oracle_pick[ck] = best

# ---- per-group aggregates --------------------------------------------------
group_tables = {name: aggregate({ck: t[ck]["err"] for ck in CASES})
                for name, t in tables.items()}
group_tables["Selector-A (goal)"]  = aggregate({ck: sel_a[ck][1] for ck in CASES})
group_tables["Selector-B (goal+mat)"] = aggregate({ck: sel_b[ck][1] for ck in CASES})
group_tables["Oracle (per-case best)"] = aggregate({ck: oracle[ck][1] for ck in CASES})

# overall mean across the 14 cases
overall = {}
for name, t in tables.items():
    overall[name] = float(np.mean([t[ck]["err"] for ck in CASES]))
overall["Selector-A (goal)"]  = float(np.mean([sel_a[ck][1] for ck in CASES]))
overall["Selector-B (goal+mat)"] = float(np.mean([sel_b[ck][1] for ck in CASES]))
overall["Oracle (per-case best)"] = float(np.mean([oracle[ck][1] for ck in CASES]))

# ---- print table -----------------------------------------------------------
cols = list(group_tables.keys())
logger.info("\n%-24s%s", "group", "".join(f"{c[:16]:>18}" for c in cols))
for g in GROUP_ORDER:
    logger.info("%-24s%s", f"{g[0]}_{g[1]}",
                "".join(f"{group_tables[c].get(g, float('nan')):>18.3f}" for c in cols))
logger.info("%-24s%s", "OVERALL (14-case)", "".join(f"{overall[c]:>18.3f}" for c in cols))

# ---- save JSON -------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
payload = {
    "routing_feature": "goal_tail_drop = (peak-final)/peak of ||F*(t)||",
    "ramp_threshold": RAMP_TAIL_DROP_THRESH,
    "per_group": {f"{g[0]}_{g[1]}": {c: group_tables[c].get(g) for c in cols}
                  for g in GROUP_ORDER},
    "overall_14case": overall,
    "selector_b_picks": {ck: sel_b[ck][0] for ck in CASES},
    "oracle_picks": oracle_pick,
}
(OUT_DIR / "ensemble_summary.json").write_text(json.dumps(payload, indent=2))
logger.info("\nwrote %s", OUT_DIR / "ensemble_summary.json")

# ---- figure: per-group bars for the key policies + ensemble + oracle -------
FIG_DIR.mkdir(parents=True, exist_ok=True)
plot_policies = ["RL+2D", "Transformer-noise", "FiLM-11D",
                 "Selector-B (goal+mat)", "Oracle (per-case best)"]
colors = ["#9aa7b8", "#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"]
labels = ["RL+2D (best single)", "Transformer-noise", "FiLM-11D",
          "Ensemble (deployable)", "Oracle (per-case best)"]

fig, ax = plt.subplots(figsize=(12, 5.2))
x = np.arange(len(GROUP_ORDER))
w = 0.16
for i, (pol, col, lab) in enumerate(zip(plot_policies, colors, labels)):
    vals = [group_tables[pol].get(g, np.nan) for g in GROUP_ORDER]
    bars = ax.bar(x + (i - 2) * w, vals, w, label=lab, color=col,
                  edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}",
                ha="center", va="bottom", fontsize=6.5, rotation=90)

ax.axhline(1.0, color="#d62728", lw=0.8, ls="--", alpha=0.6)
ax.text(len(GROUP_ORDER) - 0.5, 1.02, "err_ratio = 1 (uncontrolled)",
        color="#d62728", fontsize=7, ha="right", va="bottom")
ax.set_xticks(x)
ax.set_xticklabels([f"{g[0]}\n{g[1]}" for g in GROUP_ORDER])
ax.set_ylabel("err_ratio  (lower is better)")
ax.set_title("Goal-aware policy ensemble vs best single policies\n"
             "Routing on the commanded goal shape — no training, no extra simulator calls")
ax.legend(fontsize=8, ncol=3, loc="upper left", framealpha=0.9)
ax.set_ylim(0, max(4.2, 0.5))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "01_ensemble_bars.png", dpi=150)
logger.info("wrote %s", FIG_DIR / "01_ensemble_bars.png")
