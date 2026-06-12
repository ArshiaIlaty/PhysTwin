#!/usr/bin/env python3
"""
analysis_loso.py — Experiment 2: held-out-material generalization.

Compares the 11D-FiLM policy that TRAINED ON sloth against the same recipe with
sloth fully held out (trained on rope+cloth only), evaluated on the sloth cases.
If the physics descriptor encodes real material properties, the held-out policy
should still control sloth by reading its stiffness — not collapse.

Reads:
  results/eval_closed_loop_v2_fixK_film/summary.json   (trained WITH sloth)
  results/eval_closed_loop_loso_sloth/summary.json     (sloth HELD OUT)
Writes:
  results/figures/ensemble/04_loso_sloth.png
  results/ensemble/loso_summary.json
"""
from __future__ import annotations
import json, logging
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("loso")
RESULTS = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS / "figures" / "ensemble"

def by_case(eval_dir):
    summ = json.load(open(RESULTS / eval_dir / "summary.json"))["per_rollout"]
    return {k: v for k, v in summ.items() if v["material"] == "sloth"}

trained = by_case("eval_closed_loop_v2_fixK_film")
heldout = by_case("eval_closed_loop_loso_sloth")

rows, labels = [], []
for k in sorted(heldout):
    prof = "ramp" if "ramp" in heldout[k]["profile"] else "replay"
    name = heldout[k]["case_name"] + f"\n{prof}"
    labels.append(name)
    rows.append((trained.get(k, {}).get("force_err_ratio", float("nan")),
                 heldout[k]["force_err_ratio"]))

t_vals = [r[0] for r in rows]; h_vals = [r[1] for r in rows]
log.info("%-40s %10s %10s", "case", "with_sloth", "held_out")
for lab, t, h in zip(labels, t_vals, h_vals):
    log.info("%-40s %10.3f %10.3f", lab.replace("\n", " "), t, h)
log.info("%-40s %10.3f %10.3f", "MEAN", np.nanmean(t_vals), np.nanmean(h_vals))

FIG_DIR.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(8.5, 4.8))
x = np.arange(len(labels)); w = 0.38
ax.bar(x - w/2, t_vals, w, label="FiLM-11D trained WITH sloth", color="#2ca02c")
ax.bar(x + w/2, h_vals, w, label="FiLM-11D sloth HELD OUT (rope+cloth only)", color="#d62728")
for i,(t,h) in enumerate(zip(t_vals,h_vals)):
    ax.text(i-w/2, t+0.02, f"{t:.2f}", ha="center", va="bottom", fontsize=8)
    ax.text(i+w/2, h+0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
ax.axhline(1.0, color="#888", ls=":", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("err_ratio (lower is better)")
ax.set_title("Generalization to a held-out material (sloth)\n"
             "Does the FiLM physics descriptor transfer to a material never seen in training?")
ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG_DIR / "04_loso_sloth.png", dpi=150)
log.info("wrote %s", FIG_DIR / "04_loso_sloth.png")

(RESULTS / "ensemble" / "loso_summary.json").write_text(json.dumps({
    "cases": {labels[i].replace("\n"," "): {"with_sloth": t_vals[i], "held_out": h_vals[i]}
              for i in range(len(labels))},
    "mean_with_sloth": float(np.nanmean(t_vals)),
    "mean_held_out": float(np.nanmean(h_vals)),
}, indent=2))
