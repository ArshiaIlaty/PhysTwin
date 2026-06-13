"""Print a compact table from models_scenario/*/metrics.json."""
from __future__ import annotations

import argparse
import json
import os


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, required=True)
    args = parser.parse_args()
    rows: list[tuple[str, str, str, str, float, float]] = []
    for name in sorted(os.listdir(args.models_dir)):
        mpath = os.path.join(args.models_dir, name, "metrics.json")
        if not os.path.isfile(mpath):
            continue
        m = _load(mpath)
        target = m.get("target_key", "?")
        split = m.get("split", "?")
        hold = m.get("scenario_holdout", {})
        tag = name
        if split == "scenario_stiffness":
            tag += f" k_holdout={hold.get('stiffness')}"
        for model in ("ridge", "mlp_per_cat"):
            for cat, vals in m.get("results", {}).get(model, {}).items():
                rows.append((
                    tag, target, model, cat,
                    float(vals.get("r2", float("nan"))),
                    float(vals.get("ratio_mae", vals.get("force_mag_mae", float("nan")))),
                ))
    if not rows:
        print(f"No metrics.json under {args.models_dir}")
        return
    print(f"{'run':40s} {'target':12s} {'model':12s} {'cat':6s} {'R2':>8s} {'MAE':>8s}")
    print("-" * 92)
    for run, target, model, cat, r2, mae in rows:
        print(f"{run:40s} {target:12s} {model:12s} {cat:6s} {r2:8.3f} {mae:8.4f}")


if __name__ == "__main__":
    main()
