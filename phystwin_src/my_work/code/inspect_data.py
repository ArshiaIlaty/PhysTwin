"""inspect_data.py
Stage-1 introspection helper. Reports shapes/keys for `final_data.pkl`,
`optimal_params.pkl`, and (if present) extracted dataset `.npz` files.

Run:
    python inspect_data.py --case_name single_push_rope_1
    python inspect_data.py --dataset_dir dataset
"""
from __future__ import annotations

import argparse
import json
import os
import pickle


def describe(obj, name="obj", depth=0):
    pad = "  " * depth
    if isinstance(obj, dict):
        print(f"{pad}{name}: dict({len(obj)} keys)")
        for k, v in obj.items():
            describe(v, name=str(k), depth=depth + 1)
    elif hasattr(obj, "shape"):
        try:
            sh = tuple(obj.shape)
        except Exception:
            sh = "?"
        dt = getattr(obj, "dtype", type(obj).__name__)
        print(f"{pad}{name}: shape={sh} dtype={dt}")
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__}(len={len(obj)})")
        if obj:
            describe(obj[0], name=f"{name}[0]", depth=depth + 1)
    else:
        print(f"{pad}{name}: {type(obj).__name__} = {obj!r:.80}")


def inspect_case(base_path: str, case_name: str) -> None:
    print(f"\n=== Inspecting case: {case_name} ===")
    fd = f"{base_path}/{case_name}/final_data.pkl"
    if os.path.exists(fd):
        print(f"\n[final_data.pkl] @ {fd}")
        with open(fd, "rb") as f:
            describe(pickle.load(f), name="final_data")
    else:
        print(f"  (missing) {fd}")

    op = f"./experiments_optimization/{case_name}/optimal_params.pkl"
    if os.path.exists(op):
        print(f"\n[optimal_params.pkl] @ {op}")
        with open(op, "rb") as f:
            describe(pickle.load(f), name="optimal_params")
    else:
        print(f"  (missing) {op}")

    md = f"{base_path}/{case_name}/metadata.json"
    if os.path.exists(md):
        print(f"\n[metadata.json] @ {md}")
        with open(md) as f:
            describe(json.load(f), name="metadata")
    else:
        print(f"  (missing) {md}")

    train = f"./experiments/{case_name}/train"
    if os.path.isdir(train):
        print(f"\n[experiments/{case_name}/train] contents:")
        for fn in sorted(os.listdir(train))[:30]:
            print(f"  {fn}")


def inspect_dataset(dataset_dir: str) -> None:
    import glob
    import numpy as np

    files = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))
    if not files:
        print(f"no .npz under {dataset_dir}")
        return
    for f in files:
        d = np.load(f, allow_pickle=True)
        case = str(d["case_name"])
        cat = str(d["object_category"])
        X = d["X"]
        y_per = d["y_per_ctrl"]
        y_net = d["y_net"]
        f_mag = np.linalg.norm(y_net, axis=1)
        print(
            f"{case} ({cat}): X={X.shape} y_per={y_per.shape} y_net={y_net.shape}  "
            f"|F| mean={f_mag.mean():.2f} max={f_mag.max():.2f}  nonzero={(f_mag>1e-3).mean():.2%}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    args = parser.parse_args()
    if args.case_name:
        inspect_case(args.base_path, args.case_name)
    if args.dataset_dir:
        print("\n=== Inspecting extracted dataset ===")
        inspect_dataset(args.dataset_dir)
    if not args.case_name and not args.dataset_dir:
        parser.error("pass --case_name and/or --dataset_dir")


if __name__ == "__main__":
    main()
