#!/usr/bin/env python3
"""
train_policy_v2.py — train the Fix I transformer policy (policy_v2.py).

Same gate structure as train_policy.py:
  --check_only      G1: delegate to policy_v2 self-test
  --overfit_tiny    G2: overfit 100 windows
  (default)         G3: full train (--seed) or multi (--seeds 1,2,3)

Inputs:  my_work/results/policy_dataset_fixI.npz
Outputs: my_work/results/models_policy_v2/seed_*/
         {policy.pt, arch.json, scalers_v2.pkl, target_scalers.pkl,
          metrics.json, train_log.json}
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from policy_v2 import (  # noqa: E402
    DEFAULT_ARCH, PolicyTransformerV2, build_window_index, compute_cmd_scale,
    compute_prev_action, fit_scalers_v2, gather_windows, load_dataset_v2,
    save_arch, _self_test,
)
from train_policy import (  # noqa: E402
    aggregate, fit_target_scalers, masked_mse, r2_scalar, r2_vec,
    random_block_split, scale_targets, unscale_predictions,
)

logger = logging.getLogger("train_policy_v2")

RESULTS = SCRIPT_DIR.parent / "results"
DEFAULT_DATA = RESULTS / "policy_dataset_fixI.npz"
DEFAULT_OUT = RESULTS / "models_policy_v2"


def prepare(data, val_frac, seed, drop_prev_action=False,
            material_key="raw_stiffness"):
    """Split + window index + scalers. Returns everything training needs.

    drop_prev_action=True zeroes the prev_action stream entirely (v2.1
    ablation: if OOD ramps recover without it, the exposure-bias feedback
    loop attribution from policy_v2_review.md is confirmed).
    """
    win = build_window_index(data["case_name"], data["frame_idx"])
    prev_action = compute_prev_action(data["action"], win["prev_idx"])
    if drop_prev_action:
        prev_action[:] = 0.0
    cmd_scale = compute_cmd_scale(
        data["force_goal"], data["action_mask"], data["case_name"])
    val_mask = random_block_split(data["case_name"], val_frac, seed)
    train_rows = np.where(~val_mask)[0]
    val_rows = np.where(val_mask)[0]
    scalers = fit_scalers_v2(data, prev_action, cmd_scale, train_rows,
                             material_key)
    target_scalers = fit_target_scalers(
        data["action"][train_rows], data["material"][train_rows],
        data["action_mask"][train_rows])
    y_scaled = scale_targets(data["action"], data["material"], target_scalers)
    return (win, prev_action, cmd_scale, scalers, target_scalers, y_scaled,
            train_rows, val_rows)


def batches(rows, batch_size, rng=None):
    rows = rng.permutation(rows) if rng is not None else rows
    for i in range(0, len(rows), batch_size):
        yield rows[i:i + batch_size]


def forward_rows(model, data, prev_action, cmd_scale, win, scalers, rows,
                 device, batch_size=1024, material_key="raw_stiffness"):
    """Predict (scaled) actions for `rows` in eval mode."""
    model.eval()
    outs = []
    with torch.no_grad():
        for b in batches(rows, batch_size):
            past, goal, cond = gather_windows(
                data, prev_action, cmd_scale, win, scalers, b, material_key)
            out = model(torch.from_numpy(past).to(device),
                        torch.from_numpy(goal).to(device),
                        torch.from_numpy(cond).to(device))
            outs.append(out.cpu().numpy())
    return np.concatenate(outs, axis=0)


def evaluate_val(model, data, prev_action, cmd_scale, win, scalers,
                 target_scalers, val_rows, device,
                 material_key="raw_stiffness"):
    """Per-material per-group metrics on val rows (unscaled actions)."""
    pred_scaled = forward_rows(model, data, prev_action, cmd_scale, win,
                               scalers, val_rows, device,
                               material_key=material_key)
    pred = unscale_predictions(pred_scaled, data["material"][val_rows],
                               target_scalers)
    out = {}
    val_mat = data["material"][val_rows]
    val_mask = data["action_mask"][val_rows]
    val_act = data["action"][val_rows]
    for m in sorted(set(val_mat.tolist())):
        m_rows = val_mat == m
        out[m] = {}
        for g in range(2):
            active = val_mask[m_rows, g] > 0.5
            n = int(active.sum())
            if n < 5:
                continue
            true_g = val_act[m_rows][active, g, :]
            pred_g = pred[m_rows][active, g, :]
            true_mag = np.linalg.norm(true_g, axis=1)
            pred_mag = np.linalg.norm(pred_g, axis=1)
            nz = true_mag > 1e-6
            dir_cos = float(
                ((pred_g[nz] * true_g[nz]).sum(axis=1)
                 / (pred_mag[nz] * true_mag[nz] + 1e-12)).mean()
            ) if nz.any() else float("nan")
            out[m][f"group_{g}"] = {
                "n_rows": n,
                "mag_r2": r2_scalar(true_mag, pred_mag),
                "dir_cosine": dir_cos,
                "vec_r2": r2_vec(true_g, pred_g),
            }
    return out


def run_overfit(args, device):
    mk = material_key_of(args)
    data = load_dataset_v2(args.data, args.exclude_materials)
    (win, prev_action, cmd_scale, scalers, target_scalers, y_scaled,
     train_rows, _) = prepare(data, args.val_frac, args.seed,
                              material_key=mk)
    rng = np.random.RandomState(args.seed)
    rows = rng.choice(train_rows, size=100, replace=False)

    torch.manual_seed(args.seed)
    model = PolicyTransformerV2(**make_arch(args)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    past, goal, cond = gather_windows(
        data, prev_action, cmd_scale, win, scalers, rows, mk)
    Xp = torch.from_numpy(past).to(device)
    Xg = torch.from_numpy(goal).to(device)
    Xc = torch.from_numpy(cond).to(device)
    Yt = torch.from_numpy(y_scaled[rows]).float().to(device)
    Mt = torch.from_numpy(data["action_mask"][rows]).float().to(device)

    final_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = masked_mse(model(Xp, Xg, Xc), Yt, Mt)
        loss.backward()
        opt.step()
        if epoch % 100 == 0 or epoch == args.epochs - 1:
            logger.info("G2 epoch %d  loss=%.6f", epoch, loss.item())
        final_loss = loss.item()
    passed = final_loss < 1e-3
    print(f"\n=== G2 tiny overfit: {'PASS' if passed else 'FAIL'} "
          f"(final loss={final_loss:.6f}) ===")
    if not passed:
        sys.exit(1)


def make_arch(args):
    arch = dict(DEFAULT_ARCH)
    arch.update({"d_model": args.d_model, "n_layers": args.n_layers,
                 "n_heads": args.n_heads, "latent_dim": args.latent_dim,
                 # v2.1: rollout closure must mirror the prev_action regime
                 "use_prev_action": not args.drop_prev_action,
                 "prev_noise": args.prev_noise})
    if args.material_stats:
        # Fix K: enriched 11-D material vector (+ adapter token by default).
        arch.update({"stiff_dim": 11, "cond_dim": 12, "material_token": True})
    if args.material_token_only:
        arch.update({"material_token": True})
    if args.no_material_token:
        arch.update({"material_token": False})
    return arch


def material_key_of(args):
    return "raw_material" if args.material_stats else "raw_stiffness"


# Position of the prev_action slice inside the 44-D past token
# ([state31 | force6 | prev_action6 | valid1]).
PREV_SLICE = slice(37, 43)


def perturb_prev_action(past, sigma):
    """DART/scheduled-sampling-style robustness noise on the prev_action
    columns of a batch of past tokens (scaled space, ~unit variance, so
    sigma is in target-scaler units — the same units the policy's own
    closed-loop outputs occupy)."""
    noise = torch.randn_like(past[:, :, PREV_SLICE]) * sigma
    past = past.clone()
    past[:, :, PREV_SLICE] += noise
    return past


def run_training(args, seeds, device):
    data = load_dataset_v2(args.data, args.exclude_materials)
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        mk = material_key_of(args)
        (win, prev_action, cmd_scale, scalers, target_scalers, y_scaled,
         train_rows, val_rows) = prepare(data, args.val_frac, seed,
                                         args.drop_prev_action, mk)
        logger.info("=== seed %d ===  train=%d val=%d device=%s "
                    "prev_noise=%.2f drop_prev=%s material_key=%s",
                    seed, len(train_rows), len(val_rows), device,
                    args.prev_noise, args.drop_prev_action, mk)

        model = PolicyTransformerV2(**make_arch(args)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        Yt = torch.from_numpy(y_scaled).float()
        Mt = torch.from_numpy(data["action_mask"]).float()

        best_val = float("inf")
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_left = args.patience
        log = []
        rng = np.random.RandomState(seed)
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for b in batches(train_rows, args.batch_size, rng):
                past, goal, cond = gather_windows(
                    data, prev_action, cmd_scale, win, scalers, b, mk)
                past_t = torch.from_numpy(past).to(device)
                if args.prev_noise > 0:
                    past_t = perturb_prev_action(past_t, args.prev_noise)
                opt.zero_grad()
                pred = model(past_t,
                             torch.from_numpy(goal).to(device),
                             torch.from_numpy(cond).to(device))
                loss = masked_mse(pred, Yt[b].to(device), Mt[b].to(device))
                loss.backward()
                opt.step()
                losses.append(loss.item())
            pred_v = forward_rows(model, data, prev_action, cmd_scale, win,
                                  scalers, val_rows, device, material_key=mk)
            val_loss = masked_mse(torch.from_numpy(pred_v), Yt[val_rows],
                                  Mt[val_rows]).item()
            log.append({"epoch": epoch, "train": float(np.mean(losses)),
                        "val": val_loss})
            if epoch % 10 == 0:
                logger.info("epoch %d  train=%.4f val=%.4f",
                            epoch, log[-1]["train"], val_loss)
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left == 0:
                    logger.info("early stop epoch %d  best_val=%.4f",
                                epoch, best_val)
                    break
        model.load_state_dict(best_state)
        model.to(device)

        metrics = evaluate_val(model, data, prev_action, cmd_scale, win,
                               scalers, target_scalers, val_rows, device, mk)
        sd_dir = args.out / f"seed_{seed}"
        sd_dir.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   sd_dir / "policy.pt")
        save_arch(sd_dir, make_arch(args))
        with open(sd_dir / "scalers_v2.pkl", "wb") as f:
            pickle.dump(scalers, f)
        with open(sd_dir / "target_scalers.pkl", "wb") as f:
            pickle.dump(target_scalers, f)
        with open(sd_dir / "metrics.json", "w") as f:
            json.dump({"metrics": metrics, "best_val_loss": float(best_val),
                       "args": {k: str(v) if isinstance(v, Path) else v
                                for k, v in vars(args).items()}}, f, indent=2)
        with open(sd_dir / "train_log.json", "w") as f:
            json.dump(log, f, indent=2)

        line = f"seed {seed}: "
        for m, gs in metrics.items():
            for g, mk in gs.items():
                line += f"{m}/{g} vec_r2={mk['vec_r2']:+.3f}  "
        logger.info(line)

    if len(seeds) > 1:
        summary = aggregate(args.out)
        with open(args.out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--exclude_materials", nargs="*", default=["toy"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--latent_dim", type=int, default=8)
    p.add_argument("--material_stats", action="store_true",
                   help="Fix K: condition on the enriched 11-D raw_material "
                        "vector (obj/ctrl-spring stiffness split + collision "
                        "elasticity/friction) and inject the material latent "
                        "as an extra sequence token (adapter-style).")
    p.add_argument("--no_material_token", action="store_true",
                   help="Fix K ablation: 11-D material via FiLM only.")
    p.add_argument("--material_token_only", action="store_true",
                   help="Fix K ablation: 5-D stiffness input but WITH the "
                        "adapter token (factorizes input-richness vs "
                        "injection-path).")
    p.add_argument("--prev_noise", type=float, default=0.0,
                   help="v2.1: sigma of Gaussian noise on prev_action during "
                        "training (scaled units). Targets exposure bias.")
    p.add_argument("--drop_prev_action", action="store_true",
                   help="v2.1 ablation: zero prev_action everywhere (train + "
                        "rollout, via arch.json use_prev_action=false).")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--check_only", action="store_true", help="G1")
    p.add_argument("--overfit_tiny", action="store_true", help="G2")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    if args.check_only:
        _self_test(args.data)
    elif args.overfit_tiny:
        run_overfit(args, device)
    else:
        seeds = ([int(s) for s in args.seeds.split(",")]
                 if args.seeds else [args.seed])
        run_training(args, seeds, device)


if __name__ == "__main__":
    main()
