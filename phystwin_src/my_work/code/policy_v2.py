"""policy_v2.py — transformer force-control policy ("Fix I").

Replaces the stateless 43-D MLP with a small transformer over frame tokens:

  tokens (k_past + k_goal = 12):
    past token j = Linear([state31 | force_now(6) | prev_action(6) | valid(1)])
    goal token j = Linear(F*(t+j).flatten(6))      # commanded future forces
  conditioning  c = [MaterialEncoder(stiffness_stats 5) -> latent(8) | cmd_scale]
                    applied as per-block FiLM (scale/shift after each FFN)
  readout: current-frame token -> LayerNorm -> Linear -> 6 (scaled Δctrl)

Design notes (see docs/closed_loop_control/policy_v2_plan.md):
  - history window targets the policy's statelessness; the goal preview
    targets release-timing myopia (the policy sees the falling edge coming);
  - cmd_scale = log mean ‖F*_commanded‖ is computed from the COMMANDED
    trajectory, so it is available at deployment by construction — it
    replaces Fix F's leaky recorded-force scale (numerically identical on
    replay, but honest);
  - FiLM conditioning (instead of input concat) targets the cross-material
    interference documented in fix_d_review.md.

Run as a script for the G1 window-assembly self-test:

    python my_work/code/policy_v2.py --data my_work/results/policy_dataset_fixI.npz
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

K_PAST = 8
K_GOAL = 4
FRAME_DIM = 31 + 6 + 6 + 1   # state31 | force_now | prev_action | valid
GOAL_DIM = 6
STIFF_DIM = 5
COND_DIM = STIFF_DIM + 1     # stiffness stats | cmd_scale

DEFAULT_ARCH = {
    "arch": "policy_transformer_v2",
    "k_past": K_PAST,
    "k_goal": K_GOAL,
    "frame_dim": FRAME_DIM,
    "goal_dim": GOAL_DIM,
    "cond_dim": COND_DIM,
    "stiff_dim": STIFF_DIM,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 2,
    "ffn_mult": 4,
    "latent_dim": 8,
    "enc_hidden": 32,
    "out_dim": 6,
    # Fix K: adapter-style injection — the conditioning embedding also enters
    # the sequence as one extra token the attention can read (in addition to
    # FiLM modulation). Off for pre-Fix-K checkpoints.
    "material_token": False,
}


# --------------------------- dataset / windows -----------------------------

def load_dataset_v2(npz_path, exclude_materials=("toy",)):
    """Load the fixI policy dataset with the fields window assembly needs.

    Mirrors train_policy.load_dataset's exclusion behavior but keeps
    frame_idx and skips the flat `features` concat (v2 builds windows
    instead). Requires raw_stiffness + frame_idx (rebuild the dataset with
    the updated build_policy_dataset.py if missing) and k_lookahead == 1
    everywhere (window math assumes consecutive frames).
    """
    d = np.load(Path(npz_path), allow_pickle=True)
    for f in ("raw_stiffness", "frame_idx"):
        if f not in d.files:
            raise KeyError(f"dataset missing '{f}' — rebuild with updated "
                           "build_policy_dataset.py")
    if "k_lookahead" in d.files and not np.all(d["k_lookahead"] == 1):
        raise ValueError("policy_v2 windows require k_lookahead == 1 rows only")

    material = np.array([str(m) for m in d["material"]])
    keep = np.ones(len(material), dtype=bool)
    for m in exclude_materials:
        keep &= material != m

    out = {
        "state":         d["state"][keep].astype(np.float32),
        "force_now":     d["force_now"][keep].astype(np.float32).reshape(-1, 6),
        "force_goal":    d["force_goal"][keep].astype(np.float32).reshape(-1, 6),
        "action":        d["action"][keep].astype(np.float32),          # [N,2,3]
        "action_mask":   d["action_mask"][keep].astype(np.float32),     # [N,2]
        "raw_stiffness": d["raw_stiffness"][keep].astype(np.float32),   # [N,5]
        "material":      material[keep],
        "case_name":     np.array([str(c) for c in d["case_name"]])[keep],
        "frame_idx":     d["frame_idx"][keep].astype(np.int64),
        "n_ctrl_parts":  d["n_ctrl_parts"][keep].astype(np.int8),
    }
    if "raw_material" in d.files:   # Fix K enriched 11-D vector
        out["raw_material"] = d["raw_material"][keep].astype(np.float32)
    return out


def build_window_index(case_name, frame_idx, k_past=K_PAST, k_goal=K_GOAL):
    """Per-row gather indices for window assembly.

    Returns dict of arrays, all [N, ...] aligned with the dataset rows:
      past_idx  [N, k_past]  row index of each past slot (clamped to the
                             trajectory's first row; slot k_past-1 = row itself)
      valid     [N, k_past]  1.0 where the slot is a real (unclamped) frame
      goal_idx  [N, k_goal]  row index of force_goal for preview slots
                             (clamped to the trajectory's last row)
      prev_idx  [N]          row index of the previous frame, or -1 at start

    case_name is unique per trajectory (synthetic names carry __synth_*), so
    grouping by it can never mix trajectories.
    """
    N = len(case_name)
    past_idx = np.zeros((N, k_past), dtype=np.int64)
    valid = np.zeros((N, k_past), dtype=np.float32)
    goal_idx = np.zeros((N, k_goal), dtype=np.int64)
    prev_idx = np.full(N, -1, dtype=np.int64)

    for case in np.unique(case_name):
        rows = np.where(case_name == case)[0]
        rows = rows[np.argsort(frame_idx[rows])]
        T = len(rows)
        pos = np.arange(T)
        # past slot s (s = 0..k_past-1) refers to local position pos - (k_past-1-s)
        for s in range(k_past):
            local = pos - (k_past - 1 - s)
            valid_s = local >= 0
            past_idx[rows, s] = rows[np.clip(local, 0, T - 1)]
            valid[rows, s] = valid_s.astype(np.float32)
        for s in range(k_goal):
            goal_idx[rows, s] = rows[np.clip(pos + s, 0, T - 1)]
        prev_idx[rows[1:]] = rows[:-1]
    return {"past_idx": past_idx, "valid": valid,
            "goal_idx": goal_idx, "prev_idx": prev_idx}


def compute_prev_action(action, prev_idx):
    """[N,6] previous frame's action (zeros at trajectory start)."""
    pa = np.zeros((len(prev_idx), 6), dtype=np.float32)
    has = prev_idx >= 0
    pa[has] = action[prev_idx[has]].reshape(-1, 6)
    return pa


def compute_cmd_scale(force_goal, action_mask, case_name):
    """[N] per-trajectory log mean ‖F*‖ over active groups.

    At BC training time the commanded trajectory IS the recorded one; at
    rollout this is computed from the user-commanded F_goal array — honest
    by construction (see module docstring).
    """
    fg = force_goal.reshape(-1, 2, 3)
    mag = np.linalg.norm(fg, axis=-1)                  # [N,2]
    out = np.zeros(len(case_name), dtype=np.float32)
    for case in np.unique(case_name):
        rows = case_name == case
        m, msk = mag[rows], action_mask[rows]
        vals = m[msk > 0.5]
        out[rows] = np.log(float(vals.mean()) + 1e-6) if len(vals) else 0.0
    return out


def cmd_scale_from_goal_traj(F_goal, n_ctrl_parts):
    """Scalar log mean ‖F*‖ from a commanded [T,2,3] goal trajectory."""
    mag = np.linalg.norm(np.asarray(F_goal, dtype=np.float32)[:, :n_ctrl_parts], axis=-1)
    return float(np.log(float(mag.mean()) + 1e-6))


# --------------------------- scalers ---------------------------------------

def fit_scalers_v2(data, prev_action, cmd_scale, train_rows,
                   material_key="raw_stiffness"):
    """Three scalers: per-frame 43-D, goal 6-D, conditioning (1+material)-D."""
    def _fit(x):
        return {"mean": x.mean(axis=0).astype(np.float32),
                "std": (x.std(axis=0) + 1e-6).astype(np.float32)}
    frame43 = np.concatenate(
        [data["state"], data["force_now"], prev_action], axis=1)[train_rows]
    cond = np.concatenate(
        [data[material_key], cmd_scale[:, None]], axis=1)[train_rows]
    return {"frame": _fit(frame43),
            "goal": _fit(data["force_goal"][train_rows]),
            "cond": _fit(cond)}


def gather_windows(data, prev_action, cmd_scale, win, scalers, rows,
                   material_key="raw_stiffness"):
    """Assemble scaled model inputs for the given row indices.

    Returns (past [n,k_past,44], goal [n,k_goal,6], cond [n,cond_dim]) float32.
    """
    frame43 = np.concatenate(
        [data["state"], data["force_now"], prev_action], axis=1)
    frame43 = (frame43 - scalers["frame"]["mean"]) / scalers["frame"]["std"]
    goal6 = (data["force_goal"] - scalers["goal"]["mean"]) / scalers["goal"]["std"]
    cond6 = np.concatenate([data[material_key], cmd_scale[:, None]], axis=1)
    cond6 = (cond6 - scalers["cond"]["mean"]) / scalers["cond"]["std"]

    pi, gi = win["past_idx"][rows], win["goal_idx"][rows]
    past = np.concatenate(
        [frame43[pi], win["valid"][rows][..., None]], axis=-1)  # [n,k_past,44]
    goal = goal6[gi]                                            # [n,k_goal,6]
    return (past.astype(np.float32), goal.astype(np.float32),
            cond6[rows].astype(np.float32))


# --------------------------- model -----------------------------------------

class FiLMBlock(nn.Module):
    """Pre-LN transformer block with FiLM modulation after the FFN."""

    def __init__(self, d, n_heads, ffn_mult, cond_out):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))
        self.film = nn.Linear(cond_out, 2 * d)
        nn.init.zeros_(self.film.weight)   # identity modulation at init
        nn.init.zeros_(self.film.bias)

    def forward(self, x, c):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        gamma, beta = self.film(c).chunk(2, dim=-1)
        return x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class PolicyTransformerV2(nn.Module):
    def __init__(self, **arch):
        super().__init__()
        a = {**DEFAULT_ARCH, **arch}
        self.arch = a
        self.k_past, self.k_goal = a["k_past"], a["k_goal"]
        self.stiff_dim = int(a["stiff_dim"])
        self.use_material_token = bool(a.get("material_token", False))
        n_tokens = a["k_past"] + a["k_goal"] + (1 if self.use_material_token else 0)
        d = a["d_model"]
        self.proj_past = nn.Linear(a["frame_dim"], d)
        self.proj_goal = nn.Linear(a["goal_dim"], d)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens, d))
        self.material_encoder = nn.Sequential(
            nn.Linear(a["stiff_dim"], a["enc_hidden"]), nn.ReLU(),
            nn.Linear(a["enc_hidden"], a["latent_dim"]))
        cond_out = a["latent_dim"] + (a["cond_dim"] - a["stiff_dim"])
        if self.use_material_token:
            # Fix K adapter-style injection: the conditioning embedding enters
            # the sequence as a token, so attention can read the material
            # directly (complementing the multiplicative FiLM path).
            self.proj_material = nn.Linear(cond_out, d)
        self.blocks = nn.ModuleList([
            FiLMBlock(d, a["n_heads"], a["ffn_mult"], cond_out)
            for _ in range(a["n_layers"])])
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, a["out_dim"])

    def forward(self, past, goal, cond):
        """past [B,k_past,frame_dim], goal [B,k_goal,6], cond [B,cond_dim]."""
        c = torch.cat(
            [self.material_encoder(cond[:, :self.stiff_dim]),
             cond[:, self.stiff_dim:]], dim=1)
        toks = [self.proj_past(past), self.proj_goal(goal)]
        if self.use_material_token:
            toks.append(self.proj_material(c).unsqueeze(1))
        x = torch.cat(toks, dim=1) + self.pos_emb
        for blk in self.blocks:
            x = blk(x, c)
        return self.head(self.ln_out(x[:, self.k_past - 1]))


def save_arch(out_dir: Path, arch: dict):
    with open(Path(out_dir) / "arch.json", "w") as f:
        json.dump(arch, f, indent=2)


def load_policy_v2(policy_dir: Path):
    """Build + load a v2 model from a seed dir holding arch.json/policy.pt."""
    policy_dir = Path(policy_dir)
    with open(policy_dir / "arch.json") as f:
        arch = json.load(f)
    model = PolicyTransformerV2(**arch)
    model.load_state_dict(torch.load(policy_dir / "policy.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()
    return model, arch


# --------------------------- G1 self-test ----------------------------------

def _self_test(npz_path):
    data = load_dataset_v2(npz_path)
    N = len(data["state"])
    win = build_window_index(data["case_name"], data["frame_idx"])
    prev_action = compute_prev_action(data["action"], win["prev_idx"])
    cmd_scale = compute_cmd_scale(
        data["force_goal"], data["action_mask"], data["case_name"])

    cn, fi = data["case_name"], data["frame_idx"]
    rng = np.random.RandomState(0)
    sample = rng.choice(N, size=min(4000, N), replace=False)

    # 1. windows never mix trajectories
    for i in sample:
        assert (cn[win["past_idx"][i]] == cn[i]).all(), f"past mixes trajs at {i}"
        assert (cn[win["goal_idx"][i]] == cn[i]).all(), f"goal mixes trajs at {i}"
    print("PASS  no cross-trajectory leakage (past+goal), n=%d" % len(sample))

    # 2. past frame indices are the expected clamped sequence; valid correct
    for i in sample:
        t = fi[i]
        expect = np.clip(np.arange(t - K_PAST + 1, t + 1), 0, None)
        got = fi[win["past_idx"][i]]
        assert (got == expect).all(), f"past frames wrong at {i}: {got} vs {expect}"
        assert (win["valid"][i] == (np.arange(t - K_PAST + 1, t + 1) >= 0)).all()
    print("PASS  past window = clamped consecutive frames, valid mask correct")

    # 3. goal preview = frames t..t+3 clamped to trajectory end
    for i in sample:
        t, T = fi[i], fi[cn == cn[i]].max() + 1
        expect = np.clip(np.arange(t, t + K_GOAL), 0, T - 1)
        assert (fi[win["goal_idx"][i]] == expect).all(), f"goal preview wrong at {i}"
    print("PASS  goal preview = clamped t..t+%d" % (K_GOAL - 1))

    # 4. prev_action correctness
    for i in sample:
        if fi[i] == 0:
            assert (prev_action[i] == 0).all()
        else:
            j = win["prev_idx"][i]
            assert cn[j] == cn[i] and fi[j] == fi[i] - 1
            assert (prev_action[i] == data["action"][j].reshape(6)).all()
    print("PASS  prev_action = previous row's action (zeros at start)")

    # 5. cmd_scale finite + constant per trajectory
    assert np.isfinite(cmd_scale).all()
    for case in np.unique(cn)[:50]:
        v = cmd_scale[cn == case]
        assert np.allclose(v, v[0])
    print("PASS  cmd_scale finite, constant per trajectory")

    # 6. model forward shape on a real batch
    scalers = fit_scalers_v2(data, prev_action, cmd_scale, np.arange(N))
    past, goal, cond = gather_windows(
        data, prev_action, cmd_scale, win, scalers, sample[:64])
    model = PolicyTransformerV2()
    out = model(torch.from_numpy(past), torch.from_numpy(goal),
                torch.from_numpy(cond))
    n_params = sum(p.numel() for p in model.parameters())
    assert out.shape == (64, 6) and torch.isfinite(out).all()
    print(f"PASS  forward [64 windows] -> {tuple(out.shape)}, "
          f"{n_params / 1e3:.0f}K params")
    print("\n=== G1 window-assembly self-test: PASS ===")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=Path(__file__).resolve().parent.parent /
                    "results" / "policy_dataset_fixI.npz")
    _self_test(ap.parse_args().data)
