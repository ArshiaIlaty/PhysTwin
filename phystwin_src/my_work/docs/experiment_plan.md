# Experiment Plan: Learning Force from Deformation via PhysTwin
**Project:** Predicting applied interaction force/wrench from observed object deformation  
**Approach:** Use PhysTwin as a physics-based data generator → train lightweight ML models on extracted (deformation, force) pairs  
**Deliverable:** Demo/presentation for robotics class

---

## Overview

PhysTwin estimates interaction forces by running differentiable physics optimization per video — a slow, per-case process. This project asks: **can a simple feedforward ML model learn the deformation → force mapping from PhysTwin's outputs, and does it generalize across material types?**

Your contribution is entirely in the layer built on top of PhysTwin. You do not retrain it.

---

## Stage 0: Environment Setup
**Goal:** Get PhysTwin running with pre-built data. No camera, no training.  
**Estimated time:** 2–4 hours (longer if CUDA env is painful)

### Steps

1. **Clone the repo**
   ```bash
   git clone https://github.com/Jianghanxiao/PhysTwin.git
   cd PhysTwin
   ```

2. **Set up the conda environment**
   ```bash
   export PATH={YOUR_DIR}/cuda/cuda-12.1/bin:$PATH
   export LD_LIBRARY_PATH={YOUR_DIR}/cuda/cuda-12.1/lib64:$LD_LIBRARY_PATH
   conda create -y -n phystwin python=3.10
   conda activate phystwin
   bash ./env_install/env_install.sh
   ```
   > ⚠️ If you hit dependency conflicts (Trellis vs diff-gaussian-rasterization), you only need the physics/inference part. You can skip installing Trellis, Grounding-SAM-2, RealSense, and SDXL per the README.

3. **Download pre-built data from HuggingFace**
   ```bash
   # Download all four required zips into the repo root
   wget https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/data.zip
   wget https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/experiments_optimization.zip
   wget https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/experiments.zip
   wget https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/gaussian_output.zip
   unzip data.zip && unzip experiments_optimization.zip && unzip experiments.zip && unzip gaussian_output.zip
   ```

4. **Smoke test — run the force visualizer on one case**
   ```bash
   python visualize_force.py --case_name single_push_rope_1 --n_ctrl_parts 1
   ```
   If you get a rendered video saved under `experiments/single_push_rope_1/`, Stage 0 is done.

### Checkpoint
- [ ] `visualize_force.py` runs successfully on at least one case
- [ ] Output video is saved to `experiments/`

---

## Stage 1: Understand the Data Structures
**Goal:** Know exactly what's available inside PhysTwin's inference outputs before writing any extraction code.  
**Estimated time:** 2–3 hours

### Steps

1. **Inspect `final_data.pkl` for one case**
   ```python
   import pickle
   with open("data/different_types/single_push_rope_1/final_data.pkl", "rb") as f:
       data = pickle.load(f)
   print(type(data))
   print(data.keys() if isinstance(data, dict) else dir(data))
   ```
   This file contains the processed tracking data — particle positions over time. Document what keys/fields are present.

2. **Inspect `optimal_params.pkl`**
   ```python
   with open("experiments_optimization/single_push_rope_1/optimal_params.pkl", "rb") as f:
       params = pickle.load(f)
   print(params)
   ```
   These are the optimized physics parameters (stiffness, damping, rest lengths). Not your primary ML target but useful context.

3. **Read `visualize_force.py` and trace into `trainer.visualize_force()`**
   - Open `qqtt/engine/trainer_warp.py` and find the `visualize_force` method
   - Identify where force tensors are computed and stored (look for variables named `ctrl_force`, `forces`, or similar)
   - Note: forces are likely Warp arrays or PyTorch tensors attached to control points at each timestep

4. **Run `script_inference.py` on one case and inspect what gets saved**
   ```bash
   python script_inference.py  # runs on all cases by default; edit to run just one
   ```
   Check what files appear under `experiments/{case_name}/` after inference — look for `.pkl`, `.npy`, or `.pt` files that may already contain per-timestep state.

5. **Document your findings** in a scratch notes file:
   - Shape of particle position array: `[T, N_particles, 3]`?
   - Shape of force array: `[T, N_ctrl_points, 3]`?
   - Where in the code are these computed?

### Checkpoint
- [ ] You know the shape and location of particle positions during inference
- [ ] You know the shape and location of force estimates during inference
- [ ] You've identified the exact lines in `trainer_warp.py` where forces are accessible

---

## Stage 2: Data Extraction
**Goal:** Write a script that hooks into PhysTwin's inference and saves paired `(deformation, force)` samples to disk as a clean numpy dataset.  
**Estimated time:** 3–5 hours

### Target cases
Run extraction on all available cases across three material types:

| Object Type | Cases |
|-------------|-------|
| Rope | `single_push_rope_1`, `single_push_rope_2`, `single_push_rope` |
| Cloth | `single_clift_cloth_1`, `double_lift_cloth_1`, `double_lift_cloth_2`, `double_lift_cloth_3` |
| Stuffed animal (sloth) | `double_stretch_sloth` |

Check `data_config.csv` in the repo root for the full list of available cases and their `n_ctrl_parts` values.

### Extraction script outline

Create `extract_dataset.py` at the repo root:

```python
"""
extract_dataset.py
Hooks into PhysTwin inference to extract per-timestep
(particle_displacement, ctrl_force) pairs and saves them as .npz files.
"""
import pickle, glob, os, json
import numpy as np
import torch
from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg

CASES = [
    ("single_push_rope_1",   1, "real"),
    ("single_push_rope_2",   1, "real"),
    ("single_clift_cloth_1", 1, "cloth"),
    ("double_lift_cloth_1",  2, "cloth"),
    ("double_stretch_sloth", 2, "real"),
    # add more as needed
]

def extract_case(case_name, n_ctrl_parts, cfg_type):
    cfg_file = "configs/cloth.yaml" if cfg_type == "cloth" else "configs/real.yaml"
    cfg.load_from_yaml(cfg_file)

    base_path = "./data/different_types"
    base_dir = f"./experiments/{case_name}"

    with open(f"./experiments_optimization/{case_name}/optimal_params.pkl", "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array([np.linalg.inv(c) for c in c2ws])
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]

    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )

    best_model_path = glob.glob(f"experiments/{case_name}/train/best_*.pth")[0]

    # --- HOOK: modify or call trainer method to return raw arrays ---
    # Option A: if trainer exposes positions/forces as return values, call directly
    # Option B: add a thin wrapper method to trainer_warp.py that saves arrays
    # See Stage 1 findings to decide which approach
    positions, forces = trainer.extract_force_data(best_model_path, n_ctrl_parts)
    # positions: [T, N_particles, 3]
    # forces:    [T, N_ctrl, 3]

    # Compute displacement from rest (frame 0)
    rest_pos = positions[0]  # [N_particles, 3]
    displacements = positions - rest_pos  # [T, N_particles, 3]

    # Flatten particles into feature vector
    T = displacements.shape[0]
    X = displacements.reshape(T, -1)  # [T, N_particles * 3]
    y = forces.reshape(T, -1)         # [T, N_ctrl * 3]

    os.makedirs("dataset", exist_ok=True)
    np.savez(
        f"dataset/{case_name}.npz",
        X=X, y=y,
        case_name=case_name,
        object_type=cfg_type,
    )
    print(f"Saved {case_name}: X={X.shape}, y={y.shape}")

if __name__ == "__main__":
    for case_name, n_ctrl, cfg_type in CASES:
        extract_case(case_name, n_ctrl, cfg_type)
```

> **Key implementation decision:** The method `trainer.extract_force_data()` doesn't exist yet — you'll need to either (a) add a thin method to `qqtt/engine/trainer_warp.py` that mirrors `visualize_force` but returns arrays instead of rendering, or (b) temporarily patch the existing `visualize_force` to also save `.npz` files as a side effect. Option (b) is lower risk and faster.

### Feature engineering notes
- **Don't flatten all particles blindly** — if there are 500+ particles, your feature vector will be 1500-dimensional with very few samples. Consider:
  - Using only contact-region particles (near the control points)
  - Subsampling to the N most-displaced particles
  - Using summary statistics (centroid displacement, bounding box deformation, max displacement)
- Decide this after Stage 1 when you know how many particles each case has

### Output format
```
dataset/
  single_push_rope_1.npz     # X: [T, D], y: [T, F]
  single_push_rope_2.npz
  single_clift_cloth_1.npz
  double_lift_cloth_1.npz
  double_stretch_sloth.npz
  ...
```

### Checkpoint
- [ ] `extract_dataset.py` runs without error on at least one case per material type
- [ ] `.npz` files saved with correct shapes
- [ ] You've verified that `y` values are non-trivial (non-zero, varying over time)

---

## Stage 3: Dataset Assembly & Inspection
**Goal:** Load all extracted cases into a unified dataset, inspect statistics, and sanity-check quality before training.  
**Estimated time:** 2–3 hours

### Steps

1. **Load and concatenate all cases**
   ```python
   import numpy as np, os, glob
   
   files = glob.glob("dataset/*.npz")
   all_X, all_y, all_labels = [], [], []
   object_type_map = {}
   
   for f in files:
       d = np.load(f, allow_pickle=True)
       all_X.append(d["X"])
       all_y.append(d["y"])
       case = str(d["case_name"])
       obj_type = str(d["object_type"])
       all_labels.extend([obj_type] * len(d["X"]))
       object_type_map[case] = obj_type
   
   X = np.concatenate(all_X, axis=0)
   y = np.concatenate(all_y, axis=0)
   labels = np.array(all_labels)
   print(f"Total samples: {X.shape[0]}, Features: {X.shape[1]}, Targets: {y.shape[1]}")
   ```

2. **Inspect force distribution per object type**
   ```python
   import matplotlib.pyplot as plt
   
   for obj_type in ["real", "cloth"]:  # adjust to your cfg_type labels
       mask = labels == obj_type
       force_mag = np.linalg.norm(y[mask], axis=1)
       plt.hist(force_mag, bins=30, alpha=0.6, label=obj_type)
   plt.xlabel("Force magnitude")
   plt.legend()
   plt.savefig("figures/force_distribution.png")
   ```

3. **Plot force over time for each case**
   For each `.npz` file, plot force magnitude vs. timestep. This is your first sanity check — does the force profile look physically plausible (zero at rest, peak at contact, decay)?

4. **Check for degenerate samples**
   - Are there timesteps where force is exactly zero? (pre/post contact frames) — consider trimming
   - Are particle displacements near zero for most timesteps? — confirms you have signal

5. **Normalize features and targets**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler_X = StandardScaler().fit(X)
   scaler_y = StandardScaler().fit(y)
   X_scaled = scaler_X.transform(X)
   y_scaled = scaler_y.transform(y)
   ```
   Save scalers with `pickle` — you'll need them at evaluation time.

### Checkpoint
- [ ] Total dataset has at least ~500 samples (a few hundred timesteps × multiple cases)
- [ ] Force distributions look physically reasonable
- [ ] Features and targets are normalized and saved

---

## Stage 4: Model Training
**Goal:** Train three models and compare performance within and across object types.  
**Estimated time:** 3–4 hours

### Train/test split strategy
**Important:** Do NOT do a random 80/20 split across timesteps from the same trajectory — that leaks temporal correlation. Instead, split by case:
- **Train cases:** 2 rope cases + 2 cloth cases + sloth
- **Test cases:** 1 held-out rope + 1 held-out cloth (use separate cases if available, otherwise hold out last 20% of each trajectory)

### Model 1: Linear Regression (baseline)
```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

model_lr = Ridge(alpha=1.0)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
print("Linear R²:", r2_score(y_test, y_pred_lr))
```

### Model 2: MLP
```python
import torch
import torch.nn as nn

class ForceMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )
    def forward(self, x):
        return self.net(x)

# Train with Adam, MSE loss, ~100 epochs
# Use early stopping on a small validation split within the train set
```

### Model 3: Per-object-type MLP
Train a separate MLP for each object type (rope, cloth, sloth) using only that type's data. Compare against the unified MLP — if per-type models are significantly better, that confirms material-specific force signatures.

### Metrics to record
For each model × object type combination:

| Model | Object Type | MSE ↓ | R² ↑ | Force Magnitude Error ↓ |
|-------|-------------|-------|------|--------------------------|
| Linear | Rope | | | |
| Linear | Cloth | | | |
| Linear | Sloth | | | |
| MLP (unified) | Rope | | | |
| MLP (unified) | Cloth | | | |
| MLP (unified) | Sloth | | | |
| MLP (per-type) | Rope | | | |
| MLP (per-type) | Cloth | | | |
| MLP (per-type) | Sloth | | | |

### Checkpoint
- [ ] All three models trained and evaluated
- [ ] Results table filled in
- [ ] MLP converges (training loss decreasing)

---

## Stage 5: Analysis & Visualization
**Goal:** Generate the figures and narrative for your demo presentation.  
**Estimated time:** 2–3 hours

### Key figures to produce

**Figure 1 — Force profile over time (per case)**
Plot ground truth force magnitude (from PhysTwin) vs. MLP prediction vs. linear prediction over time for one representative trajectory per object type. This is the most intuitive result to show.

**Figure 2 — R² by model × object type (bar chart)**
Side-by-side bars showing how well each model performs per material. The expected story: linear struggles with cloth (nonlinear), MLP does better, per-type MLP best.

**Figure 3 — Error distribution (box plots)**
Per-component force error (Fx, Fy, Fz) across object types. Shows whether the model is more accurate in certain force directions.

**Figure 4 — Qualitative overlay (optional, high impact)**
Overlay predicted force vectors onto the PhysTwin rendered video frames using `visualize_force.py` as a reference. Replace the ground truth force arrows with your model's predictions and compare.

### Demo narrative structure (5 min)
1. **Motivation (30s):** PhysTwin recovers forces from video via slow physics optimization. Can we make this fast and generalizable with ML?
2. **Method (1min):** PhysTwin as data generator → extract (deformation, force) pairs → train linear + MLP models
3. **Results (2.5min):** Walk through Figures 1–3. Highlight the cloth vs rope difference.
4. **Takeaway (1min):** Simple MLP captures force from deformation reasonably well for rope; cloth is harder (nonlinear, spatially distributed forces). Per-type models confirm material specificity.

### Checkpoint
- [ ] All figures generated and saved
- [ ] Demo narrative rehearsed end-to-end

---

## Fallback Plan (Plan B)

If Stage 1–2 extraction proves intractable (e.g., force tensors not easily accessible, major architectural barrier):

**Switch to PokeFlex dataset** — real-world paired (3D mesh deformation, force/torque) data from a robot arm poking 18 deformable objects. Download from: https://arxiv.org/abs/2409.17124

The ML pipeline (Stages 3–5) transfers directly. PhysTwin becomes the motivation in your framing rather than the data source: *"PhysTwin shows forces can be recovered from video; PokeFlex gives us real-world ground truth to train a fast learned predictor."*

Trigger Plan B if: Stage 2 takes more than 8 hours without a working extraction script.

---

## Timeline Summary

| Stage | Task | Est. Time | Cumulative |
|-------|------|-----------|------------|
| 0 | Environment setup | 2–4 hrs | 4 hrs |
| 1 | Understand data structures | 2–3 hrs | 7 hrs |
| 2 | Data extraction | 3–5 hrs | 12 hrs |
| 3 | Dataset assembly & inspection | 2–3 hrs | 15 hrs |
| 4 | Model training | 3–4 hrs | 19 hrs |
| 5 | Analysis & visualization | 2–3 hrs | 22 hrs |

Total estimate: **~20–22 hours** of focused work.

---

## File Structure (End State)

```
PhysTwin/
├── extract_dataset.py          ← NEW: your extraction script
├── train_models.py             ← NEW: linear + MLP training
├── evaluate_models.py          ← NEW: metrics + figures
├── dataset/
│   ├── single_push_rope_1.npz
│   ├── single_clift_cloth_1.npz
│   ├── double_stretch_sloth.npz
│   └── ...
├── figures/
│   ├── force_over_time.png
│   ├── r2_by_model.png
│   └── error_distribution.png
└── models/
    ├── mlp_unified.pt
    ├── mlp_rope.pt
    ├── mlp_cloth.pt
    └── scalers.pkl
```
