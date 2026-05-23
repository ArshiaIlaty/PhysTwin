# Lessons learned — closed-loop control extension

Append a dated entry every time the user corrects an approach OR confirms a
non-obvious decision worked. Lead with the rule, then **Why** and **How to
apply**.

For the upstream project's lessons see [../tasks/lessons.md](../tasks/lessons.md).
SLURM defaults and HPC environment notes from that file still apply
(`--partition=free-gpu --account=mgamalel`, `/pub/mgamalel/envs/phystwin`,
build flags for CUDA submodules, etc.) — don't re-learn them here.

---

## (template)
## YYYY-MM-DD — short rule name
- **Rule**: …
- **Why**: …
- **How to apply**: …

## 2026-05-23 — In the dataset npz files, `material` is cfg_type, not the material taxonomy
- **Rule**: When reading per-trajectory npz files from `results/dataset_v2/`
  or `results/dataset_synth_raw/`, the field that holds the rope/cloth/sloth/
  toy taxonomy is **`object_category`**, not `material`. The `material` field
  holds PhysTwin's cfg_type, which is only `"real"` or `"cloth"` (used to
  select `configs/real.yaml` vs `configs/cloth.yaml` at extraction time).
- **Why**: `extract_dataset.py` saves `material=cfg_type` and
  `object_category=_object_category(case_name)`. Same convention in
  `generate_synthetic.py`. The names are misleading. First run of
  `build_policy_dataset.py` produced `per material: {cloth, real}` which is
  meaningless for a per-material scaler / per-material training.
- **How to apply**: New scripts should read `object_category` (with a
  fallback to `material`). Audit existing scripts before re-running — the
  v3.1 trainer reportedly does the right thing, but anything new should
  not assume `material` is what its name suggests.
