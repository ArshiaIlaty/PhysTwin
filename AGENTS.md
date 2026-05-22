## Workflow Orchestration

### 1. Plan Mode Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy to keep main context window clean

- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop

- After ANY correction from the user: update 'tasks/lessons.md' with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done

- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests -> then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to 'tasks/todo.md' with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review to 'tasks/todo.md'
6. **Capture Lessons**: Update 'tasks/lessons.md' after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

## Project-specific rules (PhysTwin on UCI HPC3)

### Environment
- Cluster: UCI HPC3. Login nodes have NO GPU — never run training/inference there.
- SLURM submissions: always use `--partition=free-gpu --account=mgamalel`. Use `gpu:A100:1` or `gpu:V100:1` based on availability. Keep wall time conservative (default ≤2h unless justified).
- CUDA: load `module load cuda/12.2.0` (system has 11.4 / 11.7.1 / 12.2.0 / 13.0.1; plan asked for 12.1 — confirm Warp + torch work on 12.2 wheels before installing).
- Conda envs go under `/pub/mgamalel/envs/` (NOT home — home is small). Activate via `source ~/.bashrc && conda activate /pub/mgamalel/envs/phystwin`.

### Paths and storage
- Repo: `/share/crsp/lab/selmalak/mgamalel/PhysTwin` (this directory).
- Large data zips → `/share/crsp/lab/selmalak/mgamalel/PhysTwin/` (~hundreds of GB available; /tmp is only 222G and node-local).
- Scratch / intermediate outputs → `/pub/mgamalel/` if /share gets crowded.
- Never write to `/data/homezvol0/mgamalel/` (home, tight quota) for anything substantial.

### Inference and training
- Run all `python script_inference.py`, `python visualize_force.py`, and model training under `srun` or `sbatch` on `free-gpu`, not the login node.
- For interactive debugging: `srun -p free-gpu -A mgamalel --gres=gpu:1 --pty --time=1:00:00 bash`.

### Project-specific feature/ML decisions (locked 2026-05-20)
- ML target = **net wrench at control points** `[T, n_ctrl_parts, 3]`, NOT per-particle internal forces.
- Input features = **summary statistics over particles** (centroid disp, bbox deformation, max/mean per-axis disp, kinetic-energy-like terms) — fixed dim, works across cases with different N_particles.
- Data-extraction hook = add a non-rendering `extract_force_data()` method to `qqtt/engine/trainer_warp.py` rather than patching `visualize_force`.

### PhysTwin-specific gotchas
- `env_install/env_install.sh` installs Trellis, Grounding-SAM-2, RealSense, SDXL by default — these are NOT needed for inference-only. Read the script before running; skip the optional blocks.
- `n_ctrl_parts` varies per case (1 or 2). Always pass the right value — check `data_config.csv` in repo root.
- `final_data.pkl` contains tracked particle positions from preprocessing; `optimal_params.pkl` contains optimized physics params from a prior optimization run.
