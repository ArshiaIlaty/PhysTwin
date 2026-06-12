#!/bin/bash
#SBATCH --job-name=rl-v2-train
#SBATCH --account=mgamalel
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/share/crsp/lab/selmalak/mgamalel/PhysTwin/phystwin_src
#SBATCH --output=my_work/results/slurm_logs/rl_v2_train_%j.out
#SBATCH --error=my_work/results/slurm_logs/rl_v2_train_%j.err

# Policy v2 (Fix I) — PPO fine-tune of the transformer policy, multi-material.
# Warm-start = models_policy_v2_noise/seed_1 (BC + prev_action noise sigma=0.3,
# the best replay-profile BC; gate triggered 3/6 vs RL-FF-Multi).
# Same scaffold/case-mix as rl_runs/fixF_multi; no Fix-G action penalty
# (alpha=0 — it was a wash on the MLP and would confound the v2 comparison).
# Docs: my_work/docs/closed_loop_control/policy_v2_review.md (addendum).

set -e
mkdir -p my_work/results/slurm_logs my_work/results/rl_runs

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load cuda/12.2.0

ENV=/pub/mgamalel/envs/phystwin
export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export CONDA_PREFIX="$ENV"
unset PYTHONPATH

echo "[$(date)] node=$SLURMD_NODENAME"
nvidia-smi -L

BC_DIR=${BC_DIR:-my_work/results/models_policy_v2_noise/seed_1}
OUT=${OUT:-my_work/results/rl_runs/v2_multi}
TOTAL_STEPS=${TOTAL_STEPS:-50000}
SEED=${SEED:-0}

python my_work/code/rl_ppo.py \
    --bc_policy_dir "$BC_DIR" \
    --case_specs single_push_rope_1:policy_ramp \
                 double_lift_cloth_3:policy_ramp \
                 double_stretch_sloth:policy_ramp \
    --ramp_direction recorded_mean \
    --total_steps "$TOTAL_STEPS" \
    --n_steps_per_update 2048 \
    --batch_size 64 \
    --ppo_epochs 10 \
    --lr_actor 3e-5 \
    --lr_critic 3e-4 \
    --ent_coef 0.005 \
    --target_kl 0.02 \
    --critic_warmup_steps 5 \
    --init_log_std -2.0 \
    --out "$OUT" \
    --seed "$SEED"

echo "[$(date)] DONE"
