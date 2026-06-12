#!/bin/bash
#SBATCH --job-name=rl-fixG-train
#SBATCH --account=mgamalel
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/share/crsp/lab/selmalak/mgamalel/PhysTwin/phystwin_src
#SBATCH --output=my_work/results/slurm_logs/rl_fixG_train_%j.out
#SBATCH --error=my_work/results/slurm_logs/rl_fixG_train_%j.err

# Fix G — PPO training with stiffness-shaped reward, multi-material env.
# Same scaffold as rl_runs/fixF_multi (Fix F BC warm-start, 50K steps).
# Adds the action penalty and (optional) overshoot penalty terms.

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

BC_DIR=${BC_DIR:-my_work/results/models_policy_fixF/seed_1}
OUT=${OUT:-my_work/results/rl_runs/fixG_multi}
TOTAL_STEPS=${TOTAL_STEPS:-50000}
ALPHA=${ALPHA:-1000.0}   # set from smoke: ~3-7% of r_force at this scale
C=${C:-9.0}
BETA=${BETA:-0.0}
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
    --hidden 512 \
    --lr_actor 3e-5 \
    --lr_critic 3e-4 \
    --ent_coef 0.005 \
    --target_kl 0.02 \
    --critic_warmup_steps 5 \
    --init_log_std -2.0 \
    --action_penalty_alpha "$ALPHA" \
    --stiffness_normalizer "$C" \
    --overshoot_beta "$BETA" \
    --out "$OUT" \
    --seed "$SEED"

echo "[$(date)] DONE"
