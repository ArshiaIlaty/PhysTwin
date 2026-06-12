#!/bin/bash
#SBATCH --job-name=phystwin-eval-ol
#SBATCH --account=mgamalel
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --chdir=/share/crsp/lab/selmalak/mgamalel/PhysTwin/phystwin_src
#SBATCH --output=my_work/results/slurm_logs/eval_ol_%j.out
#SBATCH --error=my_work/results/slurm_logs/eval_ol_%j.err

set -e
mkdir -p my_work/results/slurm_logs my_work/results/eval_open_loop_fixF

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load cuda/12.2.0

ENV=/pub/mgamalel/envs/phystwin
export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export CONDA_PREFIX="$ENV"
unset PYTHONPATH

echo "[$(date)] node=$SLURMD_NODENAME"
nvidia-smi -L

# Open-loop ablation sweep: same Fix F architecture, force-feedback channel
# blanked. Requires the open-loop-trained checkpoint
# (models_policy_fixF_openloop, built with train_policy.py --zero_force_now).
echo "=== open-loop Fix F full sweep (14 rollouts) ==="
python my_work/code/eval_closed_loop.py \
  --policy_dir my_work/results/models_policy_fixF_openloop/seed_1 \
  --out        my_work/results/eval_open_loop_fixF \
  --open_loop

echo "[$(date)] DONE"
