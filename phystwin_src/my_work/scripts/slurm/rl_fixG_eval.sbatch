#!/bin/bash
#SBATCH --job-name=rl-fixG-eval
#SBATCH --account=mgamalel
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --chdir=/share/crsp/lab/selmalak/mgamalel/PhysTwin/phystwin_src
#SBATCH --output=my_work/results/slurm_logs/rl_fixG_eval_%j.out
#SBATCH --error=my_work/results/slurm_logs/rl_fixG_eval_%j.err

# Fix G eval — 14-case sweep on the trained Fix G policy.
set -e
mkdir -p my_work/results/slurm_logs

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load cuda/12.2.0

ENV=/pub/mgamalel/envs/phystwin
export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export CONDA_PREFIX="$ENV"
unset PYTHONPATH

echo "[$(date)] node=$SLURMD_NODENAME"
nvidia-smi -L

RUN_DIR=${RUN_DIR:-my_work/results/rl_runs/fixG_multi}
OUT=${OUT:-my_work/results/eval_rl_fixG_multi}
CHECKPOINT=${CHECKPOINT:-best.pt}

python my_work/code/rl_eval.py \
    --rl_run_dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --out "$OUT"

echo "[$(date)] DONE"
