#!/bin/bash
#SBATCH --job-name=phystwin-eval-ol-fin
#SBATCH --account=mgamalel
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --chdir=/share/crsp/lab/selmalak/mgamalel/PhysTwin/phystwin_src
#SBATCH --output=my_work/results/slurm_logs/eval_ol_fin_%j.out
#SBATCH --error=my_work/results/slurm_logs/eval_ol_fin_%j.err

set -e
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load cuda/12.2.0
ENV=/pub/mgamalel/envs/phystwin
export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export CONDA_PREFIX="$ENV"
unset PYTHONPATH

echo "[$(date)] node=$SLURMD_NODENAME"; nvidia-smi -L

POL=my_work/results/models_policy_fixF_openloop/seed_1
OUT=my_work/results/eval_open_loop_fixF

# 1) Run the one case the timed-out sweep missed.
echo "=== finishing missing case: double_stretch_sloth policy_ramp (open-loop) ==="
python my_work/code/run_closed_loop.py \
  --case_name double_stretch_sloth --profile policy_ramp \
  --policy_dir $POL --out_dir $OUT --open_loop --ramp_scale 1.0

# 2) Aggregate all 14 __openloop npzs into summary.json (no reruns).
echo "=== aggregating full open-loop summary ==="
python my_work/code/eval_closed_loop.py \
  --policy_dir $POL --out $OUT --open_loop --summary_only

echo "[$(date)] DONE"
