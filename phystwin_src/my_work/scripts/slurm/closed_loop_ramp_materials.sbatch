#!/bin/bash
#SBATCH --job-name=phystwin-cloop-mat
#SBATCH --account=mgamalel
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --chdir=/share/crsp/lab/selmalak/mgamalel/PhysTwin/phystwin_src
#SBATCH --output=my_work/results/slurm_logs/cloop_mat_%j.out
#SBATCH --error=my_work/results/slurm_logs/cloop_mat_%j.err

set -e

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load cuda/12.2.0

ENV=/pub/mgamalel/envs/phystwin
export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export CONDA_PREFIX="$ENV"
unset PYTHONPATH

POLICY_DIR=my_work/results/models_policy/seed_1

# Test G5 ramp on cloth + sloth, both unshaped and shaped (500N).
# Question: does the policy release at all on a material where training
# data has a cleaner release-row signal (cloth 39.5% retract vs rope 17.3%)?
for CASE in double_lift_cloth_3 double_stretch_sloth; do
  for SHAPE in direct incremental; do
    echo "############################################################"
    echo "### $CASE | ramp | shaping=$SHAPE"
    echo "############################################################"
    python my_work/code/run_closed_loop.py \
      --case_name "$CASE" --profile policy_ramp \
      --policy_dir "$POLICY_DIR" --ramp_scale 1.0 \
      --goal_shaping "$SHAPE" --max_step_force 500
  done
done

echo "[$(date)] DONE"
