#!/bin/bash
#SBATCH --job-name=phystwin-cloop-shape
#SBATCH --account=mgamalel
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --chdir=/share/crsp/lab/selmalak/mgamalel/PhysTwin/phystwin_src
#SBATCH --output=my_work/results/slurm_logs/cloop_shape_%j.out
#SBATCH --error=my_work/results/slurm_logs/cloop_shape_%j.err

set -e
mkdir -p my_work/results/slurm_logs my_work/results/closed_loop_rollouts

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load cuda/12.2.0

ENV=/pub/mgamalel/envs/phystwin
export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export CONDA_PREFIX="$ENV"
unset PYTHONPATH

echo "[$(date)] node=$SLURMD_NODENAME"
nvidia-smi -L

CASE_SINGLE=${CASE_SINGLE:-single_push_rope_1}
CASE_DOUBLE=${CASE_DOUBLE:-double_lift_cloth_3}
POLICY_DIR=${POLICY_DIR:-my_work/results/models_policy/seed_1}
# Step size sweep — 500 N (conservative), 1000 N (within rope p95 force change)
STEPS=${STEPS:-500 1000}

run () {
  echo
  echo "############################################################"
  echo "### $1"
  echo "############################################################"
  shift
  python my_work/code/run_closed_loop.py "$@"
}

for s in $STEPS; do
  run "G5 ramp + incremental shaping (max_step=${s}N) on $CASE_SINGLE" \
      --case_name "$CASE_SINGLE" --profile policy_ramp \
      --policy_dir "$POLICY_DIR" --ramp_scale 1.0 \
      --goal_shaping incremental --max_step_force "$s"
  run "G4 policy/recorded + incremental shaping (max_step=${s}N) on $CASE_SINGLE" \
      --case_name "$CASE_SINGLE" --profile policy_recorded_goal \
      --policy_dir "$POLICY_DIR" \
      --goal_shaping incremental --max_step_force "$s"
done

echo "[$(date)] DONE"
