#!/bin/bash
#SBATCH --job-name=phystwin-cloop
#SBATCH --account=mgamalel
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --chdir=/share/crsp/lab/selmalak/mgamalel/PhysTwin/phystwin_src
#SBATCH --output=my_work/results/slurm_logs/cloop_%j.out
#SBATCH --error=my_work/results/slurm_logs/cloop_%j.err

set -e
mkdir -p my_work/results/slurm_logs my_work/results/closed_loop_rollouts

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi
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
GATES=${GATES:-G1 G2 G3 G4 G4b G5}

run_gate () {
  local label="$1"; shift
  echo
  echo "############################################################"
  echo "### $label"
  echo "### $*"
  echo "############################################################"
  python my_work/code/run_closed_loop.py "$@"
}

for g in $GATES; do
  case "$g" in
    G1)  run_gate "G1 random      ($CASE_SINGLE)" \
            --case_name "$CASE_SINGLE" --profile random ;;
    G2)  run_gate "G2 zero        ($CASE_SINGLE)" \
            --case_name "$CASE_SINGLE" --profile zero ;;
    G3)  run_gate "G3 replay      ($CASE_SINGLE)" \
            --case_name "$CASE_SINGLE" --profile replay_action ;;
    G4)  run_gate "G4 policy/rec  ($CASE_SINGLE)" \
            --case_name "$CASE_SINGLE" --profile policy_recorded_goal \
            --policy_dir "$POLICY_DIR" ;;
    G4b) run_gate "G4b policy/rec ($CASE_DOUBLE)" \
            --case_name "$CASE_DOUBLE" --profile policy_recorded_goal \
            --policy_dir "$POLICY_DIR" ;;
    G5)  run_gate "G5 ramp        ($CASE_SINGLE)" \
            --case_name "$CASE_SINGLE" --profile policy_ramp \
            --policy_dir "$POLICY_DIR" --ramp_scale 1.0 ;;
    *) echo "unknown gate: $g"; exit 1 ;;
  esac
done

echo "[$(date)] DONE"
