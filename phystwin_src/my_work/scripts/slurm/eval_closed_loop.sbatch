#!/bin/bash
#SBATCH --job-name=phystwin-eval
#SBATCH --account=mgamalel
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --chdir=/share/crsp/lab/selmalak/mgamalel/PhysTwin/phystwin_src
#SBATCH --output=my_work/results/slurm_logs/eval_%j.out
#SBATCH --error=my_work/results/slurm_logs/eval_%j.err

set -e
mkdir -p my_work/results/slurm_logs my_work/results/eval_closed_loop

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load cuda/12.2.0

ENV=/pub/mgamalel/envs/phystwin
export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export CONDA_PREFIX="$ENV"
unset PYTHONPATH

echo "[$(date)] node=$SLURMD_NODENAME"
nvidia-smi -L

# GATE selector: G1 | G2 | G3. Defaults to G3 (full sweep).
GATE=${GATE:-G3}

case "$GATE" in
  G1) echo "=== G1 single-case smoke ==="
      python my_work/code/eval_closed_loop.py \
        --cases single_push_rope_1 \
        --profiles policy_recorded_goal \
        --out my_work/results/eval_g1 ;;
  G2) echo "=== G2 per-material smoke ==="
      python my_work/code/eval_closed_loop.py \
        --cases single_push_rope_1 double_lift_cloth_3 double_stretch_sloth \
        --profiles policy_recorded_goal \
        --out my_work/results/eval_g2 ;;
  G3) echo "=== G3 full sweep (14 rollouts) ==="
      python my_work/code/eval_closed_loop.py ;;
  *) echo "unknown GATE: $GATE"; exit 1 ;;
esac

echo "[$(date)] DONE"
