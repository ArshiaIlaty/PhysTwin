#!/bin/bash
#SBATCH --job-name=phystwin-synth
#SBATCH --account=mgamalel
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --chdir=/share/crsp/lab/selmalak/mgamalel/PhysTwin/phystwin_src
#SBATCH --output=slurm/logs/synth_%j.out
#SBATCH --error=slurm/logs/synth_%j.err

set -e
mkdir -p slurm/logs dataset_synth_raw

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load cuda/12.2.0

ENV=/pub/mgamalel/envs/phystwin
export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export CONDA_PREFIX="$ENV"
unset PYTHONPATH

echo "[$(date)] node=$SLURMD_NODENAME"
nvidia-smi -L

python generate_synthetic.py \
  --materials rope,cloth,sloth,toy \
  --n_per_motion 6 \
  --T 120 \
  --amp_min 0.10 --amp_max 0.40 \
  --seed 0

echo "--- accepted synth files ---"
ls -la dataset_synth_raw/ | head -40

echo "[$(date)] DONE"
