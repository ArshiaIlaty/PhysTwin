#!/bin/bash
#SBATCH --job-name=phystwin-extract
#SBATCH --account=mgamalel
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --chdir=/share/crsp/lab/selmalak/mgamalel/PhysTwin/phystwin_src
#SBATCH --output=slurm/logs/extract_%j.out
#SBATCH --error=slurm/logs/extract_%j.err

set -e
mkdir -p slurm/logs dataset

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load cuda/12.2.0

ENV=/pub/mgamalel/envs/phystwin
export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export CONDA_PREFIX="$ENV"
unset PYTHONPATH

echo "[$(date)] node=$SLURMD_NODENAME"
nvidia-smi -L

python extract_dataset.py --out_dir dataset
echo "--- dataset listing ---"
ls -la dataset/

echo "[$(date)] DONE"
