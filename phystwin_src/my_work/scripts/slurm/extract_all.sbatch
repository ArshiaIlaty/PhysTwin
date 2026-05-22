#!/bin/bash
#SBATCH --job-name=phystwin-extract
#SBATCH --account=mgamalel
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --chdir=/share/crsp/lab/selmalak/mgamalel/PhysTwin/phystwin_src
#SBATCH --output=my_work/results/slurm_logs/extract_%j.out
#SBATCH --error=my_work/results/slurm_logs/extract_%j.err

set -e
mkdir -p my_work/results/slurm_logs my_work/results/dataset

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load cuda/12.2.0

ENV=/pub/mgamalel/envs/phystwin
export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export CONDA_PREFIX="$ENV"
unset PYTHONPATH

echo "[$(date)] node=$SLURMD_NODENAME"
nvidia-smi -L

python my_work/code/extract_dataset.py --out_dir my_work/results/dataset
echo "--- dataset listing ---"
ls -la my_work/results/dataset/

echo "[$(date)] DONE"
