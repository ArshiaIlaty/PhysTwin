#!/bin/bash
#SBATCH --job-name=phystwin-smoke
#SBATCH --account=mgamalel
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --chdir=/share/crsp/lab/selmalak/mgamalel/PhysTwin/phystwin_src
#SBATCH --output=my_work/results/slurm_logs/smoke_%j.out
#SBATCH --error=my_work/results/slurm_logs/smoke_%j.err

set -e
mkdir -p my_work/results/slurm_logs my_work/results/dataset_smoke

# Module init
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi
module load cuda/12.2.0

# Activate env via direct PATH prepend (conda activate stacks badly under .bashrc-activated base).
ENV=/pub/mgamalel/envs/phystwin
export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1   # don't pick up user-site packages
export CONDA_PREFIX="$ENV"
unset PYTHONPATH

echo "[$(date)] node=$SLURMD_NODENAME pwd=$(pwd)"
nvidia-smi -L
which python && python --version
python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
python -c "import warp; print('warp=', warp.__version__)"
python -c "from qqtt import InvPhyTrainerWarp; print('qqtt import OK')"

CASE=${CASE:-single_push_rope_1}
NCTRL=${NCTRL:-1}

echo "=== Smoke-test: extract_dataset on $CASE (no rendering) ==="
python my_work/code/extract_dataset.py --case_name "$CASE" --n_ctrl_parts "$NCTRL" --cfg_type real --out_dir my_work/results/dataset_smoke
ls -la my_work/results/dataset_smoke/
python my_work/code/inspect_data.py --dataset_dir my_work/results/dataset_smoke

echo "[$(date)] DONE"
