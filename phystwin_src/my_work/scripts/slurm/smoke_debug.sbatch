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

mkdir -p my_work/results/slurm_logs
exec > >(tee -a "my_work/results/slurm_logs/smoke_${SLURM_JOB_ID}.out") 2> >(tee -a "my_work/results/slurm_logs/smoke_${SLURM_JOB_ID}.err" >&2)

echo "[$(date)] before set -e"
echo "PWD=$(pwd)  HOST=$(hostname)"
echo "SHELL=$SHELL  BASH_VERSION=$BASH_VERSION"

set -x
# Try sourcing module init explicitly
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi
if [ -f /opt/rcic/Modules/default/init/bash ]; then
  source /opt/rcic/Modules/default/init/bash
fi
type module || true
module load cuda/12.2.0 || echo "module load failed: $?"
# conda
if [ -f /data/homezvol0/mgamalel/anaconda3/etc/profile.d/conda.sh ]; then
  source /data/homezvol0/mgamalel/anaconda3/etc/profile.d/conda.sh
fi
conda activate /pub/mgamalel/envs/phystwin || echo "conda activate failed: $?"
which python && python --version
python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), torch.cuda.device_count())" || echo "torch import failed: $?"
python -c "import warp; print('warp=', warp.__version__)" || echo "warp import failed: $?"
python -c "from qqtt import InvPhyTrainerWarp; print('qqtt import OK')" || echo "qqtt import failed: $?"

set +x
echo "=== Smoke-test: extract_dataset on single_push_rope_1 ==="
python my_work/code/extract_dataset.py --case_name single_push_rope_1 --n_ctrl_parts 1 --cfg_type real --out_dir my_work/results/dataset_smoke
ls -la my_work/results/dataset_smoke/

echo "[$(date)] DONE"
