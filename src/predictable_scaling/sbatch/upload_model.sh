#!/bin/bash
#SBATCH --job-name=scaling-baseline
#SBATCH --partition=8gpus
#SBATCH --account=ACD115198
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn007,25a-hgpn071
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load cuda/13.0

export HF_HOME=/work/${USER}/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=lo
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

uv sync --frozen

source .venv/bin/activate

echo "host: $(hostname) | start: $(date)"
nvidia-smi

ulimit -u "$(ulimit -Hu)"

flock models push models/training-eval-smoke cerulean-labs/lab6-kyle-ben

echo "done: $(date)"