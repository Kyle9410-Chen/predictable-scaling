#!/bin/bash
#SBATCH --job-name=scaling-baseline
#SBATCH --partition=8gpus
#SBATCH --account=ACD115198
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn007,25a-hgpn071
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load cuda/13.0

export HF_HOME=/work/${USER}/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=lo

uv sync --frozen

echo "host: $(hostname) | start: $(date)"
nvidia-smi

uv run --frozen --env-file .env torchrun \
    -m predictable_scaling.main \
    --max-tokens 100000

echo "done: $(date)"