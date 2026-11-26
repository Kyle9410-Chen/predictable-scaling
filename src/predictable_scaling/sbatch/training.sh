#!/bin/bash
#SBATCH --job-name=scaling-baseline
#SBATCH --partition=8gpus
#SBATCH --account=ACD115198
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn006,25a-hgpn007,25a-hgpn009,25a-hgpn069,25a-hgpn071,25a-hgpn083,25a-hgpn141,25a-hgpn167
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=5:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load cuda/13.0

export HF_HOME=/work/${USER}/.cache/huggingface
export WANDB_DIR=/work/${USER}/wandb
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=lo
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Don't make too many request to s3
export LANCE_IO_THREADS=8

uv sync --frozen

echo "host: $(hostname) | start: $(date)"
nvidia-smi

ulimit -u "$(ulimit -Hu)"

uv run --frozen --env-file .env torchrun --nproc-per-node=8 \
    -m predictable_scaling.main \
    --max-tokens 1500000000 \
    --name warmup-flesch-dolma3mix-150M-1.5B \
    --tags 150M,warmup,flesch \
    --log-every 100 \
    --eval-every 500 \
    --batch-size 8 \
    --opt soap-muon \
    --adam_lr 1.2e-3 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --muon_lr 2.5e-3 \
    --soap_lr 3e-3 \
    --lr-scheduler powercool \
    --powercool-exp 2 \
    --wd 0.1 \
    --hp-config configs/hp/150M-soap-muon.yaml \
    --dataset "s3://flock-datasets/-projects/28/ben-kyle/dolma3_mix_150B_flesch_warmup_300M" \
    --dataset-version 3 \
    --save-dir "/work/${USER}/predictable-scaling/models" \
    "$@"

echo "done: $(date)"