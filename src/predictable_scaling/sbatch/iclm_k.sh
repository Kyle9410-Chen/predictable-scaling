#!/bin/bash
#SBATCH --job-name=iclm-k
#SBATCH --partition=h200_256gpu
#SBATCH --account=ACD115198
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn006,25a-hgpn007,25a-hgpn009,25a-hgpn069,25a-hgpn071,25a-hgpn083,25a-hgpn141,25a-hgpn167
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out

# Redo the ICLM ordering with a different number of neighbours K, on the SAME kept documents as the k=10 run
# (keep_${TAU}.npy is reused, not recomputed), so only the neighbour graph changes and the `single` dataset
# stays valid. Needs the embeddings and keep file from iclm_proto.sh (N=all). Then build the datasets with
#   K=32 DATASETS="iclm randgroup" sbatch --dependency=afterok:<this job> src/predictable_scaling/sbatch/iclm_build.sh
#   K=32 TAU=0.94 sbatch src/predictable_scaling/sbatch/iclm_k.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load cuda/13.0

export HF_HOME=/work/${USER}/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

SRC=/work/${USER}/datasets/dolma3_mix_150B_scored.lance
VER=2
K=${K:-32}
TAU=${TAU:-0.94}
CAP=${CAP:-32768}
P=/work/${USER}/iclm/full

echo "host: $(hostname) | start: $(date) | K=$K TAU=$TAU CAP=$CAP"
run="uv run --frozen python -m predictable_scaling.iclm"
dist="uv run --frozen torchrun --nproc-per-node=8 -m predictable_scaling.iclm"
step() { echo "=== $1 | $(date +%T)"; }

step knn
$dist knn --emb "$P/emb" --out "$P/knn_k${K}" --k "$K"
step chains
[ -f "$P/chains_k${K}_${TAU}.parquet" ] || $run chains --doctable "$P/doctable.parquet" --knn "$P/knn_k${K}" \
    --keep "$P/keep_${TAU}.npy" --cap "$CAP" --out "$P/chains_k${K}_${TAU}.parquet"
step inspect
$run inspect --doctable "$P/doctable.parquet" --emb "$P/emb" --chains "$P/chains_k${K}_${TAU}.parquet" \
    --src "$SRC" --version $VER --out "$P/chains_k${K}_${TAU}.md"

echo "done: $(date)"
