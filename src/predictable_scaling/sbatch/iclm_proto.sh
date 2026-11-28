#!/bin/bash
#SBATCH --job-name=iclm-proto
#SBATCH --partition=8gpus
#SBATCH --account=ACD115198
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn006,25a-hgpn007,25a-hgpn009,25a-hgpn069,25a-hgpn071,25a-hgpn083,25a-hgpn141,25a-hgpn167
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out

# ICLM steps 0-4 of docs/iclm_plan.md, on a random sample of N unique documents or on all of them (N=all).
# Every step skips work whose output exists, so the job can be resubmitted after a timeout.
#   sbatch src/predictable_scaling/sbatch/iclm_proto.sh              # N=1M, TAU=0.90
#   N=200000 TAU=0.92 sbatch src/predictable_scaling/sbatch/iclm_proto.sh
#   N=all TAU=0.94 sbatch --time=6:00:00 src/predictable_scaling/sbatch/iclm_proto.sh   # -> $ROOT/full

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load cuda/13.0

export HF_HOME=/work/${USER}/.cache/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

SRC=/work/${USER}/datasets/dolma3_mix_150B_scored.lance
VER=2
ROOT=/work/${USER}/iclm
N=${N:-1000000}
TAU=${TAU:-0.90}
CAP=${CAP:-32768}
if [ "$N" = all ]; then P=${ROOT}/full; else P=${ROOT}/proto_${N}; fi
mkdir -p "$P"

echo "host: $(hostname) | start: $(date) | N=$N TAU=$TAU CAP=$CAP"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run="uv run --frozen python -m predictable_scaling.iclm"
dist="uv run --frozen torchrun --nproc-per-node=8 -m predictable_scaling.iclm"
step() { echo "=== $1 | $(date +%T)"; }

step doctable
[ -f "$ROOT/doctable.parquet" ] || $run doctable --src "$SRC" --version $VER --out "$ROOT/doctable.parquet"
step sample
if [ "$N" = all ]; then
    ln -sfn "$ROOT/doctable.parquet" "$P/doctable.parquet"
else
    [ -f "$P/doctable.parquet" ] || $run sample --doctable "$ROOT/doctable.parquet" --n "$N" --out "$P/doctable.parquet"
fi
step embed
$dist embed --doctable "$P/doctable.parquet" --src "$SRC" --version $VER --out "$P/emb"
step knn
$dist knn --emb "$P/emb" --out "$P/knn"
step simstats
$run simstats --doctable "$P/doctable.parquet" --emb "$P/emb" --knn "$P/knn" --src "$SRC" --version $VER --out "$P/simstats.md"
step dedup
$run dedup --doctable "$P/doctable.parquet" --knn "$P/knn" --tau "$TAU" --out "$P/keep_${TAU}.npy"
step chains
$run chains --doctable "$P/doctable.parquet" --knn "$P/knn" --keep "$P/keep_${TAU}.npy" --cap "$CAP" --out "$P/chains_${TAU}.parquet"
step inspect
$run inspect --doctable "$P/doctable.parquet" --emb "$P/emb" --chains "$P/chains_${TAU}.parquet" --src "$SRC" --version $VER --out "$P/chains_${TAU}.md"

echo "done: $(date)"
