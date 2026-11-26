#!/bin/bash
#SBATCH --job-name=preprocess-dataset
#SBATCH --partition=h200_256gpu
#SBATCH --account=ACD115198
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn007,25a-hgpn009,25a-hgpn069,25a-hgpn071,25a-hgpn083,25a-hgpn141,25a-hgpn167
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=5:30:00
#SBATCH --output=logs/%x-%j.out

source $HOME/.bashrc

# one thread per worker process, otherwise numpy/fasttext/torch each spawn 112 threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

SRC=/work/$USER/datasets/dolma3_mix_150B_scored.lance
DST=/work/$USER/datasets/dolma3_mix_150B_scored_edu

echo "========================================"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
START_TIME=$(date +%s)

uv run --frozen src/predictable_scaling/preprocess.py \
    --src $SRC \
    --dst $DST \
    --workers 8

END_TIME=$(date +%s)
echo "========================================"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')  "
echo "Total Cost: $((END_TIME - START_TIME)) s"
echo "========================================"