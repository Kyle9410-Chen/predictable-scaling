#!/bin/bash
#SBATCH --job-name=data-preprocess
#SBATCH --account=ACD115198
#SBATCH --partition=ct112
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

source $HOME/.bashrc

# one thread per worker process, otherwise numpy/fasttext/torch each spawn 112 threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
# compute nodes have no network; use the HF cache in $HOME directly
export HF_HUB_OFFLINE=1

SRC=/work1/$USER/dolma3_mix_150B_scored_n_tok
DST=/work1/$USER/dolma3_mix_150B_flesch_warmup

echo "========================================"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
START_TIME=$(date +%s)

uv run --frozen src/predictable_scaling/preprocess.py \
    --src $SRC \
    --dst $DST \
    --workers 8 \
    --cl-tokens 300000000 \
    "$@"

END_TIME=$(date +%s)
echo "========================================"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')  "
echo "Total Cost: $((END_TIME - START_TIME)) s"
echo "========================================"