#!/bin/bash
#SBATCH --job-name=data-preprocess
#SBATCH --account=ACD115198
#SBATCH --partition=ct112
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

source $HOME/.bashrc

# one thread per worker process, otherwise numpy/fasttext/torch each spawn 112 threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

SRC=/work1/$USER/dolma3_mix_150B_decon
DST=/work1/$USER/dolma3_mix_150B_scored

echo "========================================"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
START_TIME=$(date +%s)

uv run --frozen src/predictable_scaling/preprocess.py \
    --src $SRC \
    --dst $DST \
    --workers $SLURM_CPUS_PER_TASK

END_TIME=$(date +%s)
echo "========================================"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')  "
echo "Total Cost: $((END_TIME - START_TIME)) s"
echo "========================================"