#!/bin/bash
#SBATCH --job-name=data-preprocess
#SBATCH --account=ACD115198
#SBATCH --partition=development
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out

source $HOME/.bashrc

SRC=/work1/$USER/raw.lance
DST=/work1/$USER/processed.lance

echo "========================================"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
START_TIME=$(date +%s)

uv run --frozen src/predictable_scaling/preprocess.py --src $SRC --dst $DST

END_TIME=$(date +%s)
echo "========================================"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')  "
echo "Total Cost: $((END_TIME - START_TIME)) s"
echo "========================================"