#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --partition=h200_256gpu
#SBATCH --account=ACD115198
#SBATCH --nodes=1
#SBATCH --gpus-per-node=5
#SBATCH --cpus-per-task=40
#SBATCH --mem=1500G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

# Reproduce docs/iclm_results.md §3: lm-eval runs (one model per GPU), then the analyses.
# Re-running skips lm-eval runs that already have results.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge; module load cuda/13.0

export HF_HOME=/work/${USER}/.cache/huggingface TOKENIZERS_PARALLELISM=false HF_DATASETS_TRUST_REMOTE_CODE=1
ulimit -u "$(ulimit -Hu)"
OUT=${OUT:-/work/${USER}/iclm/bench/results/benchmark}
B="uv run --frozen python -m predictable_scaling.benchmark"

echo "========================================"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
START_TIME=$(date +%s)

M=/work/${USER}/predictable-scaling/models
MODELS=(
    $M/iclm-single-t0.94-150M-30B
    $M/iclm-randgroup-k10-t0.94-150M-30B
    $M/iclm-k10-t0.94-dolma3mix-150M-30B
    $M/iclm-randgroup-k32-pp-a0.85-t0.94-150M-30B
    $M/iclm-k32-pp-a0.85-t0.94-150M-30B
)
g=0
for m in "${MODELS[@]}"; do  # one checkpoint per GPU
    CUDA_VISIBLE_DEVICES=$g $B evaluate --model $m --out $OUT &
    g=$((g + 1))
done
wait
$B language-modeling --out $OUT > $OUT/language_modeling.txt
$B multiple-choice --out $OUT > $OUT/multiple_choice.txt

END_TIME=$(date +%s)
echo "========================================"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')  "
echo "Total Cost: $((END_TIME - START_TIME)) s"
echo "========================================"