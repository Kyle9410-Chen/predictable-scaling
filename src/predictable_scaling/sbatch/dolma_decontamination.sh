#!/bin/bash
#SBATCH --job-name=data-preprocess
#SBATCH --account=ACD115198
#SBATCH --partition=development
#SBATCH --nodes=1
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

source $HOME/.bashrc
set -euo pipefail

echo "========================================"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
START_TIME=$(date +%s)

T=src/predictable_scaling/tools/lance_dolma.py
MARK_CONFIG=src/predictable_scaling/tools/mark.yaml
REMOVE_CONFIG=src/predictable_scaling/tools/remove.yaml
PY=.venv/bin/python
WORK=/work1/$USER/.preprocess/$SLURM_JOB_ID
export TMPDIR=/work1/$USER/tmp; mkdir -p "$TMPDIR"

# 1. Lance -> Dolma (compute node, local Lance; one output file per fragment, 1149 files)
srun --kill-on-bad-exit=1 $PY $T to-dolma --src /work1/$USER/dolma3_mix_150B_lance --out /work1/$USER/dolma3_dolma

# 2. Dolma mark step only (writes attributes/paloma_decon/part-*.jsonl.gz)
# mark with dolma 0.9.2: AI2's paloma_decon.bin only works with ahash 0.8.3 (dolma >=1.0 silently flags 0)
# venv: uv venv /work1/$USER/.venvs/dolma092 --python 3.11 && uv pip install --python ... dolma==0.9.2
/work1/$USER/.venvs/dolma092/bin/dolma -c $MARK_CONFIG dedupe          # documents: /work1/$USER/dolma3_dolma/documents/*.jsonl.gz

## Guard: check if marking flags nothing (which should not happen)
FLAGGED=$(zcat /work1/$USER/dolma3_dolma/attributes/paloma_decon/*.jsonl.gz | grep -c '_decontamination":\[\[' || true)
echo "paloma flagged docs: $FLAGGED"
[ "$FLAGGED" -gt 0 ] || { echo "ERROR: 0 flagged — filter/hash mismatch?"; exit 1; }

dolma -c $REMOVE_CONFIG mix
## Guard: check if mixer dropped nothing
DROPPED=$(grep -o 'Dropped [0-9]*' logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out | awk '{s+=$2} END {print s+0}')
echo "mixer dropped: $DROPPED"
[ "$DROPPED" -gt 0 ] || { echo "ERROR: mixer dropped 0 — filter mismatch?"; exit 1; }


# 3. Dolma -> Lance
$PY $T plan --src /work1/$USER/dolma3_decon --work-dir $WORK
srun --kill-on-bad-exit=1 $PY $T to-lance --dest /work1/$USER/dolma3_mix_150B_decon --work-dir $WORK
$PY $T commit --dest /work1/$USER/dolma3_mix_150B_decon --work-dir $WORK --expected $SLURM_NTASKS

END_TIME=$(date +%s)
echo "========================================"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')  "
echo "Total Cost: $((END_TIME - START_TIME)) s"
echo "========================================"