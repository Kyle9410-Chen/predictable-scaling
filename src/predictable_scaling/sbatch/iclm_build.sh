#!/bin/bash
#SBATCH --job-name=iclm-build
#SBATCH --partition=h200_256gpu
#SBATCH --account=ACD115198
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn006,25a-hgpn007,25a-hgpn009,25a-hgpn069,25a-hgpn071,25a-hgpn083,25a-hgpn141,25a-hgpn167
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=1700G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out

# ICLM steps 5-7 of docs/iclm_plan.md (CPU only): build the iclm, randgroup and single datasets from the
# step 3-4 outputs of iclm_proto.sh (N=all), then check each one.
#   TAU=0.94 sbatch src/predictable_scaling/sbatch/iclm_build.sh
# A dataset that is already committed is skipped. If a write dies before committing, delete its partial
# output directory and its tmp directory before resubmitting (the writer refuses to overwrite).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export HF_HOME=/work/${USER}/.cache/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
# Each of the ~24 worker processes would otherwise start a Lance/Arrow pool of one thread per CPU,
# which exceeds the per-user thread limit ("OS can't spawn worker thread").
export LANCE_CPU_THREADS=4
export LANCE_IO_THREADS=8
export RAYON_NUM_THREADS=4
ulimit -u "$(ulimit -Hu)"

SRC=/work/${USER}/datasets/dolma3_mix_150B_scored.lance
VER=2
K=${K:-10}
TAU=${TAU:-0.94}
P=/work/${USER}/iclm/full
OUT=/work/${USER}/datasets
DATASETS=${DATASETS:-"iclm randgroup single"}
# Chain variants (built here when the chains file is missing):
#   PERPASS=1   separate walk per repeat pass, no two documents adjacent in more than one row
#   MAXADJ=0.85 never put two documents more similar than this next to each other
V=""
[ -n "${PERPASS:-}" ] && V="pp"
[ -n "${MAXADJ:-}" ] && V="${V:+${V}_}a${MAXADJ}"
TAG=k${K}${V:+_$V}_t${TAU}
# k=10 plain chains predate the k suffix; other k come from iclm_k.sh
if [ "$K" = 10 ] && [ -z "$V" ]; then SFX=${TAU}; else SFX=k${K}${V:+_$V}_${TAU}; fi
CHAINS=$P/chains_${SFX}.parquet
RAND=$P/randgroup_${SFX}.parquet
if [ "$K" = 10 ]; then KNN=$P/knn; else KNN=$P/knn_k${K}; fi

echo "host: $(hostname) | start: $(date) | TAU=$TAU K=$K V=${V:-none} | $DATASETS"
run="uv run --frozen python -m predictable_scaling.iclm"
step() { echo "=== $1 | $(date +%T)"; }

if [ ! -f "$CHAINS" ]; then
    step chains
    $run chains --doctable "$P/doctable.parquet" --knn "$KNN" --keep "$P/keep_${TAU}.npy" --cap 32768 \
        --out "$CHAINS" ${PERPASS:+--per-pass} ${MAXADJ:+--max-adj "$MAXADJ"}
    step inspect
    $run inspect --doctable "$P/doctable.parquet" --emb "$P/emb" --chains "$CHAINS" \
        --src "$SRC" --version $VER --out "${CHAINS%.parquet}.md"
fi

step randgroup
[ -f "$RAND" ] || $run randgroup --doctable "$P/doctable.parquet" --chains "$CHAINS" --out "$RAND"

build() {  # name groups
    local D=$OUT/dolma3_mix_150B_$1.lance
    step "write $1"
    if [ -d "$D/_versions" ]; then
        echo "$D already committed, skipping write"
    else
        $run write --doctable "$P/doctable.parquet" --groups "$2" --keep "$P/keep_${TAU}.npy" \
            --src "$SRC" --version $VER --out "$D" --tmp "$P/tmp_$1"
    fi
    step "check $1"
    $run check --dataset "$D" --doctable "$P/doctable.parquet" --groups "$2" --keep "$P/keep_${TAU}.npy"
}

for d in $DATASETS; do
    case $d in
        iclm) build "iclm_${TAG}" "$CHAINS" ;;
        randgroup) build "randgroup_${TAG}" "$RAND" ;;
        single) build "single_t${TAU}" single ;;  # depends on tau only
    esac
done

echo "done: $(date)"
