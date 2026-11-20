"""
Convert dolma3_mix dataset to Lance format
Handle incompatiable datatypes and missing values
"""

import heapq
import json
import os
import sys
from pathlib import Path
from compression import zstd

import click
import lance
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.json as pj
from lance.fragment import FragmentMetadata, write_fragments


SCHEMA = pa.schema([
    ("id", pa.string()), ("text", pa.string()), ("subset", pa.string()),
    ("source", pa.string()), ("added", pa.string()), ("created", pa.string()),
    ("version", pa.string()), ("metadata", pa.string()), ("extra", pa.string()),
])
TOP_STR = ("source", "added", "created", "version")


def _str(v):
    """Coerce incompatible scalars to string; keep None as null; nested -> JSON."""
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _json(v):
    return None if v is None else json.dumps(v, ensure_ascii=False)


def read_jsonl_zst(path: str, bad_out, batch_rows: int = 4096):
    """Yield RecordBatches with SCHEMA; nothing dropped (unknown keys -> extra, bad lines -> bad_out)."""
    subset = Path(path).parent.name
    cols = {n: [] for n in SCHEMA.names}

    def flush():
        b = pa.RecordBatch.from_pydict(cols, schema=SCHEMA)
        for v in cols.values():
            v.clear()
        return b

    with zstd.open(path, "rb") as f:
        for lineno, line in enumerate(f, 1):  # binary iteration splits on b"\n" only
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                bad_out.write(json.dumps({"file": path, "line": lineno, "error": str(e),
                                          "raw": line.decode("utf-8", "replace")}) + "\n")
                continue
            cols["id"].append(_str(r.pop("id", None)))
            cols["text"].append(_str(r.pop("text", None)))
            cols["subset"].append(subset)
            for k in TOP_STR:
                cols[k].append(_str(r.pop(k, None)))
            cols["metadata"].append(_json(r.pop("metadata", None)))
            cols["extra"].append(_json(r) if r else None)
            if len(cols["id"]) >= batch_rows:
                yield flush()
    if cols["id"]:
        yield flush()


def preprocess(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Conversion is lossless: no filtering here. Do experiments on the converted dataset."""
    return batch

@click.group()
def cli():
    pass

@cli.command()
@click.option("--src", required=True, help="downloaded HF folder")
@click.option("--work-dir", required=True)
def plan(src, work_dir):
    """Run once before srun: list input files with sizes."""
    files = sorted(Path(src).glob("data/**/*.jsonl.zst"))
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(work_dir) / "files.tsv", "w") as f:
        for p in files:
            f.write(f"{p.stat().st_size}\t{p}\n")
    print(f"{len(files)} files, {sum(p.stat().st_size for p in files)/1e9:.1f} GB")

@cli.command()
@click.option("--dest", required=True, help="local Lance dataset path")
@click.option("--work-dir", required=True)
@click.option("--rows-per-file", default=100_000, help="rows per Lance fragment")
def write(dest, work_dir, rows_per_file):
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    files = my_files(work_dir, rank, world)
    rows = 0

    with open(Path(work_dir) / f"bad_lines.rank{rank:05d}.jsonl", "w") as bad:
        def batches():
            nonlocal rows
            for f in files:
                for b in read_jsonl_zst(f, bad):
                    b = preprocess(b)
                    rows += b.num_rows
                    yield b

        frags = write_fragments(pa.RecordBatchReader.from_batches(SCHEMA, batches()), dest,
                                schema=SCHEMA, max_rows_per_file=rows_per_file)
        n_bad = bad.tell()

    out = Path(work_dir) / f"rank{rank:05d}.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps([f.to_json() for f in frags]))
    tmp.rename(out)
    print(f"rank {rank}: {len(files)} files -> {rows:,} rows, {len(frags)} fragments"
          f"{', BAD LINES: see bad_lines file' if n_bad else ''}", flush=True)


def my_files(work_dir: str, rank: int, world: int) -> list[str]:
    """Greedy size balancing (largest file -> least loaded rank); deterministic on every rank."""
    entries = [l.split("\t") for l in (Path(work_dir) / "files.tsv").read_text().splitlines()]
    entries = sorted(((int(s), p) for s, p in entries), reverse=True)
    heap = [(0, r) for r in range(world)]
    mine = []
    for size, path in entries:
        load, r = heapq.heappop(heap)
        if r == rank:
            mine.append(path)
        heapq.heappush(heap, (load + size, r))
    return mine

@cli.command()
@click.option("--dest", required=True)
@click.option("--work-dir", required=True)
@click.option("--expected", required=True, type=int)
def commit(dest, work_dir, expected):
    paths = sorted(Path(work_dir).glob("rank*.json"))
    if len(paths) != expected:
        sys.exit(f"only {len(paths)}/{expected} ranks finished; not committing")
    frags = [FragmentMetadata.from_json(json.dumps(d))
             for p in paths for d in json.loads(p.read_text())]
    ds = lance.LanceDataset.commit(dest, lance.LanceOperation.Overwrite(SCHEMA, frags))
    print(f"committed v{ds.version}: {ds.count_rows():,} rows, {len(ds.get_fragments())} fragments")


if __name__ == "__main__":
    cli()


# Download
"""
uv run flock datasets download allenai/dolma3_mix-150B-1025 --to /work1/$USER/dolma3_mix_150B_hf
"""

# Convert
"""
#!/bin/bash
#SBATCH --job-name=data-preprocess
#SBATCH --account=ACD115198
#SBATCH --partition=development
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

source $HOME/.bashrc

SRC=/work1/$USER/dolma3_mix_150B_hf
DEST=/work1/$USER/dolma3_mix_150B_lance
WORK=/work1/$USER/.preprocess/$SLURM_JOB_ID
PY=".venv/bin/python src/predictable_scaling/tools/convert_dolma3mix.py"
export LANCE_CPU_THREADS=1 LANCE_IO_THREADS=4

echo "========================================"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
START_TIME=$(date +%s)

$PY plan --src "$SRC" --work-dir "$WORK"
srun --kill-on-bad-exit=1 $PY write --dest "$DEST" --work-dir "$WORK"
$PY commit --dest "$DEST" --work-dir "$WORK" --expected "$SLURM_NTASKS"

END_TIME=$(date +%s)
echo "========================================"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')  "
echo "Total Cost: $((END_TIME - START_TIME)) s"
echo "========================================"
"""

# Upload to S3
"""
uv run flock s3 configure      # once
uvx --from awscli aws s3 sync /work1/$USER/dolma3_mix_150B_lance \
    s3://flock-datasets/lab7/ben-kyle/dolma3_mix_150B --profile flock
uv run python -m predictable_scaling.dataset_utils show \
    --src s3://flock-datasets/lab7/ben-kyle/dolma3_mix_150B --limit 3
"""