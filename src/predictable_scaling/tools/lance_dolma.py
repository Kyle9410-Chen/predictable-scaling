"""
Convert between Lance (team storage) and Dolma (documents/*.jsonl.gz) formats.

Lance schema (lossless superset of Dolma):
    id, text, subset, source, added, created, version : string
    metadata, extra, attributes                        : string (JSON)

Commands (parallel via SLURM_PROCID/SLURM_NTASKS, or --rank/--world):
    to-dolma   Lance dataset (local or s3://) -> <out>/documents/part-<frag>.jsonl.gz
    plan       list Dolma document files (run once)
    to-lance   Dolma files -> Lance fragments (uncommitted)
    commit     commit all ranks' fragments as one Lance version
"""
import gzip
import heapq
import json
import os
import sys
from pathlib import Path

import click
import lance
import pyarrow as pa
from compression import zstd
from lance.fragment import FragmentMetadata, write_fragments

STR_COLS = ("id", "text", "subset", "source", "added", "created", "version")
JSON_COLS = ("metadata", "extra", "attributes")
SCHEMA = pa.schema([(c, pa.string()) for c in STR_COLS + JSON_COLS])
DOC_EXTS = (".jsonl.gz", ".json.gz", ".jsonl.zst", ".json.zst", ".jsonl", ".json")


# ------------------------------------------------------------------ helpers
def _open(path, mode="rb"):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, mode, **({"encoding": "utf-8"} if "t" in mode else {}))
    if p.endswith(".zst"):
        return zstd.open(p, mode, **({"encoding": "utf-8"} if "t" in mode else {}))
    return open(p, mode, **({"encoding": "utf-8"} if "t" in mode else {}))


def _rank_world(rank, world):
    return (int(os.environ.get("SLURM_PROCID", 0)) if rank is None else rank,
            int(os.environ.get("SLURM_NTASKS", 1)) if world is None else world)


def _str(v):
    """Coerce incompatible scalars to string; None stays null; nested -> JSON."""
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _json(v):
    return None if v is None else json.dumps(v, ensure_ascii=False, default=str)


def _nonblank(fh):
    for line in fh:
        if line.strip():
            yield line


# ------------------------------------------------------------------ Lance -> Dolma
def row_to_doc(r: dict, fallback_id: str) -> tuple[dict, dict | None]:
    r = dict(r)
    extra, meta_s, attrs_s = r.pop("extra", None), r.pop("metadata", None), r.pop("attributes", None)
    d = json.loads(extra) if extra else {}
    d["id"] = r.pop("id", None) or fallback_id
    d["text"] = r.pop("text", None)
    subset = r.pop("subset", None)
    d["source"] = r.pop("source", None) or (subset.split("-")[0] if subset else "unknown")
    for k in ("added", "created", "version"):
        if (v := r.pop(k, None)) is not None:
            d[k] = v
    if subset is not None:
        d["subset"] = subset
    meta = json.loads(meta_s) if meta_s else None
    if r:  # columns outside SCHEMA (e.g. fineweb-edu's url/score) -> metadata
        meta = {**(meta or {}), **r}
    if meta is not None:
        d["metadata"] = meta
    return d, (json.loads(attrs_s) if attrs_s else None)


@click.group()
def cli():
    pass


@cli.command("to-dolma")
@click.option("--src", required=True, help="Lance dataset (local path or s3://)")
@click.option("--out", required=True, help="Dolma dataset root; writes <out>/documents/")
@click.option("--version", type=int, default=None, help="Lance version (default latest)")
@click.option("--rank", type=int, default=None)
@click.option("--world", type=int, default=None)
@click.option("--batch-size", default=4096)
def to_dolma(src, out, version, rank, world, batch_size):
    rank, world = _rank_world(rank, world)
    if src.startswith("s3://"):
        from flock.hub import s3
        s3.get_process(read_only=True)
    ds = lance.dataset(src, version=version)
    has_attrs = "attributes" in ds.schema.names and ds.count_rows(filter="attributes IS NOT NULL") > 0
    frags = sorted(ds.get_fragments(), key=lambda f: f.fragment_id)[rank::world]
    n = 0
    for frag in frags:
        name = f"part-{frag.fragment_id:06d}.jsonl.gz"
        doc_path = Path(out) / "documents" / name
        if doc_path.exists():  # resumable: a doc file only exists once complete
            continue
        attr_path = Path(out) / "attributes" / "lance" / name
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_doc = doc_path.with_name(name + ".tmp")
        tmp_attr = attr_path.with_name(name + ".tmp") if has_attrs else None
        if tmp_attr:
            attr_path.parent.mkdir(parents=True, exist_ok=True)
            
        gz = lambda p: gzip.open(p, "wt", encoding="utf-8", compresslevel=3)
        with gz(tmp_doc) as fd, (gz(tmp_attr) if tmp_attr else open(os.devnull, "w")) as fa:
            i = 0
            for batch in frag.to_batches(batch_size=batch_size):
                for r in batch.to_pylist():
                    d, attrs = row_to_doc(r, f"{frag.fragment_id}-{i}")
                    fd.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
                    if tmp_attr:  # one line per doc, keeps files aligned
                        fa.write(json.dumps({"id": d["id"], "source": d["source"],
                                             "attributes": attrs or {}}) + "\n")
                    i += 1
            n += i
        if tmp_attr:
            tmp_attr.rename(attr_path)
        tmp_doc.rename(doc_path)
    print(f"rank {rank}: {len(frags)} fragments -> {n:,} docs", flush=True)


# ------------------------------------------------------------------ Dolma -> Lance
def doc_to_row(d: dict, default_subset: str | None, merged_attrs: dict) -> dict:
    row = {c: _str(d.pop(c, None)) for c in STR_COLS if c != "subset"}
    row["subset"] = _str(d.pop("subset", None)) or default_subset
    row["metadata"] = _json(d.pop("metadata", None))
    attrs = {**(d.pop("attributes", None) or {}), **merged_attrs}
    row["attributes"] = _json(attrs) if attrs else None
    row["extra"] = _json(d) if d else None
    return row


def read_dolma(root: Path, rel: str, attr_names: list[str], bad, batch_rows=4096):
    parent = str(Path(rel).parent)
    default_subset = None if parent == "." else parent
    attr_fhs = [_open(root / "attributes" / a / rel) for a in attr_names]
    attr_iters = [_nonblank(fh) for fh in attr_fhs]
    cols = {c: [] for c in SCHEMA.names}

    def flush():
        b = pa.RecordBatch.from_pydict(cols, schema=SCHEMA)
        for v in cols.values():
            v.clear()
        return b

    try:
        with _open(root / "documents" / rel) as f:
            for lineno, line in enumerate(_nonblank(f), 1):
                attr_lines = [next(it) for it in attr_iters]
                try:
                    d = json.loads(line)
                except json.JSONDecodeError as e:
                    bad.write(json.dumps({"file": rel, "line": lineno, "error": str(e)}) + "\n")
                    continue
                merged = {}
                for name, al in zip(attr_names, attr_lines):
                    a = json.loads(al)
                    if str(a.get("id")) != str(d.get("id")):
                        raise ValueError(f"{rel}:{lineno} attributes/{name} id {a.get('id')} != doc id {d.get('id')}")
                    merged.update(a.get("attributes") or {})
                for k, v in doc_to_row(d, default_subset, merged).items():
                    cols[k].append(v)
                if len(cols["id"]) >= batch_rows:
                    yield flush()
        for name, it in zip(attr_names, attr_iters):
            if next(it, None) is not None:
                raise ValueError(f"{rel}: attributes/{name} has more lines than documents")
    finally:
        for fh in attr_fhs:
            fh.close()
    if cols["id"]:
        yield flush()


@cli.command()
@click.option("--src", required=True, help="Dolma root (contains documents/)")
@click.option("--work-dir", required=True)
def plan(src, work_dir):
    docs = Path(src) / "documents"
    files = sorted(p for p in docs.rglob("*") if p.is_file() and p.name.endswith(DOC_EXTS))
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(work_dir) / "files.tsv", "w") as f:
        for p in files:
            f.write(f"{p.stat().st_size}\t{p.relative_to(docs)}\n")
    (Path(work_dir) / "src.txt").write_text(str(Path(src).resolve()))
    print(f"{len(files)} files, {sum(p.stat().st_size for p in files)/1e9:.1f} GB")


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


@cli.command("to-lance")
@click.option("--dest", required=True, help="local Lance dataset path")
@click.option("--work-dir", required=True)
@click.option("--attributes", "attr_names", multiple=True, help="Dolma attribute set(s) to merge, e.g. paloma_decon")
@click.option("--rows-per-file", default=100_000)
@click.option("--rank", type=int, default=None)
@click.option("--world", type=int, default=None)
def to_lance(dest, work_dir, attr_names, rows_per_file, rank, world):
    rank, world = _rank_world(rank, world)
    root = Path((Path(work_dir) / "src.txt").read_text())
    files = my_files(work_dir, rank, world)
    rows = 0
    with open(Path(work_dir) / f"bad_lines.rank{rank:05d}.jsonl", "w") as bad:
        def batches():
            nonlocal rows
            for rel in files:
                for b in read_dolma(root, rel, list(attr_names), bad):
                    rows += b.num_rows
                    yield b
        frags = write_fragments(pa.RecordBatchReader.from_batches(SCHEMA, batches()), dest,
                                schema=SCHEMA, max_rows_per_file=rows_per_file)
        n_bad = bad.tell()
    out = Path(work_dir) / f"rank{rank:05d}.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"rows": rows, "fragments": [f.to_json() for f in frags]}))
    tmp.rename(out)
    print(f"rank {rank}: {len(files)} files -> {rows:,} rows, {len(frags)} fragments"
          f"{' (BAD LINES)' if n_bad else ''}", flush=True)


@cli.command()
@click.option("--dest", required=True)
@click.option("--work-dir", required=True)
@click.option("--expected", required=True, type=int, help="number of ranks")
def commit(dest, work_dir, expected):
    parts = [json.loads(p.read_text()) for p in sorted(Path(work_dir).glob("rank*.json"))]
    if len(parts) != expected:
        sys.exit(f"only {len(parts)}/{expected} ranks finished; not committing")
    frags = [FragmentMetadata.from_json(json.dumps(d)) for p in parts for d in p["fragments"]]
    ds = lance.LanceDataset.commit(dest, lance.LanceOperation.Overwrite(SCHEMA, frags))
    want = sum(p["rows"] for p in parts)
    got = ds.count_rows()
    print(f"committed v{ds.version}: {got:,} rows (ranks reported {want:,}), {len(ds.get_fragments())} fragments")
    if got != want:
        sys.exit("ROW COUNT MISMATCH")


if __name__ == "__main__":
    cli()
