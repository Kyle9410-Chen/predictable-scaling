from predictable_scaling.dataloader import load_dataset
import lance
import pyarrow as pa
from typing import Literal
from flock.hub import s3
from pathlib import Path
from argparse import ArgumentParser
import os
import time
from concurrent.futures import ThreadPoolExecutor

def _stream(ds: lance.LanceDataset, columns: list[str]|None=None, limit: int | None=None, batch_size: int=1024):
    """
    Stream rows from dataset. Instead of loading all into memory at once
    """
    scanner = ds.scanner(columns=columns, limit=limit, batch_size=batch_size)
    return scanner.to_reader()

def copy_exact(src_uri: str, version: int | None, dst_dir: str, workers: int = 6) -> int:
    """
    Copy one version of a Lance dataset from S3 to local disk, byte for byte (no re-encoding):
    same files, fragments, row order and version number. Safe to rerun: finished files are skipped.

    Args:
        src_uri (str): s3 path to dataset
        version (int): dataset version (None = latest)
        dst_dir (str): local directory for the copy
        workers (int): files copied at the same time

    Returns:
        int: number of rows in the copied version
    """
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config

    s3.get_process(read_only=True)
    ds = lance.dataset(src_uri, version=version)
    fragments = ds.get_fragments()
    data_files = [f for frag in fragments for f in frag.metadata.files]

    # what a plain file copy can't reproduce
    bases = {f.base_id for f in data_files} - {None}
    if bases:
        roots = [base.path for id, base in ds.base_paths().items() if id in bases]
        raise ValueError(f"{src_uri} is a shallow clone; copy the dataset it points to: {roots}")
    if any(frag.metadata.deletion_file for frag in fragments) or ds.list_indices():
        raise ValueError("datasets with deleted rows or indices are not supported")

    bucket, _, prefix = src_uri.removeprefix("s3://").partition("/")
    prefix = prefix.rstrip("/") + "/"
    client = boto3.client("s3", config=Config(retries={"max_attempts": 10, "mode": "adaptive"},
                                              max_pool_connections=workers * 4 + 4))
    size = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            size[obj["Key"][len(prefix):]] = obj["Size"]

    # Lance names manifests either "<2^64-1-version>.manifest" (new) or "<version>.manifest" (old)
    v = ds.version
    manifest = next(k for k in (f"_versions/{2**64 - 1 - v}.manifest", f"_versions/{v}.manifest") if k in size)
    files = sorted({"data/" + f.path for f in data_files})
    files += [k for k in size if k.startswith(f"_transactions/{v - 1}-")]

    config = TransferConfig(multipart_chunksize=64 << 20, max_concurrency=4)

    def fetch(key: str) -> None:
        path = os.path.join(dst_dir, key)
        if os.path.exists(path) and os.path.getsize(path) == size[key]:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for attempt in range(6):
            try:
                client.download_file(bucket, prefix + key, path + ".part", Config=config)
                if os.path.getsize(path + ".part") != size[key]:
                    raise IOError(f"{key}: incomplete download")
                os.replace(path + ".part", path)  # only complete files get their final name
                return
            except Exception as e:
                if attempt == 5:
                    raise
                print(f"retry {attempt + 1}: {key}: {e}", flush=True)
                time.sleep(10 * (attempt + 1))

    total, start = sum(size[k] for k in files), time.time()
    print(f"copying {src_uri} @ v{v}: {len(files)} files, {total / 1e9:.2f} GB", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        for _ in pool.map(fetch, files):
            pass
    fetch(manifest)  # last: an unfinished copy can't be opened as a dataset
    print(f"done in {time.time() - start:.0f}s", flush=True)

    return lance.dataset(dst_dir, version=v).count_rows()

def download(src_uri: str, version: int, dst_dir: str, columns: list[str], mode: str="create", limit: int=20):
    """
    Download dataset from s3 to local
    
    Args:
        src_uri (str): s3 path to dataset
        version (int): dataset version
        dst_dir (str): local path to save dataset
        mode (str): "create" | "overwrite" | "append"
        limit (int): max rows to download, default 20

    Returns:
        number of rows downloaded
    """
    s3.get_process(read_only=True)
    src = lance.dataset(src_uri, version)
    print(f"src {src_uri} @ v{src.version}: {src.count_rows()} rows\n{src.schema}")
    out = lance.write_dataset(_stream(src, columns=columns, limit=limit), dst_dir, mode=mode)
    return out.count_rows()

def upload(src_dir: str, dst_uri: str, mode="create") -> int:
    """
    Upload local dataset to s3

    Args:
        src_dir (str): local dataset path
        dst_uri (str): s3 path to upload
        mode (str): "create" | "overwrite" | "append"
    
    Returns:
        int: uploaded dataset version
    """
    s3.get_process()
    src = lance.dataset(str(src_dir))
    out = lance.write_dataset(_stream(src), dst_uri, mode=mode)
    print(f"uploaded {out.count_rows()} rows to {dst_uri} @ v{out.version}")

    return out.version

def _open(src: str, version: int | None = None) -> lance.LanceDataset:
    if str(src).startswith("s3://"):
        s3.get_process(read_only=True)
    return lance.dataset(src, version=version)

def read(src_dir: str, columns: list[str]|None=None, limit: int|None=1000, offset: int | None = None,
         fmt: Literal["rows", "pandas", "arrow"] = "rows", max_rows: int|None= 1_000_000):
    """
    Read local Lance dataset into memory.

    Args:
        src_dir: local dataset dir
        columns: columns to load (None = all)
        limit: max rows to load (default 1000; None = all, subject to max_rows)
        offset: rows to skip first
        fmt: "rows" -> list[dict], "pandas" -> DataFrame, "arrow" -> pa.Table
        max_rows: refuse limit=None on datasets bigger than this (None = no check)

    For full passes over large datasets use iter_rows() instead.
    """
    ds = _open(src_dir)
    if limit is None and max_rows is not None:
        n = ds.count_rows()
        if n > max_rows:
            raise ValueError(f"{src_dir} has {n:,} rows (> max_rows={max_rows:,}); "
                             "pass limit=, raise max_rows=, or use iter_rows()")

    t = ds.to_table(columns=columns, limit=limit, offset=offset)
    if fmt == "rows":
        return t.to_pylist()
    if fmt == "pandas":
        return t.to_pandas()
    if fmt == "arrow":
        return t
    raise ValueError(f"Unknown format: {fmt!r}")

def iter_batches(src_dir: str, columns: list[str] | None = None, limit: int | None = None,
                 offset: int | None = None, batch_size: int = 1024):
    """Lazily yield pyarrow RecordBatches. Use for parallel / vectorized processing."""
    ds = _open(src_dir)
    yield from ds.to_batches(columns=columns, limit=limit, offset=offset, batch_size=batch_size)

def iter_rows(src_dir: str, columns: list[str]|None=None, limit: int|None=None, offset: int | None = None, batch_size: int=1024):
    """
    Lazily yield rows as dicts; memory stays at ~one batch regardless of dataset size.

    Args:
        limit / offset: row slice, pushed down to Lance
        batch_size: rows per batch

    Example:
        for row in iter_rows(path, columns=["text"], offset=1_000, limit=500):
            ...
    """
    for batch in iter_batches(src_dir, columns, limit, offset, batch_size):
        yield from batch.to_pylist()

def write(table: pa.Table, dst_dir: str, mode="create"):
    """
    Write a table to a Lance dataset locally.

    Args:
        table: Arrow table to write
        dst_dir: local dir
        mode: create/overwrite/append
    
    Example:
    ---
        table = pa.Table.from_pylist([{"id": 1, "text": "hello"}])
        write(table, "/tmp/my.lance", mode="create")
    """
    lance.write_dataset(table, dst_dir, mode=mode)

def schema(src_dir: str) -> pa.Schema:
    return _open(src_dir).schema

def count_rows(src_dir: str) -> int:
    return _open(src_dir).count_rows()

# example of preprocessing
"""
def preprocess_row(row: dict) -> dict | None:
    text = row.get("text", "")
    if len(text) < 2000:
        return None
    return {"id": row.get("id"), "text": text.lower()}

def preprocess():
    raw = lance.dataset(str(RAW)).to_table()
    rows = [out for r in raw.to_pylist() if (out := preprocess_row(r)) is not None]
    t = pa.Table.from_pylist(rows)
    lance.write_dataset(t, str(PROCESSED), mode="overwrite")
"""

def get_args():
    p = ArgumentParser(description="Move Lance datasets between Flock S3 and local disk")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="S3 -> local")
    d.add_argument("--src", required=True, help="s3://flock-datasets/...")
    d.add_argument("--dst", required=True, help="local dir, e.g. /work1/$USER/raw.lance")
    d.add_argument("--version", type=int, default=1, help="source version (0 = latest)")
    d.add_argument("--columns", nargs="+", default=None, help="e.g. --columns id text (default: all)")
    d.add_argument("--limit", type=int, default=20, help="max rows (0 = all)")
    d.add_argument("--mode", choices=["create", "overwrite", "append"], default="create")

    u = sub.add_parser("upload", help="local -> S3")
    u.add_argument("--src", required=True, help="local Lance dataset dir")
    u.add_argument("--dst", required=True, help="s3://flock-datasets/...")
    u.add_argument("--mode", choices=["create", "overwrite", "append"], default="create")

    s = sub.add_parser("show", help="print rows of a local or s3 dataset")
    s.add_argument("--src", required=True)
    s.add_argument("--columns", nargs="+", default=None)
    s.add_argument("--limit", type=int, default=5)
    s.add_argument("--offset", type=int, default=0)
    s.add_argument("--width", type=int, default=200, help="truncate long strings")

    c = sub.add_parser("copy", help="S3 -> local, byte for byte")
    c.add_argument("--src", required=True, help="s3://flock-datasets/...")
    c.add_argument("--dst", required=True, help="local dir")
    c.add_argument("--version", type=int, default=0, help="source version (0 = latest)")

    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.cmd == "download":
        n = download(args.src, version=args.version or None, dst_dir=args.dst, columns=args.columns, mode=args.mode, limit=args.limit or None)
        print(f"downloaded {n} rows to {args.dst}")
    if args.cmd == "upload":
        v = upload(args.src, args.dst, mode=args.mode)
        print(f"uploaded to {args.dst} @ v{v}")
    if args.cmd == "show":
        ds = _open(args.src)
        print(f"{args.src} @ v{ds.version}: {ds.count_rows()} rows, {len(ds.get_fragments())} fragments\n{ds.schema}\n")
        for i, r in enumerate(read(args.src, args.columns, args.limit, args.offset), args.offset):
            print(f"#{i} " + " | ".join(
                f"{k}={(repr(v[:args.width]) + '...') if isinstance(v, str) and len(v) > args.width else repr(v)}"
                for k, v in r.items()))
    if args.cmd == "copy":
        n = copy_exact(args.src, args.version or None, args.dst)
        print(f"copied {n} rows to {args.dst}")