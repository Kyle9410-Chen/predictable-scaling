from concurrent.futures import ProcessPoolExecutor
from functools import cache, partial
import time
import json
from pathlib import Path
import datetime

import lance
from transformers import AutoTokenizer

from predictable_scaling.dataset_utils import iter_batches, count_rows, schema as get_schema
from predictable_scaling.scorer import mtld, flesch_reading_ease
import pyarrow as pa
import pyarrow.ipc as ipc, shutil
import numpy as np

# produce train and evaluation split as seperate indices.
# train split include curriculum part.
# training should train on train dataset without shuffling
def build_ordered_schedule(src, key, curriculum_tokens, n_eval, seed: int=0, descending: bool=False):
    ds = lance.dataset(src)
    t = ds.to_table(columns=[key, "n_tokens"])
    s, n_tokens = t[key].to_numpy(zero_copy_only=False), t["n_tokens"].to_numpy()
    pos = np.flatnonzero(np.isfinite(s))
    np.random.default_rng(seed).shuffle(pos)

    # split train and eval
    val, train = pos[:n_eval], pos[n_eval:]

    # split train into curriculum and rest
    n_cl = int(np.searchsorted(np.cumsum(n_tokens[train]), curriculum_tokens))
    cl, rest = train[:n_cl], train[n_cl:]

    # sort curriculum
    cl = cl[np.argsort(-s[cl] if descending else s[cl], kind="stable")]
    rest = np.sort(rest)
    val = np.sort(val)

    # print token counts
    print("Curriculum tokens:", n_tokens[cl].sum())
    print("Total train tokens:", n_tokens[train].sum())
    print("Eval tokens:", n_tokens[val].sum())

    return val, cl, rest, n_tokens

def _take_in_order(ds, idx):
    o = np.argsort(idx)
    return ds.take(idx[o]).take(np.argsort(o))

def _write_cl(ds, dst, cl, n_tokens, n_cl_frags):
    cum = np.cumsum(n_tokens[cl])
    groups = np.split(cl, np.searchsorted(cum, cum[-1] * np.arange(1, n_cl_frags) / n_cl_frags))
    for i, g in enumerate(groups):
        lance.write_dataset(_take_in_order(ds, g), dst, mode="create" if i==0 else "append",
                            max_rows_per_file=len(g) + 1)
    tok = [int(n_tokens[g].sum()) for g in groups]
    print(f"CL: {len(groups)} frags, tokens/frag min {min(tok):,} max {max(tok):,}")

def _bucket_shuffle(ds, dst, rest, mode, tmp, n_buckets=128, seed=0, rows_per_file=32_768):
    """Uniformly shuffle rows `rest` (sorted positions) into dst using two sequential passes."""
    rng = np.random.default_rng(seed + 1)
    keep = np.zeros(ds.count_rows(), dtype=bool)
    keep[rest] = True

    # pass 1: scatter, sequential read of the source
    tmp = Path(tmp)
    if tmp.exists():
        print(f"{tmp} exists, skipping pass 1 (resume)", flush=True)
    else:
        tmp.mkdir(parents=True, exist_ok=False)
        sinks = [pa.OSFile(str(tmp / f"b{b:04d}.arrow"), "wb") for b in range(n_buckets)]
    
        writers = [ipc.new_stream(f, ds.schema) for f in sinks]
        offset = 0
        for batch in ds.to_batches(batch_size=65_536):            # in row-position order
            m = keep[offset:offset + batch.num_rows]
            offset += batch.num_rows
            if not m.any():
                continue
            batch = batch.filter(pa.array(m))
            bucket = rng.integers(n_buckets, size=batch.num_rows)
            for b in np.unique(bucket):
                writers[b].write_batch(batch.filter(pa.array(bucket == b)))
        assert offset == len(keep)
        for w, f in zip(writers, sinks):
            w.close(); f.close()

    # pass 2: gather, shuffle each bucket in memory, append
    large = pa.schema([f.with_type(pa.large_string()) if f.type == pa.string() else f for f in ds.schema]) # cast into large string to prevent overflow
    for b in range(n_buckets):
        path = tmp / f"b{b:04d}.arrow"
        with pa.memory_map(str(path)) as f:
            t = ipc.open_stream(f).read_all().cast(large).combine_chunks()
        perm = rng.permutation(t.num_rows)
        batches = (b for s in range(0, len(perm), 100_000) for b in t.take(perm[s:s + 100_000]).cast(ds.schema).to_batches())
        lance.write_dataset(pa.RecordBatchReader.from_batches(ds.schema, batches), dst, mode=mode, max_rows_per_file=rows_per_file)
        mode = "append"
        path.unlink()
        print(f"bucket {b + 1}/{n_buckets} written", flush=True)
    shutil.rmtree(tmp)


def materialize_schedule(src, dst, cl, rest, n_tokens, n_cl_frags, tmp, seed=0):
    ds = lance.dataset(src)
    if len(cl):
        _write_cl(ds, dst, cl, n_tokens, n_cl_frags)
    _bucket_shuffle(ds, dst, rest, "append" if len(cl) else "create", tmp, seed=seed)

def materialize_rows(src, dst, idx, chunk=200_000, rows_per_file=32_768):
    """Plain sequential copy of sorted row positions (used for eval)."""
    ds = lance.dataset(src)
    def batches():
        for i in range(0, len(idx), chunk):
            yield from ds.take(idx[i:i + chunk]).to_batches()
    reader = pa.RecordBatchReader.from_batches(ds.schema, batches())
    return lance.write_dataset(reader, dst, mode="create", max_rows_per_file=rows_per_file)

def add_prose_difficulty_score(batch: pa.RecordBatch, schema: pa.Schema) -> pa.RecordBatch:
    rows = batch.to_pylist()
    for r in rows:
        r["mtld"] = mtld(r["text"])
        r["flesch_reading_ease"] = flesch_reading_ease(r["text"])
    return pa.RecordBatch.from_pylist(rows, schema), batch.num_rows

def score_difficulty(src, dst, workers=1, log_every=30.0, limit=None):
    schema = (get_schema(src)
              .append(pa.field("mtld", pa.float64()))
              .append(pa.field("flesch_reading_ease", pa.float64())))
    
    total_rows = count_rows(src)
    print(f"Total rows: {total_rows}")

    with ProcessPoolExecutor(workers) as ex:
        results = ex.map(
            partial(add_prose_difficulty_score, schema=schema),
            iter_batches(src, limit=limit),
            buffersize=workers * 2, # Py3.14+: bounds in-flight batches -> bounded memory
        )
        reader = pa.RecordBatchReader.from_batches(schema, _log_progress(results, total_rows, every=log_every))
        return lance.write_dataset(reader, dst, mode="create")

@cache
def _tokenizer():
    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

def count_tokens(batch, schema):
    ids = _tokenizer()(batch.column("text").to_pylist())["input_ids"]
    n = pa.array([len(x)+1 for x in ids], pa.int32()) # +1 = EOS appended in PackedDataset
    return pa.RecordBatch.from_arrays([*batch.columns, n], schema=schema), batch.num_rows

def add_token_count(src, dst, workers=1, log_every=30.0, limit=None):
    schema = (get_schema(src)
                  .append(pa.field("n_tokens", pa.int32())))
    total_rows = count_rows(src)
    print(f"Total rows: {total_rows}")

    with ProcessPoolExecutor(workers) as ex:
        results = ex.map(
            partial(count_tokens, schema=schema),
            iter_batches(src, limit=limit),
            buffersize=workers * 2,
        )
        reader = pa.RecordBatchReader.from_batches(schema, _log_progress(results, total_rows, every=log_every))
        return lance.write_dataset(reader, dst, mode="create")

def _log_progress(results, total: int | None, every: float = 30.0):
    """Pass batches through, printing progress every `every` seconds."""
    start = last = time.monotonic()
    n_in = n_out = n_batches = 0

    def report(final=False):
        elapsed = time.monotonic() - start
        rate = n_in / elapsed if elapsed else 0.0
        if total:
            pct = f"{n_in / total:6.2%}"
            eta = f"eta {(total - n_in) / rate / 60:7.1f} min" if rate and not final else ""
        else:
            pct, eta = "", ""
        kept = f"{n_out / n_in:.1%}" if n_in else "-"
        print(f"[{'done' if final else 'progress'}] {n_in:,}/{total or '?':,} rows {pct} | "
              f"kept {n_out:,} ({kept}) | batches {n_batches:,} | "
              f"{rate:,.0f} rows/s | elapsed {elapsed / 60:.1f} min {eta}", flush=True)

    for out_batch, n in results:
        n_in += n
        n_out += out_batch.num_rows
        n_batches += 1
        if time.monotonic() - last >= every:
            report()
            last = time.monotonic()
        if out_batch.num_rows:
            yield out_batch
    report(final=True)

def write_readme(dst, *, src, key, descending, cl_frac, n_cl, n_train, n_val, seed, val_uri, rows_per_file):
    meta = {
        "source": src, "sort_key": key, "descending": descending,
        "cl_frac": cl_frac, "n_cl": n_cl, "n_train": n_train,
        "n_val": n_val, "val_uri": val_uri, "seed": seed,
        "max_rows_per_file": rows_per_file,
        "created": datetime.date.today().isoformat(),
    }
    front = "\n".join(f"{k}: {json.dumps(v)}" for k, v in meta.items())  # JSON values are valid YAML
    order = "descending (easy→hard)" if descending else "ascending (easy→hard)"
    body = f"""# {Path(dst).name}

Curriculum-ordered copy of `{src}`. Read sequentially; do **not** shuffle fragments.

| rows | content |
|---|---|
| `[0, {n_cl:,})` | random {cl_frac:.0%} subset, sorted by `{key}` {order} |
| `[{n_cl:,}, {n_train:,})` | remaining rows, randomly shuffled |

Held-out validation set ({n_val:,} random rows, excluded above): `{val_uri}`.
Built with seed {seed}, non-finite `{key}` rows dropped.
"""
    Path(dst, "README.md").write_text(f"---\n{front}\n---\n\n{body}")

def get_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--src")
    p.add_argument("--dst")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int)
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()

    # produced s3://flock-datasets/lab7/ben-kyle/dolma3_mix_150B_scored
    # ---
    # score_difficulty(args.src, args.dst, args.workers, limit=args.limit, log_every=30.0)

    # add token count to s3://flock-datasets/lab7/ben-kyle/dolma3_mix_150B_scored
    # ---
    # add_token_count(args.src, args.dst, args.workers, limit=args.limit, log_every=30.0)

    # Curriculum fragments
    # Each dataloader shard will get a fragment with round robin
    # So smaller fragments means the batch see wider range of difficulty
    # produce s3://flock-datasets/-projects/28/ben-kyle/dolma3_mix_150B_eval and s3://flock-datasets/-projects/28/ben-kyle/dolma3_mix_150B_train
    # ---
    # val, cl, rest, n_tokens= build_ordered_schedule(args.src, "mtld", 0, 20000)
    # materialize_schedule(args.src, "/work1/benyang0509/dolma3_mix_150B_train", cl, rest, n_tokens,
    #                  n_cl_frags=128, tmp="/work1/benyang0509/_shuffle_tmp")
    # materialize_rows(args.src, "/work1/benyang0509/dolma3_mix_150B_eval", val)
    