from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time

import lance

from predictable_scaling.dataset_utils import iter_batches, count_rows, schema as get_schema
from predictable_scaling.scorer import mtld, flesch_reading_ease
import pyarrow as pa

def add_prose_difficulty_score(batch: pa.RecordBatch, schema: pa.Schema) -> pa.RecordBatch:
    rows = batch.to_pylist()
    for r in rows:
        r["mtld"] = mtld(r["text"])
        r["flesch_reading_ease"] = flesch_reading_ease(r["text"])
    return pa.RecordBatch.from_pylist(rows, schema), batch.num_rows

def preprocess(src, dst, workers=1, log_every=30.0, limit=None):
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
    preprocess(args.src, args.dst, args.workers, limit=args.limit)
