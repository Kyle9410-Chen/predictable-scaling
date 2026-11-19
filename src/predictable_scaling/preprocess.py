from predictable_scaling.dataset_utils import iter_rows, write
import pyarrow as pa

def preprocess(src, dst):
    out = []
    for row in iter_rows(src):
        if len(row["text"]) < 2000:
            continue
        row["text"] = row["text"].lower()
        out.append(row)
    write(pa.Table.from_pylist(out), dst)

def get_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--src")
    p.add_argument("--dst")
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    preprocess(args.src, args.dst)
