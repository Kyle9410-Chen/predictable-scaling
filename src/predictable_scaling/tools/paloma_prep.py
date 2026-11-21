"""HF allenai/paloma -> Dolma documents (for a positive test / building a bloom filter)."""
import argparse, glob, gzip, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="hf download dir")
    ap.add_argument("--out", required=True, help="writes <out>/documents/*.jsonl.gz")
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="drop paragraphs with fewer whitespace tokens (0 = keep all)")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/documents", exist_ok=True)
    files = sorted(glob.glob(f"{a.src}/*/*/*.jsonl.gz"))
    docs = paras = dropped = 0
    for path in files:
        subset, _split, name = path.split("/")[-3:]
        dst = f"{a.out}/documents/{subset}__{name}"
        with gzip.open(path, "rt", encoding="utf-8") as fi, \
             gzip.open(dst, "wt", encoding="utf-8", compresslevel=3) as fo:
            for i, line in enumerate(fi):
                d = json.loads(line)
                ps = [p for p in d["text"].split("\n") if p.strip()]
                keep = [p for p in ps if len(p.split()) >= a.min_tokens]
                dropped += len(ps) - len(keep)
                if not keep:
                    continue
                docs += 1
                paras += len(keep)
                fo.write(json.dumps({"id": str(d.get("id") or f"{subset}/{name}/{i}"),
                                     "text": "\n".join(keep),
                                     "source": f"paloma/{subset}"},
                                    ensure_ascii=False) + "\n")
    print(f"files={len(files)} docs={docs} paragraphs={paras} dropped_short={dropped}")

if __name__ == "__main__":
    main()
