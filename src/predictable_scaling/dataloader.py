import lance
from flock.hub import s3

def load_dataset():
    s3.get_process(read_only=True)

    ds = lance.dataset(
        "s3://flock-datasets/fineweb-edu",
        version=1
    )

    return ds

def main():
    ds = load_dataset()
    print(ds.version, ds.count_rows())
    print(ds.schema)

if __name__ == "__main__":
    main()