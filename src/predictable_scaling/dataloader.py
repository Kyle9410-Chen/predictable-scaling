import lance
import os
from flock.hub import s3

def load_dataset():
    ds = lance.dataset(
        "s3://flock-datasets/fineweb-edu",
        storage_options={
            "aws_endpoint": "https://s3.flock.cerulean.works",
            "aws_access_key_id": os.getenv("S3_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("S3_SECRET_ACCESS_KEY"),
            "aws_region": "us-east-1",
            "aws_virtual_hosted_style_request": "false",
        },
    )

    return ds

def main():
    ds = load_dataset()
    print(ds.version, ds.count_rows())
    print(ds.schema)

if __name__ == "__main__":
    main()