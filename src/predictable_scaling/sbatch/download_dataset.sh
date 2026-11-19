# On F1, only login node has network.
# So the download must happen on the login node.

SRC=s3://flock-datasets/fineweb-edu
DST=/work1/$USER/raw.lance
uv run --frozen src/predictable_scaling/dataset_utils.py download --src $SRC --dst $DST --limit 10