# On F1, only login node has network.
# So the upload must happen on the login node.

SRC=/work1/$USER/raw.lance
DST=s3://flock-datasets/lab7/ben-kyle/smoke-fineweb-edu
uv run --frozen src/predictable_scaling/dataset_utils.py upload --src $SRC --dst $DST --mode overwrite