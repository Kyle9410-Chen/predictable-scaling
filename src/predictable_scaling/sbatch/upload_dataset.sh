# On F1, only login node has network.
# So the upload must happen on the login node.

SRC=/work1/$USER/dolma3_mix_150B_train
DST=s3://flock-datasets/-projects/28/ben-kyle/dolma3_mix_150B_train
uv run --frozen src/predictable_scaling/dataset_utils.py upload --src $SRC --dst $DST --mode overwrite