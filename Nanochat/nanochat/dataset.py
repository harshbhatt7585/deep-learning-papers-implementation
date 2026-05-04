import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir


BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542 # last datashard is shard_06542.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet"
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")


def list_parquet_files(data_dir=None, warn_on_legancy=False):

    data_dir = DATA_DIR if data_dir is None else data_dir

    if not os.path.exists(data_dir):
        data_dir = os.path.join(base_dir, "base_data")

    parquet_files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    )
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths