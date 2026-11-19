"""Download the TwinD training set from the Hugging Face Hub and extract it to PNG frames.

    python prepare_data.py                              # TwinD: reals, CogVideoX reconstructions, FF++ fakes
    python prepare_data.py --decoders ltx2 kvae2d       # + reconstructions of other autoencoders (ablations)

Writes <out>/<path> for every frame and <out>/index.csv (path, label, dataset, method, video, paired_real).
"""

import argparse
import os
from glob import glob

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from tqdm import tqdm

REPO = "yermandy/TwinD"
TWIND = ["real", "cogvideox", "DF", "F2F", "FS", "NT"]
DECODERS = ["cogvideox", "kvae2d", "ltxvideo", "hunyuanvideo", "ltx2"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    parser.add_argument("--decoders", nargs="*", default=[], choices=DECODERS)
    parser.add_argument("--source", default=REPO, help="Hugging Face dataset repo or a local copy of it")
    args = parser.parse_args()

    methods = sorted(set(TWIND + args.decoders))
    patterns = [f"data/*/{m}/*.parquet" for m in methods]
    if os.path.isdir(args.source):
        root = args.source
    else:
        root = snapshot_download(args.source, repo_type="dataset", allow_patterns=patterns)
    shards = sorted({s for p in patterns for s in glob(f"{root}/{p}")})

    index = []
    for shard in tqdm(shards, desc="Extracting"):
        table = pq.read_table(shard)
        images = table.column("image").combine_chunks().field("bytes")
        rows = table.drop_columns(["image"]).to_pylist()
        for row, image in zip(rows, images, strict=True):
            path = f"{args.out}/{row['path']}"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(image.as_py())
            index.append(row)

    pd.DataFrame(index).to_csv(f"{args.out}/index.csv", index=False)
    print(f"{len(index)} frames in {args.out}, index: {args.out}/index.csv")


if __name__ == "__main__":
    main()
