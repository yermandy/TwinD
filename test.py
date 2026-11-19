"""Test TwinD on face crops (made by detector.py): the video score is the mean fake probability of its frames,
the metric is the video-level AUROC.

    python test.py --data example                                  # the released model from the Hub
    python test.py --data example --weights runs/seed1/TwinD.safetensors

<data>/real/<video>/<frame>.png and <data>/fake/<video>/<frame>.png. Writes <data>/frame_level_predictions.csv
(path, p_fake, label) and <data>/video_level_predictions.csv (video, label, p_fake).
"""

import argparse
import time
from glob import glob

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import load_twind


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


class Frames(Dataset):
    def __init__(self, files: list[str], preprocess):
        self.files, self.preprocess = files, preprocess

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.preprocess(Image.open(self.files[index]).convert("RGB"))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yermandy/TwinD", help="A .safetensors file or the Hub model")
    parser.add_argument("--data", default="example")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_compile", action="store_true", help="Skip torch.compile (faster start on few frames)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    model = load_twind(args.weights, device, compile=not args.no_compile)

    files = sorted(glob(f"{args.data}/real/*/*.png") + glob(f"{args.data}/fake/*/*.png"))
    loader = DataLoader(Frames(files, model.preprocessing()), args.batch_size, num_workers=args.num_workers)
    probs = []
    model_time, model_frames = 0.0, 0  # forward passes only, after the first batch (compilation, warm-up)
    start = time.perf_counter()
    for i, images in enumerate(tqdm(loader, desc="Testing")):
        images = images.to(device)
        sync(device)
        t = time.perf_counter()
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            p = model(images).float().softmax(1)[:, 1]
        sync(device)
        if i > 0:
            model_time, model_frames = model_time + time.perf_counter() - t, model_frames + len(images)
        probs.append(p.cpu())
    total = time.perf_counter() - start

    frames = pd.DataFrame({"path": files, "p_fake": torch.cat(probs).numpy()})
    frames["label"] = frames.path.str.split("/").str[-3].map({"real": 0, "fake": 1})
    video = frames.path.str.rsplit("/", n=1).str[0]  # the frame folder
    videos = (
        frames.groupby(video.rename("video")).agg(label=("label", "first"), p_fake=("p_fake", "mean")).reset_index()
    )
    frames.to_csv(f"{args.data}/frame_level_predictions.csv", index=False, float_format="%.5f")
    videos.to_csv(f"{args.data}/video_level_predictions.csv", index=False, float_format="%.5f")

    print(videos.round(5).to_string(index=False))
    print(f"\nVideo-level AUROC: {100 * roc_auc_score(videos.label, videos.p_fake):.1f} % ({len(videos)} videos)")
    speed = f"{model_frames / model_time:.0f} frames/s (model)" if model_frames else "n/a (model, needs > 1 batch)"
    print(f"Inference speed: {speed}, {len(files) / total:.0f} frames/s (end to end, batch {args.batch_size})")


if __name__ == "__main__":
    main()
