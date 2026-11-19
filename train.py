"""Train one TwinD model (one seed): 10 epochs, Adam 3e-4 after a 1-epoch warm-up, L2-SP, SWA of the last 5 epochs.

    python train.py --seed 1 --out runs/seed1
    python train.py --seed 1 --decoder ltx2 --out runs/ltx2-seed1    # reconstruction twins of another autoencoder

Writes <out>/TwinD.safetensors: the SWA model with LoRA folded in (plain PE-Core-L + linear head).
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import PairsDataset, PairsSampler
from model import TwinD

LR, MIN_LR = 3e-4, 1e-5
EPOCHS, SWA_START = 10, 5  # SWA over the epoch-end weights of epochs 6-10
L2SP = 1.0
BATCH_SIZE = 128  # 64 pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default="data", help="Folder written by prepare_data.py")
    parser.add_argument(
        "--decoder", default="cogvideox", choices=["cogvideox", "kvae2d", "ltxvideo", "hunyuanvideo", "ltx2"]
    )
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--no_compile", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = "cuda"

    model = TwinD()
    model.add_lora(r=32, alpha=32)
    model.to(device).train()
    if not args.no_compile:
        model.backbone.forward = torch.compile(model.backbone.forward)

    dataset = PairsDataset(args.data, f"{args.data}/index.csv", model.preprocessing(), args.decoder)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=PairsSampler(dataset, args.seed),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]  # LoRA factors and the classifier
    optimizer = torch.optim.Adam(params, lr=LR)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=MIN_LR / LR, total_iters=len(loader))
    # L2-SP pulls the LoRA factors toward their initial values, the classifier is left alone
    lora = [p for n, p in model.named_parameters() if n.endswith((".A", ".B"))]
    anchors = [p.detach().clone() for p in lora]

    swa, n_swa = None, 0
    for epoch in range(EPOCHS):
        for images, labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(images).float(), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                lr = optimizer.param_groups[0]["lr"]
                for p, p0 in zip(lora, anchors, strict=True):
                    p.sub_(p - p0, alpha=lr * L2SP)
            optimizer.step()
            warmup.step()

        if epoch >= SWA_START:
            # Running mean of the epoch-end weights, LoRA folded in (the mean of the products A B is not the product
            # of the means)
            weights = model.merged_state_dict()
            n_swa += 1
            swa = weights if swa is None else {k: swa[k] + (w - swa[k]) / n_swa for k, w in weights.items()}

    os.makedirs(args.out, exist_ok=True)
    save_file(swa, f"{args.out}/TwinD.safetensors")
    print(f"Saved {args.out}/TwinD.safetensors")


if __name__ == "__main__":
    main()
