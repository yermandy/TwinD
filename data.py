"""Training data of TwinD: pairs of a fake and the real frame it was made from, both with the same augmentation.

A pair is one of
- an FF++ fake (Deepfakes, Face2Face, FaceSwap, NeuralTextures) and its pixel-aligned FF++ real frame,
- a twin of a real FF++ / CCv2 frame, drawn every time the pair is used, half of the time each kind:
  - reconstruction twin: the autoencoder reconstruction with the real pixels put back inside a random mask,
  - blob twin: the real frame blended with an altered copy of itself inside a soft round blob.
"""

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import Dataset, Sampler

FF_METHODS = ["DF", "F2F", "FS", "NT"]
EPOCH_PAIRS = 34_434  # 6 x the 5,739 FF++ real frames
TWIN_SHARE = 2 / 3  # of the epoch; the FF++ fakes take the rest


# ---------------------------------------------------------------------------------------------------------- twins
def grid_mask(h: int, w: int, real_share: float, cells: int = 32) -> np.ndarray:
    """round(real_share * 32 * 32) random cells of a 32 x 32 grid set to 1 (real)."""
    n = cells**2
    grid = np.zeros(n, dtype=np.float32)
    grid[np.random.choice(n, int(round(real_share * n)), replace=False)] = 1.0
    grid = Image.fromarray((grid.reshape(cells, cells) * 255).astype(np.uint8))
    return np.asarray(grid.resize((w, h), Image.NEAREST), dtype=np.float32) / 255.0


def noise_mask(h: int, w: int, real_share: float, sigma: float) -> np.ndarray:
    """White noise blurred with a Gaussian of width `sigma` x the image side, thresholded to cover `real_share`."""
    smooth = cv2.GaussianBlur(np.random.rand(h, w).astype(np.float32), (0, 0), sigma * (h + w) / 2)
    return (smooth >= np.quantile(smooth, 1.0 - real_share)).astype(np.float32)


def reconstruction_twin(real: Image.Image, recon: Image.Image) -> Image.Image:
    """Eq. (1): the reconstruction with the real pixels put back inside a grid or noise mask covering U(0, 0.9)."""
    w, h = recon.size
    real_share = np.random.uniform(0.0, 0.9)
    if real_share <= 0.0:
        return recon
    if np.random.choice(2, p=[0.5, 0.5]) == 0:
        mask = grid_mask(h, w, real_share)
    else:
        mask = noise_mask(h, w, real_share, np.random.uniform(0.03, 0.09))
    m = mask[..., None]
    mixed = np.asarray(real, dtype=np.float32) * m + np.asarray(recon, dtype=np.float32) * (1.0 - m)
    return Image.fromarray(np.round(mixed).clip(0, 255).astype(np.uint8))


def blob_mask(h: int, w: int) -> Image.Image:
    """Soft disk 2^(-(d / rho)^8) centered at (50 %, 60 %) of the crop, covering U(8 %, 50 %) of it."""
    area = np.random.uniform(0.08, 0.5) * h * w
    rho = np.sqrt(area / np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - 0.5 * w) ** 2 + (yy - 0.6 * h) ** 2) / max(rho, 1.0)
    return Image.fromarray(np.round(255 * np.exp(-np.log(2) * d**8)).astype(np.uint8), mode="L")


ALTER_COLOR = T.Compose([T.ColorJitter(contrast=0.18), T.ColorJitter(hue=0.03)])
SHIFT = T.RandomAffine(degrees=0, translate=(0.018, 0.018))  # nearest, the same shift for the image and its mask


def alter(image: Image.Image) -> Image.Image:
    """Contrast and hue change, then a 2x nearest downscale and a bilinear upscale back."""
    img = np.asarray(ALTER_COLOR(image))
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR))


def blob_twin(real: Image.Image) -> Image.Image:
    """Eq. (2): F blended into B through s M_b, one of F / B the real frame and the other its altered copy."""
    w, h = real.size
    mask = blob_mask(h, w)
    if np.random.rand() < 0.5:
        foreground, background = alter(real), real
    else:
        foreground, background = real, alter(real)
    foreground, mask = SHIFT(foreground, mask)  # B is never shifted
    s = np.random.choice([0.3, 1.0])
    m = np.asarray(mask, dtype=np.float32)[..., None] * (s / 255.0)
    blended = np.asarray(foreground, dtype=np.float32) * m + np.asarray(background, dtype=np.float32) * (1.0 - m)
    return Image.fromarray(np.round(blended).clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------------------------------- augmentation
class RandomSharpen:
    """Unsharp mask I + a (I - G_sigma * I), a ~ U(0.5, 3), sigma ~ U(0.5, 1.5) px."""

    def __call__(self, img: Image.Image) -> Image.Image:
        a = 0.5 + 2.5 * torch.rand(1).item()
        s = 0.5 + 1.0 * torch.rand(1).item()
        x = np.asarray(img, np.float32)
        x = x + a * (x - cv2.GaussianBlur(x, (0, 0), s))
        return Image.fromarray(np.round(x).clip(0, 255).astype(np.uint8))


class RandomAffine:
    """T.RandomAffine whose interpolation is bicubic, bilinear or nearest with probabilities 0.45, 0.45, 0.1."""

    def __init__(self):
        modes = [T.InterpolationMode.BICUBIC, T.InterpolationMode.BILINEAR, T.InterpolationMode.NEAREST]
        self.weights = torch.tensor([0.45, 0.45, 0.1], dtype=torch.float64)
        self.affines = [T.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1), interpolation=m) for m in modes]

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.affines[int(torch.multinomial(self.weights, 1).item())](img)


class RandomOrder:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img: Image.Image) -> Image.Image:
        for i in torch.randperm(len(self.transforms)).tolist():
            img = self.transforms[i](img)
        return img


def augmentation():
    """Flip, then affine, blur, brightness / contrast jitter and sharpening in a random order, then JPEG 40-100."""
    return T.Compose(
        [
            T.RandomHorizontalFlip(0.5),
            RandomOrder(
                [
                    RandomAffine(),
                    T.RandomApply([T.GaussianBlur(7, sigma=(0.1, 2.0))], p=0.1),
                    T.ColorJitter(brightness=0.1, contrast=0.1),
                    RandomSharpen(),
                ]
            ),
            T.JPEG((40, 100)),
        ]
    )


# --------------------------------------------------------------------------------------------------------- dataset
class PairsDataset(Dataset):
    """Items are (pair index, augmentation seed, which): which = 0 for the real frame, 1 for the fake of the pair.

    `index`: data/index.csv written by prepare_data.py. `decoder`: the autoencoder of the reconstruction twins."""

    def __init__(self, root: str, index: str, preprocess, decoder: str = "cogvideox"):
        df = pd.read_csv(index, keep_default_na=False)
        fakes = df[df.method.isin([decoder, *FF_METHODS])]
        self.root, self.preprocess, self.augment = root, preprocess, augmentation()
        self.fake, self.real = fakes.path.tolist(), fakes.paired_real.tolist()
        self.is_twin = (fakes.method == decoder).to_numpy()
        self.method = fakes.method.tolist()

    def __len__(self):
        return len(self.fake)

    def load(self, path: str) -> Image.Image:
        return Image.open(f"{self.root}/{path}").convert("RGB")

    def __getitem__(self, item):
        i, seed, which = item
        real = self.load(self.real[i])
        if which == 0:
            image = real
        elif not self.is_twin[i]:
            image = self.load(self.fake[i])
        elif np.random.rand() < 0.5:
            image = reconstruction_twin(real, self.load(self.fake[i]))
        else:
            image = blob_twin(real)
        # Both frames of a pair get the same augmentation: the torchvision transforms draw from the torch RNG
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            image = self.augment(image)
        return self.preprocess(image), which


class PairsSampler(Sampler):
    """Every epoch: 34,434 pairs, 2/3 twin pairs and 1/3 FF++ fake pairs (re-sampled every epoch), shuffled; the
    real and the fake of a pair are consecutive, so a batch of 128 frames holds 64 pairs."""

    def __init__(self, dataset: PairsDataset, seed: int):
        self.rng = np.random.RandomState(seed)
        method = np.array(dataset.method)
        is_twin = dataset.is_twin
        w_twin = TWIN_SHARE * EPOCH_PAIRS / is_twin.sum()
        w_ff = (1 - TWIN_SHARE) * EPOCH_PAIRS / (~is_twin).sum()
        self.groups = [
            (np.flatnonzero(method == m), w_twin if is_twin[method == m][0] else w_ff) for m in sorted(set(method))
        ]
        self.length = 2 * sum(int(round(len(rows) * w)) for rows, w in self.groups)

    def __len__(self):
        return self.length

    def __iter__(self):
        epoch = []
        for rows, w in self.groups:
            n = int(round(len(rows) * w))
            epoch.append(np.concatenate([self.rng.permutation(rows) for _ in range(-(-n // len(rows)))])[:n])
        pairs = np.concatenate(epoch)
        self.rng.shuffle(pairs)
        seeds = self.rng.randint(0, 2**31 - 1, size=len(pairs))
        return iter([(int(i), int(s), which) for i, s in zip(pairs, seeds, strict=True) for which in (1, 0)])
