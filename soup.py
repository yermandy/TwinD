"""Uniform model soup: the mean of the weights of TwinD models trained with different seeds.

python soup.py runs/seed1/TwinD.safetensors runs/seed2/TwinD.safetensors runs/seed3/TwinD.safetensors
"""

import argparse

from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", nargs="+")
    parser.add_argument("--out", default="TwinD.safetensors")
    args = parser.parse_args()

    soup = None
    for path in args.weights:
        state = load_file(path)
        soup = state if soup is None else {k: soup[k] + state[k] for k in soup}
    save_file({k: v / len(args.weights) for k, v in soup.items()}, args.out)
    print(f"Saved {args.out}, the mean of {len(args.weights)} models")


if __name__ == "__main__":
    main()
