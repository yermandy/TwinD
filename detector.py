"""Face crops for TwinD: detect the face with SCRFD-10GF, align it with a similarity transform on its 5 keypoints,
enlarge the face box 1.3x and crop it at about its original size.

    python detector.py -i videos/ -o faces/              # up to 32 evenly spaced frames per video
    python detector.py -i video.mp4 -o faces/
    python detector.py -i list.txt -o faces/             # a text file with one video / image path per line

Output: <output>/<relative path of the video with "/" -> "--">/frame_XXXX.png (images: <name>.png).
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import cv2
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm

from scrfd import SCRFD

EXTENSIONS = ("mp4", "avi", "mov", "mkv", "webm", "jpg", "jpeg", "png")


def video_frames(path: str, num_frames: int):
    """Up to `num_frames` evenly spaced frames (BGR) with their indices, all frames if num_frames = -1."""
    try:
        total = iio.improps(path, plugin="pyav").shape[0]
    except Exception as e:
        print(f"Warning: Video {path} cannot be opened! {e}")
        return
    ids = np.arange(total) if num_frames == -1 else np.linspace(0, total - 1, num_frames, endpoint=True, dtype=int)
    for frame_id in ids:
        try:
            yield cv2.cvtColor(iio.imread(path, index=frame_id, plugin="pyav"), cv2.COLOR_RGB2BGR), frame_id
        except Exception as e:
            print(f"Warning: Failed to read frame {frame_id} of {path}. Skipping. {e}")


def align_face(img: np.ndarray, landmarks: np.ndarray, target_size: None | tuple = None, scale: float = 1.3):
    """Similarity transform of the 5 keypoints onto a template, the face box enlarged `scale` x. With no
    `target_size`, the crop keeps the face at about its size in the frame."""
    dst = np.array([[0.34, 0.46], [0.66, 0.46], [0.5, 0.64], [0.37, 0.82], [0.63, 0.82]], dtype=np.float32)

    if target_size is None:
        # Mean ratio of the keypoint distances in the frame and in the template
        iu = np.triu_indices(len(dst), k=1)
        src_dists = np.linalg.norm(landmarks[:, None] - landmarks[None], axis=-1)[iu]
        dst_dists = np.linalg.norm(dst[:, None] - dst[None], axis=-1)[iu]
        size = np.round(np.mean(src_dists / dst_dists) * scale).astype(int)
        target_size = (size, size)

    # Template in pixels, shifted by the margin and scaled back so the face box is enlarged `scale` x
    margin = scale - 1
    for axis in (0, 1):
        m = target_size[axis] * margin / 2.0
        dst[:, axis] *= target_size[axis]
        dst[:, axis] += m
        dst[:, axis] *= target_size[axis] / (target_size[axis] + 2 * m)

    M = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst, method=cv2.LMEDS)[0]
    return cv2.warpAffine(img, M, target_size, flags=cv2.INTER_LINEAR)


def largest_face(model: SCRFD, frame: np.ndarray, min_face_size: int) -> None | np.ndarray:
    """Keypoints of the largest detected face, None if there is none (of at least `min_face_size` px)."""
    xyxy, landmarks = model.detect(frame)
    if min_face_size > 0 and len(xyxy):
        keep = np.minimum(xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]) >= min_face_size
        xyxy, landmarks = xyxy[keep], landmarks[keep]
    if len(xyxy) == 0:
        return None
    return landmarks[np.argmax((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]))]


def process(path: str, output: str, model: SCRFD, args):
    if path.lower().endswith(("jpg", "jpeg", "png")):
        frames = [(cv2.cvtColor(iio.imread(path), cv2.COLOR_RGB2BGR), None)]
    else:
        frames = video_frames(path, args.num_frames)
    num_saved = 0
    for frame, frame_id in frames:
        landmarks = largest_face(model, frame, args.min_face_size)
        if landmarks is None:
            print(f"No face in frame {frame_id} of {path}")
            continue
        name = f"frame_{frame_id:04d}.png" if frame_id is not None else os.path.basename(output) + ".png"
        out_dir = output if frame_id is not None else os.path.dirname(output)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(f"{out_dir}/{name}", align_face(frame, landmarks, args.target_size, args.scale))
        num_saved += 1
    if num_saved == 0:
        print(f"No faces were saved from {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="A video / image, a folder of them or a .txt list")
    parser.add_argument("-o", "--output", default="faces")
    parser.add_argument("-n", "--num_frames", type=int, default=32, help="Evenly spaced frames per video, -1: all")
    parser.add_argument("-s", "--scale", type=float, default=1.3, help="Enlargement of the face box")
    parser.add_argument("--target_size", default="none", help="'width,height' of the crops, 'none': original size")
    parser.add_argument("--det_thres", type=float, default=0.5)
    parser.add_argument("--min_face_size", type=int, default=0, help="Drop faces with a shorter box side, 0: off")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    args.target_size = None if args.target_size.lower() == "none" else tuple(map(int, args.target_size.split(",")))

    if os.path.isdir(args.input):
        root = args.input
        files = sorted(f for e in EXTENSIONS for f in glob(f"{root}/**/*.{e}", recursive=True))
    elif args.input.endswith(".txt"):
        root = os.path.dirname(args.input)
        files = sorted(open(args.input).read().splitlines())
    else:
        root, files = os.path.dirname(args.input), [args.input]

    model = SCRFD(det_thresh=args.det_thres)

    def output_dir(path: str) -> str:
        return f"{args.output}/{os.path.splitext(os.path.relpath(path, root))[0].replace(os.sep, '--')}"

    with ThreadPoolExecutor(args.num_workers) as executor:
        futures = [executor.submit(process, f, output_dir(f), model, args) for f in files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Detecting faces"):
            future.result()


if __name__ == "__main__":
    main()
