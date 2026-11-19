"""SCRFD-10GF face detector (det_10g of the InsightFace buffalo_l pack) as a TorchScript model, with its decoding."""

import os

import cv2
import numpy as np
import torch
import torchvision
from huggingface_hub import hf_hub_download

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

HF_REPO = "yermandy/TwinD"  # scrfd_10g.pth, next to the TwinD weights
STRIDES = [8, 16, 32]  # 3 feature levels, 2 anchors per location
NUM_ANCHORS = 2


class SCRFD:
    def __init__(
        self, weights: str = HF_REPO, det_thresh: float = 0.5, nms_thresh: float = 0.4, device: str | None = None
    ):
        """`weights`: a local TorchScript file or the Hub repo holding scrfd_10g.pth."""
        if not os.path.isfile(weights):
            weights = hf_hub_download(weights, "scrfd_10g.pth")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = torch.jit.load(weights, map_location=self.device).eval()
        self.det_thresh, self.nms_thresh = det_thresh, nms_thresh
        self.input_size = (640, 640)
        self.centers: dict[tuple[int, int, int], torch.Tensor] = {}

    def anchor_centers(self, height: int, width: int, stride: int) -> torch.Tensor:
        key = (height, width, stride)
        if key not in self.centers:
            ys, xs = torch.meshgrid(
                torch.arange(height, dtype=torch.float32, device=self.device),
                torch.arange(width, dtype=torch.float32, device=self.device),
                indexing="ij",
            )
            centers = (torch.stack([xs, ys], dim=-1) * stride).reshape(-1, 2)
            self.centers[key] = centers.unsqueeze(1).expand(-1, NUM_ANCHORS, -1).reshape(-1, 2)
        return self.centers[key]

    @torch.inference_mode()
    def forward(self, img: np.ndarray):
        """Scores, boxes and 5 keypoints of every anchor above the detection threshold."""
        size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128.0, size, (127.5, 127.5, 127.5), swapRB=True)
        outs = self.net(torch.from_numpy(blob).to(self.device, non_blocking=True))
        scores, boxes, kpss = [], [], []
        for i, stride in enumerate(STRIDES):
            centers = self.anchor_centers(blob.shape[2] // stride, blob.shape[3] // stride, stride)
            score = outs[i].reshape(-1)
            box = (outs[i + 3] * stride).reshape(-1, 4)
            kps = (outs[i + 6] * stride).reshape(-1, 10)
            keep = score >= self.det_thresh
            scores.append(score[keep].unsqueeze(-1))
            boxes.append(torch.cat([centers - box[:, :2], centers + box[:, 2:]], dim=-1)[keep])
            kpss.append(torch.stack([centers[:, :1] + kps[:, 0::2], centers[:, 1:2] + kps[:, 1::2]], dim=-1)[keep])
        return torch.cat(scores), torch.cat(boxes), torch.cat(kpss)

    @torch.inference_mode()
    def detect(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Faces in a BGR image: (N, 5) boxes x1, y1, x2, y2, score and (N, 5, 2) keypoints (eyes, nose, mouth)."""
        w, h = self.input_size
        if img.shape[0] / img.shape[1] > h / w:
            new_h, new_w = h, int(h / (img.shape[0] / img.shape[1]))
        else:
            new_w, new_h = w, int(w * img.shape[0] / img.shape[1])
        scale = new_h / img.shape[0]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(img, (new_w, new_h))

        scores, boxes, kpss = self.forward(canvas)
        boxes, kpss = boxes / scale, kpss / scale
        keep = torchvision.ops.nms(boxes, scores.squeeze(-1), self.nms_thresh)
        det = torch.cat([boxes[keep], scores[keep]], dim=-1).cpu().numpy()
        return det, kpss[keep].cpu().numpy()
