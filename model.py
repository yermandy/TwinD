"""TwinD: PE-Core-L (timm) + LoRA on the attention projections + a linear real / fake classifier."""

import math
import os
import re

import timm
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn

HF_REPO = "yermandy/TwinD"  # the released model: TwinD.safetensors
BACKBONE = "vit_pe_core_large_patch14_336"
IMG_SIZE = 224
LORA_TARGETS = r"blocks\.\d+\.attn\.(qkv|proj)"


class LoRALinear(nn.Module):
    """y = x (W + s A B)^T + b with W frozen, A (out, r) = 0 and B (g r, in) ~ kaiming-uniform at init, s = alpha / r.

    g row-blocks get one adapter each: g = 3 on the fused qkv projection adapts W_q, W_k and W_v separately."""

    def __init__(self, linear: nn.Linear, r: int, alpha: float, g: int = 1):
        super().__init__()
        W = linear.weight.data
        self.A = nn.Parameter(torch.zeros(W.shape[0], r, dtype=W.dtype, device=W.device))
        self.B = nn.Parameter(torch.empty(g * r, W.shape[1], dtype=W.dtype, device=W.device))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        self.register_buffer("weight", W.clone())
        self.register_buffer("bias", None if linear.bias is None else linear.bias.data.clone())
        self.scale, self.r, self.g = alpha / r, r, g

    def merged_weight(self) -> torch.Tensor:
        A = self.A.view(self.g, -1, self.r)
        B = self.B.view(self.g, self.r, -1)
        return self.weight + self.scale * torch.bmm(A, B).reshape_as(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.merged_weight(), self.bias)

    def merge(self) -> nn.Linear:
        """The plain nn.Linear with the LoRA update folded into its weight."""
        out_features, in_features = self.weight.shape
        linear = nn.Linear(in_features, out_features, bias=self.bias is not None, device=self.weight.device)
        linear.weight.data.copy_(self.merged_weight().detach())
        if self.bias is not None:
            linear.bias.data.copy_(self.bias)
        return linear


class TwinD(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE, pretrained=pretrained, dynamic_img_size=True)
        self.backbone.head = nn.Identity()
        self.head = nn.Linear(self.backbone.num_features, 2)  # logits of (real, fake)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def preprocessing(self):
        """Bicubic resize to 224 x 224 and normalization to [-1, 1], PE-Core's own eval transform."""
        data_config = timm.data.resolve_model_data_config(self.backbone)
        data_config["input_size"] = (3, IMG_SIZE, IMG_SIZE)
        return timm.data.create_transform(**data_config, is_training=False)

    def add_lora(self, r: int = 32, alpha: float = 32):
        """Freezes the backbone and adds LoRA to W_q, W_k, W_v (fused qkv, 3 adapters) and W_o of every block."""
        self.backbone.requires_grad_(False)
        for name, module in list(self.backbone.named_modules()):
            if re.fullmatch(LORA_TARGETS, name):
                parent, child = name.rsplit(".", 1)
                g = 3 if child == "qkv" else 1
                self.backbone.get_submodule(parent).register_module(child, LoRALinear(module, r, alpha, g))

    def merge_lora(self):
        """Folds every LoRA update into its frozen weight, so the model is a plain PE-Core-L + linear head."""
        for name, module in list(self.backbone.named_modules()):
            if isinstance(module, LoRALinear):
                parent, child = name.rsplit(".", 1)
                self.backbone.get_submodule(parent).register_module(child, module.merge())

    def merged_state_dict(self) -> dict[str, torch.Tensor]:
        """state_dict of the model with LoRA folded in (fp32, on CPU), without changing the model."""
        state = {k: v.detach().float().cpu().clone() for k, v in self.state_dict().items()}
        for name, module in self.backbone.named_modules():
            if isinstance(module, LoRALinear):
                for p in ("A", "B"):
                    del state[f"backbone.{name}.{p}"]
                state[f"backbone.{name}.weight"] = module.merged_weight().detach().float().cpu()
        return state


def load_twind(weights: str = HF_REPO, device: str = "cuda", compile: bool = True) -> TwinD:
    """A trained TwinD: a .safetensors file written by train.py / soup.py, or the released model from the Hub.
    `compile`: torch.compile the backbone, as in training."""
    if not os.path.isfile(weights):
        weights = hf_hub_download(weights, "TwinD.safetensors")
    model = TwinD(pretrained=False)
    model.load_state_dict(load_file(weights))
    model.to(device).eval()
    if compile:
        model.backbone.forward = torch.compile(model.backbone.forward)
    return model
