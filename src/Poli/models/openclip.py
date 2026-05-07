"""
OpenCLIP ViT-B/16, with and without test-time registers.

- no-reg: official `ViT-B-16 / laion2b_s34b_b88k` from open_clip.
- with-reg: HuggingFace `amildravid4292/clip-vitb16-test-time-registers`
  (Jiang et al. 2025 — test-time register injection, not training-time).
  Number of registers is a runtime parameter (0, 1, 4, 8, 16 supported).

Both expose the same interface via RegisteredViT. We hook the last
resblock output to capture the full token sequence
[CLS, REG..., patch_0, ..., patch_N] before pooling/projection.
"""
from __future__ import annotations

import open_clip
import torch
import torch.nn as nn
from transformers import AutoModel

from .base import RegisteredViT, ViTOutput

_NO_REG_NAME = ("ViT-B-16", "laion2b_s34b_b88k")
_WITH_REG_HF = "amildravid4292/clip-vitb16-test-time-registers"


class _OpenCLIPNoReg(RegisteredViT):
    def __init__(self, clip_model: nn.Module, *, img_size: int):
        patch_size = clip_model.visual.patch_size[0]
        grid = img_size // patch_size
        super().__init__(
            backbone=clip_model,
            embed_dim=768,
            num_registers=0,
            patch_grid=(grid, grid),
            img_size=img_size,
            patch_size=patch_size,
            name="OpenCLIP ViT-B/16 (laion2b)",
        )

    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        captured: dict[str, torch.Tensor] = {}

        def hook(_m, _inp, out):
            captured["tokens"] = out

        last = self.backbone.visual.transformer.resblocks[-1]
        h = last.register_forward_hook(hook)
        try:
            self.backbone.encode_image(x)
        finally:
            h.remove()
        return captured["tokens"]


class _OpenCLIPTestTimeReg(RegisteredViT):
    def __init__(self, hf_model: nn.Module, *, num_registers: int, img_size: int):
        v = hf_model.model.visual
        patch_size = v.patch_size[0] if isinstance(v.patch_size, tuple) else v.patch_size
        grid = img_size // patch_size
        self._n_reg_runtime = num_registers
        super().__init__(
            backbone=hf_model,
            embed_dim=768,
            num_registers=num_registers,
            patch_grid=(grid, grid),
            img_size=img_size,
            patch_size=patch_size,
            name=f"OpenCLIP ViT-B/16 + test-time-reg{num_registers}",
        )

    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        captured: dict[str, torch.Tensor] = {}

        def hook(_m, _inp, out):
            captured["tokens"] = out

        last = self.backbone.model.visual.transformer.resblocks[-1]
        h = last.register_forward_hook(hook)
        try:
            self.backbone.model.encode_image(
                x, num_register_tokens=self._n_reg_runtime
            )
        finally:
            h.remove()
        return captured["tokens"]

    def forward(self, x: torch.Tensor) -> ViTOutput:
        # Override base: HF test-time-registers uses layout
        # [CLS, patch_0, ..., patch_{P-1}, REG_0, ..., REG_{R-1}]
        # (registers appended at the end, not after CLS).
        tokens = self._forward_tokens(x)
        cls = tokens[:, 0]
        patches = tokens[:, 1 : 1 + self.num_patches]
        if self.num_registers > 0:
            reg = tokens[:, 1 + self.num_patches :]
        else:
            reg = None
        return ViTOutput(cls=cls, patches=patches, registers=reg)


def load_openclip(
    with_registers: bool, *, num_registers: int = 4, img_size: int = 224
):
    """Load OpenCLIP ViT-B/16 with or without test-time registers."""
    if with_registers:
        if num_registers not in (0, 1, 4, 8, 16):
            raise ValueError(
                f"num_registers={num_registers} unsupported by HF checkpoint; "
                "use one of 0, 1, 4, 8, 16."
            )
        m = AutoModel.from_pretrained(_WITH_REG_HF, trust_remote_code=True).eval()
        return _OpenCLIPTestTimeReg(m, num_registers=num_registers, img_size=img_size)
    name, pretrained = _NO_REG_NAME
    m, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    return _OpenCLIPNoReg(m.eval(), img_size=img_size)
