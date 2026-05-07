"""
DINOv2 ViT-L/14 with and without registers, via timm.

The timm model exposes:
  - `forward_features(x)` -> (B, 1+R+P, C) post-norm tokens (R=4 for reg variant)
  - `register_tokens` (Parameter, shape (1, 4, C)) when present
"""
from __future__ import annotations

import timm
import torch

from .base import RegisteredViT

_NO_REG = "vit_large_patch14_dinov2.lvd142m"
_WITH_REG = "vit_large_patch14_reg4_dinov2.lvd142m"


class DINOv2(RegisteredViT):
    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # timm's forward_features returns the full token sequence with norm applied.
        return self.backbone.forward_features(x)


def load_dinov2(with_registers: bool, *, img_size: int = 518) -> DINOv2:
    name = _WITH_REG if with_registers else _NO_REG
    backbone = timm.create_model(name, pretrained=True, img_size=img_size).eval()

    # Embed dim & patch size from the model itself.
    embed_dim = backbone.embed_dim
    patch_size = backbone.patch_embed.patch_size[0]
    grid = img_size // patch_size
    num_reg = getattr(backbone, "num_reg_tokens", 0) or 0
    if with_registers and num_reg != 4:
        raise RuntimeError(
            f"Expected 4 registers in {name}, got {num_reg}"
        )

    return DINOv2(
        backbone=backbone,
        embed_dim=embed_dim,
        num_registers=num_reg,
        patch_grid=(grid, grid),
        img_size=img_size,
        patch_size=patch_size,
        name=f"DINOv2-L/14{'+reg' if with_registers else ''}",
    )
