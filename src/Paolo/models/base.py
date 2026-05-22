"""
Common interface for ViT backbones (with or without registers).

Layout assumed at the output of the last LayerNorm:
    [CLS, REG_0, ..., REG_{R-1}, PATCH_0, ..., PATCH_{P-1}]
where R may be 0 for vanilla models.

Each subclass implements `_forward_tokens(x)` returning the full
(B, 1+R+P, C) post-norm token tensor; the base class slices it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ViTOutput:
    cls: torch.Tensor                    # (B, C)
    patches: torch.Tensor                # (B, P, C)
    registers: Optional[torch.Tensor]    # (B, R, C) or None


class RegisteredViT(nn.Module):
    """Wrapper that exposes CLS / patches / registers separately."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        embed_dim: int,
        num_registers: int,
        patch_grid: tuple[int, int],
        img_size: int,
        patch_size: int,
        name: str,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.num_registers = num_registers
        self.patch_grid = patch_grid
        self.num_patches = patch_grid[0] * patch_grid[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.name = name

    # Subclasses override this.
    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> ViTOutput:
        tokens = self._forward_tokens(x)
        # tokens: (B, 1 + R + P, C)
        cls = tokens[:, 0]
        if self.num_registers > 0:
            reg = tokens[:, 1 : 1 + self.num_registers]
            patches = tokens[:, 1 + self.num_registers :]
        else:
            reg = None
            patches = tokens[:, 1:]
        if patches.shape[1] != self.num_patches:
            raise RuntimeError(
                f"[{self.name}] expected {self.num_patches} patch tokens, "
                f"got {patches.shape[1]} (total tokens={tokens.shape[1]}, "
                f"registers={self.num_registers})"
            )
        return ViTOutput(cls=cls, patches=patches, registers=reg)

    @torch.no_grad()
    def extract_cls(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).cls
