"""
DeiT-III ViT-B/16, with optional INJECTED (non-trained) register tokens.

The Darcet et al. paper trained their own DeiT-III+reg checkpoint and never
released it publicly. As a fallback we start from the official no-reg DeiT-III
and *inject* `num_registers` extra tokens between [CLS] and the patch tokens
at inference time. The register parameters are NOT trained — they are
initialised randomly with the same scheme used for [CLS] (truncated normal,
std 0.02). This is honest about its limitations:

  - the rest of the network has not been trained to use registers, so we do
    NOT expect to see the artifact-removal effect that the paper reports;
  - this is reported in the write-up as the weakest of the three rows;
  - it gives us a "with-reg" data point for completeness.

Implementation note: timm's vision_transformer factors the CLS / register /
patch concatenation inside `_pos_embed`. We monkey-patch that method so the
model accepts an arbitrary number of registers without breaking position
embedding interpolation.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .base import RegisteredViT

_NAME = "deit3_base_patch16_224.fb_in22k_ft_in1k"


class _DeiT3(RegisteredViT):
    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # timm forward_features applies the final norm and returns the
        # full (B, 1+R+P, C) sequence.
        return self.backbone.forward_features(x)


def _inject_registers(model: nn.Module, num_registers: int) -> None:
    """
    Add `num_registers` learnable-but-untrained register tokens to a timm
    ViT model that was trained without registers. Modifies the model in place.

    timm's `_pos_embed` does:
        x = patch_embed(x)
        if cls_token: x = cat([cls_token, x], dim=1)
        if reg_token: x = cat([cls_token, reg_token, patches], dim=1)
        x = x + pos_embed[:, :num_prefix_tokens] + interpolated_pos_embed_for_patches
    so we just need to provide a `reg_token` Parameter and update
    `num_prefix_tokens` from 1 to 1+num_registers.
    """
    if num_registers <= 0:
        return

    embed_dim = model.embed_dim
    reg = torch.empty(1, num_registers, embed_dim)
    nn.init.trunc_normal_(reg, std=0.02)
    model.reg_token = nn.Parameter(reg, requires_grad=False)

    # Patch the prefix-token bookkeeping used by `_pos_embed`.
    if hasattr(model, "num_prefix_tokens"):
        model.num_prefix_tokens = 1 + num_registers


def load_deit3(with_registers: bool, *, num_registers: int = 4):
    """
    Load DeiT-III ViT-B/16. If `with_registers=True`, inject `num_registers`
    untrained register tokens (default: 4 to match the paper's reg count).
    """
    model = timm.create_model(_NAME, pretrained=True).eval()

    if with_registers:
        _inject_registers(model, num_registers)
        n_reg = num_registers
        label = f"DeiT-III ViT-B/16 + reg{num_registers} (untrained, injected)"
    else:
        n_reg = 0
        label = "DeiT-III ViT-B/16 (no-reg)"

    img_size = 224
    patch_size = model.patch_embed.patch_size[0]
    grid = img_size // patch_size

    return _DeiT3(
        backbone=model,
        embed_dim=model.embed_dim,
        num_registers=n_reg,
        patch_grid=(grid, grid),
        img_size=img_size,
        patch_size=patch_size,
        name=label,
    )
