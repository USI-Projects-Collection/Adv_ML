"""
DINOv2-reg4 with a runtime-configurable number of register tokens, used for the
Figure 8 ablation N ∈ {0, 1, 2, 4, 8, 16}.

Strategy
--------
We always start from `vit_large_patch14_reg4_dinov2.lvd142m` (4 trained registers)
and modify the registers in place so the *physical* sequence length matches the
desired N:

  - N == 4 : original model, no patch.
  - N <  4 : truncate `model.reg_token` to the first N entries; update
             `model.num_prefix_tokens = 1 + N` so timm's `_pos_embed` slices
             correctly.
  - N >  4 : pad `model.reg_token` with copies of the trained registers
             (cycle them) so the rest of the network — which has only ever seen
             4 trained registers — is exposed to inputs in roughly the same
             distribution. Untrained random init produced strong artifacts in
             initial tests.

Note: this is *not* the same as retraining DINOv2 with N registers (what the
paper does for Figure 8). It is the closest no-retraining proxy. Document this
honestly in the report.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from models.base import RegisteredViT


_NAME = "vit_large_patch14_reg4_dinov2.lvd142m"


class DINOv2Sweep(RegisteredViT):
    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)


def _set_registers(model: nn.Module, n: int) -> None:
    """Resize model.reg_token in place to N entries."""
    embed_dim = model.embed_dim
    base = model.reg_token.detach()  # (1, 4, embed_dim) — the trained registers
    base_n = base.shape[1]
    if n == 0:
        # Drop register tokens entirely. timm checks reg_token is not None
        # before concatenating, so we set it to None.
        model.reg_token = None
    elif n <= base_n:
        new = base[:, :n].clone()
        model.reg_token = nn.Parameter(new, requires_grad=False)
    else:
        # Cycle the 4 trained registers to fill N slots.
        idx = torch.arange(n) % base_n
        new = base[:, idx, :].clone()
        model.reg_token = nn.Parameter(new, requires_grad=False)
    # timm's vision transformer reads num_prefix_tokens for pos-embed slicing.
    if hasattr(model, "num_prefix_tokens"):
        model.num_prefix_tokens = 1 + n


def load_dinov2_with_n_registers(num_registers: int, *, img_size: int = 518) -> DINOv2Sweep:
    if num_registers < 0 or num_registers > 16:
        raise ValueError(f"num_registers must be in 0..16, got {num_registers}")

    backbone = timm.create_model(_NAME, pretrained=True, img_size=img_size).eval()
    _set_registers(backbone, num_registers)

    embed_dim = backbone.embed_dim
    patch_size = backbone.patch_embed.patch_size[0]
    grid = img_size // patch_size

    return DINOv2Sweep(
        backbone=backbone,
        embed_dim=embed_dim,
        num_registers=num_registers,
        patch_grid=(grid, grid),
        img_size=img_size,
        patch_size=patch_size,
        name=f"DINOv2-L/14 +reg{num_registers} (sweep)",
    )
