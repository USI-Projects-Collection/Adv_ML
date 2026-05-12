"""
Test-time registers for timm DINOv2-L/14 (no-reg baseline).

Re-implementation of Jiang et al. (2025) "Vision Transformers Don't Need
Trained Registers" — Algorithm 1 + Section 4.

Differences from the official repo (nickjiang2378/test-time-registers):
- They instrument Meta's native DINOv2 (SwiGLU MLP). We use timm DINOv2-L
  (GELU MLP), so their precomputed neuron indices are *not* reusable.
- We re-run the neuron-finding pass on timm DINOv2-L using the same logic
  (find neurons whose absolute activation at outlier patch positions is
  consistently high across images).
- Their convention places test-time registers at the END of the sequence:
  [CLS, PATCH..., TT_REG...]. We follow the same convention internally.

Hyperparameters (from their configs/dinov2_large.yaml):
- detect_outliers_layer = -2 (second-to-last block output for norm check)
- register_norm_threshold = 150  (patches with output norm > 150 are outliers)
- highest_layer = 17 (look for register neurons in MLP of blocks 0..17)
- top_k = 50

We hook the MLP at the GELU output (`block.mlp.act`), which is the post-GELU
neuron activation map (B, N, 4096) for DINOv2-L.

API
---
- find_register_neurons(images, *, top_k=50, ...) -> list[(layer, idx)]
- load_dinov2_with_tt_registers(num_registers, neurons, ...) -> RegisteredViT
- TTRegDINOv2.forward(x) returns ViTOutput where `.registers` is the slice of
  TT register tokens (last N positions of the post-norm sequence).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.base import RegisteredViT, ViTOutput


_TIMM_NAME = "vit_large_patch14_dinov2.lvd142m"

# Defaults transcribed from configs/dinov2_large.yaml in nickjiang2378/test-time-registers.
DEFAULT_DETECT_OUTLIERS_LAYER = -2
DEFAULT_NORM_THRESHOLD = 150.0
DEFAULT_HIGHEST_LAYER = 17  # inclusive
DEFAULT_TOP_K = 50


# -----------------------------------------------------------------------------
# Algorithm 1 — find register neurons on timm DINOv2-L/14
# -----------------------------------------------------------------------------
@torch.no_grad()
def find_register_neurons(
    images: Iterable[torch.Tensor],
    *,
    img_size: int = 518,
    detect_outliers_layer: int = DEFAULT_DETECT_OUTLIERS_LAYER,
    norm_threshold: float = DEFAULT_NORM_THRESHOLD,
    highest_layer: int = DEFAULT_HIGHEST_LAYER,
    top_k: int = DEFAULT_TOP_K,
    device: str = "cpu",
    verbose: bool = True,
) -> list[tuple[int, int, float]]:
    """
    Run a forward pass per image on baseline DINOv2-L/14, collect:
      - per-block output norms (residual stream) at `detect_outliers_layer`
        to detect outlier patch positions.
      - per-block, per-neuron post-GELU activations at all blocks 0..highest_layer.
    Average the absolute activation at outlier positions for each
    (layer, neuron) and return the top_k pairs.

    Returns
    -------
    list of (layer_idx, neuron_idx, score) sorted by score descending,
    length == top_k.
    """
    model = timm.create_model(_TIMM_NAME, pretrained=True, img_size=img_size).eval()
    model = model.to(device)
    num_blocks = len(model.blocks)
    if highest_layer >= num_blocks:
        raise ValueError(f"highest_layer={highest_layer} but model has {num_blocks} blocks")

    # Storage for hooks (per-forward).
    block_outputs: list[torch.Tensor] = []     # per-block output of the residual stream
    neuron_acts: list[torch.Tensor] = []       # per-block post-GELU activations

    hooks = []
    def make_block_hook(idx):
        def hook(_m, _inp, out):
            # out: (B, N, C); we'll need norm at detect_outliers_layer
            block_outputs.append(out.detach())
        return hook

    def make_mlp_act_hook(idx):
        def hook(_m, _inp, out):
            # out: (B, N, 4096); post-GELU
            neuron_acts.append(out.detach())
        return hook

    for i, blk in enumerate(model.blocks):
        hooks.append(blk.register_forward_hook(make_block_hook(i)))
        if i <= highest_layer:
            hooks.append(blk.mlp.act.register_forward_hook(make_mlp_act_hook(i)))

    try:
        num_neurons = model.blocks[0].mlp.fc1.out_features  # 4096
        # accumulators: per (layer, neuron) sum over outlier positions, and counts
        score_sum = torch.zeros(highest_layer + 1, num_neurons, device=device)
        score_count = 0  # number of images that contributed (had at least one outlier)

        for img_idx, x in enumerate(images):
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = x.to(device)
            block_outputs.clear()
            neuron_acts.clear()
            _ = model(x)

            # Detect outlier patch positions: use norms at detect_outliers_layer.
            block_out = block_outputs[detect_outliers_layer]   # (B, N, C)
            # token layout for baseline DINOv2: [CLS, PATCH...] (no registers)
            num_prefix = model.num_prefix_tokens
            patch_tokens = block_out[0, num_prefix:]            # (P, C)
            norms = patch_tokens.norm(dim=-1)                   # (P,)
            outlier_pos = (norms > norm_threshold).nonzero(as_tuple=True)[0]
            if outlier_pos.numel() == 0:
                if verbose and img_idx < 5:
                    print(
                        f"  [neuron_search] img {img_idx}: no outlier patches "
                        f"(max norm {norms.max().item():.1f})"
                    )
                continue

            # For each layer up to highest_layer, take abs activations at outlier
            # positions and average.
            for layer_idx in range(highest_layer + 1):
                acts = neuron_acts[layer_idx][0]                # (N, 4096)
                acts_patch = acts[num_prefix:]                  # (P, 4096)
                outlier_acts = acts_patch[outlier_pos].abs()   # (#outliers, 4096)
                score_sum[layer_idx] += outlier_acts.mean(dim=0)

            score_count += 1
            if verbose and (img_idx + 1) % 25 == 0:
                print(f"  [neuron_search] processed {img_idx + 1} images "
                      f"({score_count} with outliers)")
    finally:
        for h in hooks:
            h.remove()

    if score_count == 0:
        raise RuntimeError(
            f"No outlier patches found across all images (threshold={norm_threshold}). "
            f"Try lowering norm_threshold."
        )

    mean_score = score_sum / score_count  # (L, 4096)
    flat = mean_score.flatten()
    topv, topi = flat.topk(top_k)
    results: list[tuple[int, int, float]] = []
    for v, i in zip(topv.tolist(), topi.tolist()):
        layer = i // num_neurons
        neuron = i % num_neurons
        results.append((layer, neuron, v))
    if verbose:
        print(f"  [neuron_search] top-{top_k} register neurons "
              f"(scored on {score_count} images):")
        for layer, neuron, score in results[:10]:
            print(f"    layer={layer:2d} neuron={neuron:4d} score={score:.3f}")
        if top_k > 10:
            print(f"    ... + {top_k - 10} more")
    return results


# -----------------------------------------------------------------------------
# Test-time registers model
# -----------------------------------------------------------------------------
class TTRegDINOv2(RegisteredViT):
    """
    DINOv2-L/14 baseline with N test-time register tokens appended at the END of
    the sequence: [CLS, PATCH_0...PATCH_P, TT_REG_0...TT_REG_{N-1}].

    During forward we:
      1. Patch-embed + pos-embed normally.
      2. Concatenate N learned-zero TT register tokens at the end.
      3. Run all 24 blocks.
      4. For each (layer, neuron) in `register_neurons` and layer <= highest_layer,
         a forward hook on `block.mlp.act` edits the post-GELU activation:
           - copies max(|patch activation|) onto each of the N TT register tokens,
           - zeros the activation at all patch positions for that neuron.
        This causes the register neurons to write into TT registers instead of
        leaving high-norm artifacts on patch tokens.
    """

    def __init__(self, *, num_registers: int, register_neurons, img_size: int, **kw):
        # Build the underlying timm model.
        backbone = timm.create_model(_TIMM_NAME, pretrained=True, img_size=img_size).eval()
        embed_dim = backbone.embed_dim
        patch_size = backbone.patch_embed.patch_size[0]
        grid = img_size // patch_size

        super().__init__(
            backbone=backbone,
            embed_dim=embed_dim,
            num_registers=num_registers,
            patch_grid=(grid, grid),
            img_size=img_size,
            patch_size=patch_size,
            name=f"DINOv2-L/14 +tt-reg{num_registers} (Jiang)",
        )
        # zero-init test-time register tokens. requires_grad=False — pure inference.
        self.tt_registers = nn.Parameter(
            torch.zeros(1, num_registers, embed_dim), requires_grad=False
        )
        # Group neurons by layer for efficient hooking.
        neurons_by_layer: dict[int, list[int]] = {}
        for layer, neuron, _score in register_neurons:
            neurons_by_layer.setdefault(layer, []).append(neuron)
        self._neurons_by_layer = neurons_by_layer
        self._hook_handles = []
        self._install_hooks()

    def _install_hooks(self):
        # Remove old hooks (e.g. if re-initialised).
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
        n_reg = self.num_registers
        if n_reg == 0:
            return  # nothing to edit; model is just baseline
        for layer, neuron_list in self._neurons_by_layer.items():
            neuron_idx = torch.tensor(neuron_list, dtype=torch.long)
            blk = self.backbone.blocks[layer]
            def hook(_m, _inp, output, neuron_idx=neuron_idx, n_reg=n_reg):
                # output: (B, N, 4096) post-GELU
                # Patches are everything between prefix and the last n_reg tokens.
                num_prefix = self.backbone.num_prefix_tokens  # 1 (CLS only on baseline)
                # Compute max(|.|) across patch positions for each selected neuron.
                patch_slice = output[:, num_prefix : output.shape[1] - n_reg, :]   # (B, P, 4096)
                sel = patch_slice[..., neuron_idx.to(output.device)]  # (B, P, K)
                # signed max — preserve sign of largest magnitude
                abs_sel = sel.abs()
                argmax = abs_sel.argmax(dim=1, keepdim=True)  # (B, 1, K)
                max_val = sel.gather(1, argmax).squeeze(1)    # (B, K)
                # Write onto the last n_reg tokens for these neurons.
                for r in range(n_reg):
                    output[:, -n_reg + r, neuron_idx.to(output.device)] = max_val
                # Zero out the same neurons at all patch positions.
                output[:, num_prefix : output.shape[1] - n_reg, neuron_idx.to(output.device)] = 0
                return output
            self._hook_handles.append(blk.mlp.act.register_forward_hook(hook))

    # We need to inject TT registers into the sequence after _pos_embed.
    # We do this by monkey-patching forward_features in a wrapper.
    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.patch_embed(x)
        x = b._pos_embed(x)
        x = b.patch_drop(x)
        x = b.norm_pre(x)
        # Append TT registers at the end of the sequence.
        if self.num_registers > 0:
            tt = self.tt_registers.expand(x.shape[0], -1, -1).to(x.dtype).to(x.device)
            x = torch.cat([x, tt], dim=1)
        for blk in b.blocks:
            x = blk(x)
        x = b.norm(x)
        return x  # (B, 1 + P + N, C)

    def forward(self, x: torch.Tensor) -> ViTOutput:
        """Override base class slicing — our layout is [CLS, PATCH, TT_REG]."""
        tokens = self._forward_tokens(x)
        cls = tokens[:, 0]
        # Patches are between CLS and the trailing TT registers.
        if self.num_registers > 0:
            patches = tokens[:, 1 : 1 + self.num_patches]
            reg = tokens[:, 1 + self.num_patches :]
        else:
            patches = tokens[:, 1:]
            reg = None
        if patches.shape[1] != self.num_patches:
            raise RuntimeError(
                f"[{self.name}] expected {self.num_patches} patch tokens, got {patches.shape[1]}"
            )
        return ViTOutput(cls=cls, patches=patches, registers=reg)


# -----------------------------------------------------------------------------
# Public factory
# -----------------------------------------------------------------------------
def load_dinov2_with_tt_registers(
    num_registers: int,
    register_neurons: Sequence[tuple[int, int, float]],
    *,
    img_size: int = 518,
) -> TTRegDINOv2:
    """
    Build a DINOv2-L/14 baseline with N test-time registers and the given
    list of (layer, neuron, score) register-neuron tuples.

    Note: when num_registers == 0 we return a plain baseline (no hooks).
    """
    return TTRegDINOv2(
        num_registers=num_registers,
        register_neurons=register_neurons,
        img_size=img_size,
    )


# -----------------------------------------------------------------------------
# Helper: load images for find_register_neurons from the ImageNet subset
# -----------------------------------------------------------------------------
def load_neuron_search_images(n: int = 100, img_size: int = 518) -> list[torch.Tensor]:
    """Return n preprocessed image tensors from the ImageNet train subset."""
    import torchvision.transforms as T
    from data_loaders.imagenet import ImageNetSubset

    tfm = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    ds = ImageNetSubset("train", transform=tfm)
    out = []
    for i in range(min(n, len(ds))):
        img, _ = ds[i]
        out.append(img)
    return out


# -----------------------------------------------------------------------------
# Caching of neuron list to disk
# -----------------------------------------------------------------------------
def cached_register_neurons(
    cache_path: Path,
    *,
    n_images: int = 100,
    top_k: int = DEFAULT_TOP_K,
    device: str = "cpu",
    img_size: int = 518,
    overwrite: bool = False,
) -> list[tuple[int, int, float]]:
    """Find register neurons (or load from cache)."""
    if cache_path.exists() and not overwrite:
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(data, list) and len(data) >= top_k:
            print(f"[tt-reg] loaded {len(data)} neurons from {cache_path}")
            return data[:top_k]
    print(f"[tt-reg] computing register neurons on {n_images} images...")
    images = load_neuron_search_images(n=n_images, img_size=img_size)
    neurons = find_register_neurons(
        images,
        img_size=img_size,
        top_k=top_k,
        device=device,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(neurons, cache_path)
    print(f"[tt-reg] saved {len(neurons)} neurons to {cache_path}")
    return neurons
