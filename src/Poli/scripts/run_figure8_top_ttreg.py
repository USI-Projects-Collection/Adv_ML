"""
Figure 8 (top) — DINOv2 baseline + N test-time registers (Jiang et al. 2025).

Same protocol as run_figure8_top.py but starting from the unmodified
DINOv2-L/14 baseline and using the Jiang method to inject N test-time
registers for N ∈ {0, 1, 2, 4, 8, 16}.

We capture the last-layer CLS->patch attention probabilities (with
fused_attn disabled) and render each map with viridis.

Outputs:
  results/figure8_top_ttreg/figure8_top_ttreg.png   (combined row)
  results/figure8_top_ttreg/attn_n{N}.png           (individual maps)
  results/figure8_top_ttreg/attn_maps.npy           (raw maps)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablation.test_time_registers import (
    cached_register_neurons,
    load_dinov2_with_tt_registers,
)


IMG_PATH = ROOT.parent.parent / "assets" / "paper_images" / "67.png"
OUT_DIR = ROOT / "results" / "figure8_top_ttreg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGS = [0, 1, 2, 4, 8, 16]
IMG_SIZE = 518
PATCH = 14
GRID = IMG_SIZE // PATCH  # 37
NEURONS_CACHE = ROOT / "results" / "dinov2_tt_register_neurons.pt"


def patch_attention(model):
    for blk in model.backbone.blocks:
        blk.attn.fused_attn = False


def cls_to_patch_attention(model, x: torch.Tensor, num_patches: int) -> np.ndarray:
    """Return CLS->patch attention reshaped to (GRID, GRID).

    Layout for TT-reg model: [CLS, PATCH..., TT_REG...] so patches start at 1.
    """
    last = model.backbone.blocks[-1].attn
    captured = {}

    def hook(mod, inputs, output):
        x_ln = inputs[0]
        B, N, C = x_ln.shape
        H = mod.num_heads
        D = C // H
        qkv = mod.qkv(x_ln).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, _v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (D ** -0.5)
        attn = attn.softmax(dim=-1)
        captured["attn"] = attn

    h = last.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(x)
    h.remove()

    a = captured["attn"][0]                          # (H, N, N)
    cls_to_all = a[:, 0]                              # (H, N)
    cls_to_patch = cls_to_all[:, 1 : 1 + num_patches] # (H, P)
    cls_to_patch = cls_to_patch.mean(dim=0)           # (P,)
    return cls_to_patch.cpu().numpy().reshape(GRID, GRID)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}")

    neurons = cached_register_neurons(NEURONS_CACHE, top_k=50, device=device)
    print(f"[neurons] {len(neurons)} neurons loaded")

    if not IMG_PATH.exists():
        raise FileNotFoundError(f"Missing image {IMG_PATH}")
    img = Image.open(IMG_PATH).convert("RGB")
    tfm = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    x = tfm(img).unsqueeze(0).to(device)
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))

    attn_maps = {}
    for n in REGS:
        print(f"[run] N={n}", flush=True)
        m = load_dinov2_with_tt_registers(n, neurons, img_size=IMG_SIZE).to(device)
        patch_attention(m)
        attn = cls_to_patch_attention(m, x, num_patches=m.num_patches)
        attn_maps[n] = attn
        del m
        if device == "mps":
            torch.mps.empty_cache()

    # Combined figure: input + 6 attention maps
    fig, axes = plt.subplots(1, 1 + len(REGS), figsize=(2 * (1 + len(REGS)), 2.2))
    axes[0].imshow(img_resized)
    axes[0].set_title("Input", fontsize=10)
    axes[0].axis("off")
    for ax, n in zip(axes[1:], REGS):
        ax.imshow(attn_maps[n], cmap="viridis", interpolation="nearest")
        ax.set_title(f"{n} [tt-reg]", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    out_png = OUT_DIR / "figure8_top_ttreg.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\n[final] saved {out_png}")

    for n, a in attn_maps.items():
        plt.imsave(OUT_DIR / f"attn_n{n}.png", a, cmap="viridis")
    np.save(OUT_DIR / "attn_maps.npy", np.stack([attn_maps[n] for n in REGS]))


if __name__ == "__main__":
    main()
