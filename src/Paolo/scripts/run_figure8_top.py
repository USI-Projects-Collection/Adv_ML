"""
Figure 8 (top) — qualitative visualization of artifacts in DINOv2 attention maps
as a function of the number of register tokens N ∈ {0, 1, 2, 4, 8, 16}.

For each N we:
  1. Load DINOv2-reg4 with N registers (via ablation.dinov2_register_sweep).
  2. Run a forward pass on the dog image (assets/paper_images/67.png).
  3. Capture the last-layer attention probabilities CLS -> patches.
  4. Average across heads, reshape to (37, 37), interpolate to image size.
  5. Render to PNG with viridis colormap.

We disable timm's `fused_attn` to obtain explicit attention probabilities.
"""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablation.dinov2_register_sweep import load_dinov2_with_n_registers


# Path to the dog image used by the paper for Figure 8.
IMG_PATH = ROOT.parent.parent / "assets" / "paper_images" / "67.png"
OUT_DIR = ROOT / "results" / "figure8_top"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGS = [0, 1, 2, 4, 8, 16]
IMG_SIZE = 518
PATCH = 14
GRID = IMG_SIZE // PATCH  # 37


def patch_attention_for_explicit_extraction(model):
    """Disable fused_attn on every block so attn() returns probs internally."""
    for blk in model.backbone.blocks:
        blk.attn.fused_attn = False


def cls_to_patch_attention(model, x: torch.Tensor, num_prefix: int) -> np.ndarray:
    """Run forward, capture last-layer softmax(QK^T/sqrt(d)), avg over heads,
    return CLS->patch attention reshaped to (37, 37) numpy."""
    last = model.backbone.blocks[-1].attn
    captured = {}

    def hook(mod, inputs, output):
        # Manually recompute attention probs from QKV linear input
        x_ln = inputs[0]
        B, N, C = x_ln.shape
        H = mod.num_heads
        D = C // H
        qkv = mod.qkv(x_ln).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, _v = qkv[0], qkv[1], qkv[2]
        scale = D ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale       # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        captured["attn"] = attn

    h = last.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(x)
    h.remove()

    a = captured["attn"][0]                            # (H, N, N)
    cls_to_all = a[:, 0]                                # (H, N)
    cls_to_patch = cls_to_all[:, num_prefix:]           # (H, P)
    cls_to_patch = cls_to_patch.mean(dim=0)             # (P,)
    return cls_to_patch.cpu().numpy().reshape(GRID, GRID)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}")

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
        m = load_dinov2_with_n_registers(n, img_size=IMG_SIZE).to(device)
        patch_attention_for_explicit_extraction(m)
        num_prefix = 1 + n
        attn = cls_to_patch_attention(m, x, num_prefix)
        attn_maps[n] = attn
        del m
        if device == "mps":
            torch.mps.empty_cache()

    # Render figure: input + 6 attention maps in a row, like paper.
    fig, axes = plt.subplots(1, 1 + len(REGS), figsize=(2 * (1 + len(REGS)), 2.2))
    axes[0].imshow(img_resized)
    axes[0].set_title("Input", fontsize=10)
    axes[0].axis("off")
    for ax, n in zip(axes[1:], REGS):
        ax.imshow(attn_maps[n], cmap="viridis", interpolation="nearest")
        ax.set_title(f"{n} [reg]", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    out_png = OUT_DIR / "figure8_top.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\n[final] saved {out_png}")

    # Save individual maps too.
    for n, a in attn_maps.items():
        plt.imsave(OUT_DIR / f"attn_n{n}.png", a, cmap="viridis")
    np.save(OUT_DIR / "attn_maps.npy", np.stack([attn_maps[n] for n in REGS]))


if __name__ == "__main__":
    main()
