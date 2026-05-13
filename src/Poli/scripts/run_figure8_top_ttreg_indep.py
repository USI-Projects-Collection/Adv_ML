"""
Figure 8 (top) — variante "independent" del metodo Jiang.

Stesso protocollo di `run_figure8_top_ttreg.py` ma usa `TTRegDINOv2Independent`
per iniettare N test-time register con i top-N outlier patch distribuiti su N
register distinti, invece di duplicare lo stesso top-1 outlier.

Output:
  results/figure8_top_ttreg_indep/figure8_top_ttreg_indep.png   (riga combinata)
  results/figure8_top_ttreg_indep/attn_n{N}.png                  (mappe singole)
  results/figure8_top_ttreg_indep/attn_maps.npy                  (raw)
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

from ablation.test_time_registers import cached_register_neurons
from ablation.test_time_registers_independent import (
    load_dinov2_with_tt_registers_independent,
)


IMG_PATH = ROOT.parent.parent / "assets" / "paper_images" / "67.png"
OUT_DIR = ROOT / "results" / "figure8_top_ttreg_indep"
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
    """CLS->patch attention from last layer, mean over heads.

    Layout: [CLS, PATCH..., TT_REG...] → patches start at 1.
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
        captured["attn"] = attn.softmax(dim=-1)

    h = last.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(x)
    h.remove()

    a = captured["attn"][0]                         # (H, N, N)
    cls_to_patch = a[:, 0, 1 : 1 + num_patches]     # (H, P)
    return cls_to_patch.mean(dim=0).cpu().numpy().reshape(GRID, GRID)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}")
    neurons = cached_register_neurons(NEURONS_CACHE, top_k=50, device=device)
    print(f"[neurons] {len(neurons)} register neurons loaded")

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
        m = load_dinov2_with_tt_registers_independent(n, neurons, img_size=IMG_SIZE).to(device)
        patch_attention(m)
        attn_maps[n] = cls_to_patch_attention(m, x, num_patches=m.num_patches)
        del m
        if device == "mps":
            torch.mps.empty_cache()

    fig, axes = plt.subplots(1, 1 + len(REGS), figsize=(2 * (1 + len(REGS)), 2.2))
    axes[0].imshow(img_resized)
    axes[0].set_title("Input", fontsize=10)
    axes[0].axis("off")
    for ax, n in zip(axes[1:], REGS):
        ax.imshow(attn_maps[n], cmap="viridis", interpolation="nearest")
        ax.set_title(f"{n} [tt-reg indep]", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    out_png = OUT_DIR / "figure8_top_ttreg_indep.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\n[final] saved {out_png}")

    for n, a in attn_maps.items():
        plt.imsave(OUT_DIR / f"attn_n{n}.png", a, cmap="viridis")
    np.save(OUT_DIR / "attn_maps.npy", np.stack([attn_maps[n] for n in REGS]))


if __name__ == "__main__":
    main()
