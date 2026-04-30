import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import timm
import os
import random

# ==============================================================================
# APPENDIX D.1 REPLICATION — Figure 15
# "Vision Transformers Need Registers" (Darcet et al., ICLR 2024)
#
# Figure 15: Strip/scatter plot of token L2 norms.
#   x-axis  : token type category  (CLS | patch)  or  (CLS | reg_0…reg_3 | patch)
#   y-axis  : L2 norm of the output token (after the final LayerNorm)
#   Each dot: one token from one image (jittered horizontally for readability)
#
# Left panel  — DINOv2 no registers  : categories = [CLS, patch]
# Right panel — DINOv2 4 registers   : categories = [CLS, reg_0, reg_1, reg_2, reg_3, patch]
#
# Key findings to reproduce:
#   1. w/o reg: patch tokens have a dense cloud ~10-30 AND a sparse high-norm
#      cloud above ~100 (the outliers). CLS sits around ~20.
#   2. w/  reg: patch tokens are clean, no outliers. Each register token sits at
#      a distinct high norm (the "quantized" effect noted by the authors).
#      The 4 registers have clearly separated norm levels (~20, ~80, ~130, ~65).
#
# ── Model source ────────────────────────────────────────────────────────────
#   vit_base_patch14_dinov2.lvd142m        — DINOv2-B, no registers  (timm)
#   vit_base_patch14_reg4_dinov2.lvd142m   — DINOv2-B, 4 registers   (timm)
#
# ── Hook target ─────────────────────────────────────────────────────────────
#   model.norm  (the final LayerNorm applied to all tokens after all blocks)
#   Output shape: (B, N_total, C)
#   Token layout w/o reg: [CLS, p_0, …, p_N]
#   Token layout w/  reg: [CLS, r_0, r_1, r_2, r_3, p_0, …, p_N]
# ==============================================================================

# ── Config ───────────────────────────────────────────────────────────────────
N_IMAGES    = 500   # images to process; 200+ gives stable clouds
BATCH_SIZE  = 16
IMG_SIZE    = 518   # DINOv2 native input resolution
N_REGS      = 4     # registers in the "with reg" model

# Subsample patch tokens for plotting — with 500 images × 1369 patches = 684k
# points, plotting all is slow. We randomly keep MAX_PATCH_DOTS per panel.
MAX_PATCH_DOTS = 30_000

# Path to a folder of images. COCO val2017 works perfectly.
# Falls back to the single project dog image if the folder is missing.
IMAGE_DIR    = "./src/Raffo/data/coco/images/val2017"
FALLBACK_IMG = "./Black_Labrador_Retriever_portrait.jpg"
# ─────────────────────────────────────────────────────────────────────────────


# ==============================================================================
# 1. DATASET
# ==============================================================================

class ImageListDataset(torch.utils.data.Dataset):
    def __init__(self, folder: str, transform, max_images: int):
        exts  = {".jpg", ".jpeg", ".png", ".webp"}
        paths = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        ])
        random.seed(42)
        random.shuffle(paths)
        self.paths     = paths[:max_images]
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            return self.transform(Image.open(self.paths[idx]).convert("RGB"))
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)


def make_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ==============================================================================
# 2. NORM EXTRACTOR  (hooks model.norm — the final LayerNorm)
# ==============================================================================

class FinalNormExtractor:
    """
    Hooks model.norm INPUT (pre-LayerNorm residual stream) to capture
    raw token norms matching Figure 15 of the paper.

    WHY inputs[0] not output: the reg model's LayerNorm has very different
    learned weight/bias than the no-reg model. Using output (post-LN) makes
    ALL 684k patch tokens appear as outliers (100% above threshold) — an
    artifact of LN rescaling, not real. inputs[0] is the raw residual stream
    where actual norm differences between normal/outlier/register tokens live.


    Token layout in timm DINOv2:
      no reg  →  [CLS,  p_0 … p_N]
      4 reg   →  [CLS,  r_0, r_1, r_2, r_3,  p_0 … p_N]
    """
    def __init__(self, model, n_regs: int):
        self.n_regs  = n_regs
        self._output = None
        self._handle = model.norm.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        # inputs[0] = pre-LayerNorm tensor (raw residual stream)
        # output    = post-LayerNorm (distorted by learned LN params in reg model)
        self._output = inputs[0].detach().cpu()   # (B, N_total, C)

    def remove(self):
        self._handle.remove()

    def split(self):
        """
        Returns dict with keys 'cls', 'patch', and optionally 'reg_0'…'reg_3'.
        Each value is a 1-D numpy array of L2 norms.
        """
        x   = self._output   # (B, N_total, C)
        out = {}

        # CLS: index 0
        out["cls"] = x[:, 0, :].norm(dim=-1).numpy()         # (B,)

        if self.n_regs > 0:
            # Registers: indices 1 … n_regs
            for i in range(self.n_regs):
                out[f"reg_{i}"] = x[:, 1 + i, :].norm(dim=-1).numpy()
            # Patches: indices n_regs+1 … end
            out["patch"] = x[:, 1 + self.n_regs:, :].norm(dim=-1).reshape(-1).numpy()
        else:
            # Patches: indices 1 … end
            out["patch"] = x[:, 1:, :].norm(dim=-1).reshape(-1).numpy()

        return out


# ==============================================================================
# 3. COLLECT NORMS  (full dataset pass)
# ==============================================================================

@torch.inference_mode()
def collect_norms(model_name: str, n_regs: int,
                  loader: DataLoader, device: torch.device) -> dict:
    """Load model, run all batches, return accumulated norm dict."""
    print(f"\n  Loading {model_name} …")
    model     = timm.create_model(model_name, pretrained=True).to(device).eval()
    extractor = FinalNormExtractor(model, n_regs=n_regs)

    # Accumulators — list of arrays, one per batch
    accum = {}

    for batch in loader:
        _ = model(batch.to(device))
        split = extractor.split()
        for k, v in split.items():
            accum.setdefault(k, []).append(v)

    extractor.remove()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Concatenate across batches
    result = {k: np.concatenate(v) for k, v in accum.items()}

    # Report
    p = result["patch"]
    print(f"    patches  : {len(p):,}  |  "
          f"outliers >150: {(p > 150).sum():,} ({100*(p>150).mean():.2f}%)")
    print(f"    CLS norm : mean={result['cls'].mean():.1f}  "
          f"std={result['cls'].std():.1f}")
    for k in sorted(k for k in result if k.startswith("reg_")):
        r = result[k]
        print(f"    {k} norm : mean={r.mean():.1f}  std={r.std():.1f}")

    return result


# ==============================================================================
# 4. PLOT  — strip plot matching Figure 15 exactly
# ==============================================================================

def strip_plot(ax, categories: list[str], data: dict[str, np.ndarray],
               color: str = "#4C72B0", jitter: float = 0.15,
               dot_alpha: float = 0.03, dot_size: float = 0.5,
               max_dots: int = MAX_PATCH_DOTS):
    """
    Draw a jittered strip plot on `ax`.
    categories : ordered list of keys in `data` to plot left→right
    data        : dict key → 1-D array of norm values
    """
    ax.set_xlim(-0.5, len(categories) - 0.5)

    for xi, cat in enumerate(categories):
        vals = data[cat]

        # Subsample patch tokens to keep plotting fast
        if cat == "patch" and len(vals) > max_dots:
            rng  = np.random.default_rng(0)
            vals = rng.choice(vals, size=max_dots, replace=False)

        # Horizontal jitter
        jx = xi + np.random.uniform(-jitter, jitter, size=len(vals))

        # Choose alpha — registers are few, make them more visible
        alpha = 0.5 if cat.startswith("reg_") or cat == "cls" else dot_alpha
        size  = 1.5 if cat.startswith("reg_") or cat == "cls" else dot_size

        ax.scatter(jx, vals, s=size, alpha=alpha, color=color,
                   linewidths=0, rasterized=True)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_xlabel("type", fontsize=10)


def plot_figure_15(results: dict, save_path: str = "figure_15_replication.png"):
    """
    results["no_reg"]   : norm dict for DINOv2 w/o registers
    results["with_reg"] : norm dict for DINOv2 w/  4 registers
    """
    np.random.seed(42)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    fig.suptitle(
        "Figure 15 — Token Norm Distribution (Appendix D.1)\n"
        "DINOv2 without registers (left) · DINOv2 with 4 registers (right)",
        fontsize=11, fontweight="bold"
    )

    # ── Left panel: no registers ──────────────────────────────────────────────
    cats_no_reg = ["CLS", "patch"]
    data_no_reg = {
        "CLS":   results["no_reg"]["cls"],
        "patch": results["no_reg"]["patch"],
    }
    strip_plot(axes[0], cats_no_reg, data_no_reg, color="#4C72B0")
    axes[0].set_title("(a) DINOv2 - no register", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("norm", fontsize=10)
    axes[0].set_ylim(bottom=0)

    # ── Right panel: 4 registers ──────────────────────────────────────────────
    cats_reg = ["CLS", "reg_0", "reg_1", "reg_2", "reg_3", "patch"]
    data_reg  = {
        "CLS":   results["with_reg"]["cls"],
        "reg_0": results["with_reg"]["reg_0"],
        "reg_1": results["with_reg"]["reg_1"],
        "reg_2": results["with_reg"]["reg_2"],
        "reg_3": results["with_reg"]["reg_3"],
        "patch": results["with_reg"]["patch"],
    }
    strip_plot(axes[1], cats_reg, data_reg, color="#4C72B0")
    axes[1].set_title("(b) DINOv2 - 4 registers", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("norm", fontsize=10)

    # Match y-axis range to the paper (~0–200)
    y_max = max(
        results["no_reg"]["patch"].max(),
        results["with_reg"]["patch"].max(),
        *(results["with_reg"][f"reg_{i}"].max() for i in range(N_REGS)),
    ) * 1.1
    y_max = max(y_max, 200)
    for ax in axes:
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"\n[Figure 15] Saved → {save_path}")
    plt.show()


# ==============================================================================
# 5. SINGLE-IMAGE FALLBACK
# ==============================================================================

@torch.inference_mode()
def collect_norms_single(image_path: str, device: torch.device) -> dict:
    """Run on a single image — coarse but demonstrates the split correctly."""
    print(f"\n  [demo] Single image: {image_path}")
    transform = make_transform()
    img = Image.open(image_path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(device)

    results = {}
    for model_name, n_regs, key in [
        ("vit_base_patch14_dinov2.lvd142m",      0, "no_reg"),
        ("vit_base_patch14_reg4_dinov2.lvd142m", 4, "with_reg"),
    ]:
        print(f"  Loading {model_name} …")
        model     = timm.create_model(model_name, pretrained=True).to(device).eval()
        extractor = FinalNormExtractor(model, n_regs=n_regs)
        _         = model(inp)
        split     = extractor.split()
        extractor.remove()
        del model
        results[key] = {k: v for k, v in split.items()}
        p = results[key]["patch"]
        print(f"    patch norms: min={p.min():.1f}  max={p.max():.1f}  "
              f"outliers>150: {(p > 150).sum()}")
    return results


# ==============================================================================
# 6. MAIN
# ==============================================================================

if __name__ == "__main__":
    device = (torch.device("cuda")  if torch.cuda.is_available()  else
              torch.device("mps")   if torch.backends.mps.is_available() else
              torch.device("cpu"))
    print(f"Device: {device}")

    if os.path.isdir(IMAGE_DIR) and len(os.listdir(IMAGE_DIR)) > 10:
        print(f"\nUsing folder: {IMAGE_DIR}  (up to {N_IMAGES} images)")
        transform = make_transform()
        dataset   = ImageListDataset(IMAGE_DIR, transform, max_images=N_IMAGES)
        loader    = DataLoader(dataset, batch_size=BATCH_SIZE,
                               num_workers=0, shuffle=False, drop_last=False)
        print(f"  {len(dataset)} images found")

        results = {}
        for model_name, n_regs, key in [
            ("vit_large_patch14_dinov2.lvd142m",      0, "no_reg"),
            ("vit_large_patch14_reg4_dinov2.lvd142m", 4, "with_reg"),
        ]:
            results[key] = collect_norms(model_name, n_regs, loader, device)

    elif os.path.exists(FALLBACK_IMG):
        results = collect_norms_single(FALLBACK_IMG, device)

    else:
        raise FileNotFoundError(
            f"No images found. Set IMAGE_DIR to a folder with images, "
            f"or place an image at {FALLBACK_IMG}"
        )

    plot_figure_15(results, save_path="./src/Raffo/img/figure_15_replication.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    for key, label in [("no_reg", "w/o registers"), ("with_reg", "w/ 4 reg")]:
        p = results[key]["patch"]
        c = results[key]["cls"]
        print(f"\n{label}:")
        print(f"  patch : mean={p.mean():.1f}  max={p.max():.1f}  "
              f"outliers>150: {(p>150).sum():,} ({100*(p>150).mean():.2f}%)")
        print(f"  CLS   : mean={c.mean():.1f}  std={c.std():.1f}")
        for i in range(N_REGS):
            k = f"reg_{i}"
            if k in results[key]:
                r = results[key][k]
                print(f"  reg_{i} : mean={r.mean():.1f}  std={r.std():.1f}")
    print("=" * 55)