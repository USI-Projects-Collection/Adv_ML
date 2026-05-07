"""
APPENDIX D.3 REPLICATION — Figure 16: Positional Focus
"Vision Transformers Need Registers" (Darcet et al., ICLR 2024)

════════════════════════════════════════════════════════════════════════════════
DATASET: Oxford-IIIT Pet Dataset  (auto-downloaded, ~800 MB)

  The paper used ImageNet-22k. We use Oxford-IIIT Pet as a freely available
  proxy with the same key properties:
    • Single object centred in every image (one pet per photo)
    • Clean or softly blurred background (photographers focus on the animal)
    • 37 breeds × ~200 images = 7,349 total  (trainval + test combined)
    • Auto-downloadable via torchvision — no license needed

  This matches the "object-centric images" property the paper describes as
  the reason the average attention maps form centred blobs. COCO (scene
  images) produced horizontally elongated blobs; Pets will produce the
  rounder, symmetric blobs seen in the paper's Figure 16.

  Fallback chain if Pets can't be downloaded:
    → COCO val2017 → FGVC-Aircraft → any folder with images

════════════════════════════════════════════════════════════════════════════════
WHAT THIS REPLICATES

  Figure 16: six heatmaps of the AVERAGE attention map in the last layer
  of DINOv2+reg4 (ViT-Large), one per token type:
    (a) [CLS]  (b) reg₀  (c) reg₁  (d) reg₂  (e) reg₃  (f) patch

  Each heatmap = mean over N images of:
    row i of the softmax attention matrix → reshape to 37×37 spatial grid

  Key findings to reproduce:
    1. CLS and registers → broad, diffuse attention (global information)
    2. Patch              → tight, localised attention (local information)
    3. Registers differ from each other: reg₃ focuses on borders, others on centre
    4. reg₂ tends slightly toward upper image regions

════════════════════════════════════════════════════════════════════════════════
HOW IT WORKS

  Token layout: [CLS=0, reg₀=1, reg₁=2, reg₂=3, reg₃=4, p₀=5 … p₁₃₆₈=1373]

  For each image:
    1. Forward pass through DINOv2+reg4
    2. Hook captures softmax attention: A ∈ R^{H × N_total × N_total}
    3. Average across H=16 heads → Ā ∈ R^{N_total × N_total}
    4. For each token i: extract row i, patch columns → reshape to 37×37
    5. Accumulate across images → divide by count

  fused_attn is disabled so timm materialises the softmax matrix explicitly,
  making it accessible via a hook on attn_drop (fires right after softmax).

Install: !pip install timm torchvision
"""

import os, gc
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import OxfordIIITPet
from PIL import Image
from tqdm import tqdm
import timm
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "vit_large_patch14_reg4_dinov2.lvd142m"
IMG_SIZE    = 518
PATCH_SIZE  = 14
GRID_SIZE   = IMG_SIZE // PATCH_SIZE   # 37
N_PATCHES   = GRID_SIZE ** 2           # 1369
N_REGS      = 4
N_PREFIX    = 1 + N_REGS              # 5  (CLS + 4 registers)

N_IMAGES    = 500                      # images to average; 500 gives stable maps
BATCH_SIZE  = 8                        # safe for ViT-Large on T4
NUM_WORKERS = 2
DATA_ROOT   = "./data"
SAVE_PATH   = "figure_16_replication.png"

# Fallback image directories if Pets download fails
FALLBACK_DIRS = [
    "./coco/images/val2017",
    "./src/Raffo/data/coco/images/val2017",
    "./data/fgvc-aircraft-2013b/data/images",
]
# ─────────────────────────────────────────────────────────────────────────────


# ==============================================================================
# 1. DATASET  —  Oxford-IIIT Pet (primary) with fallback chain
# ==============================================================================

def make_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_pet_loader(data_root, batch_size, max_images, num_workers):
    """
    Loads Oxford-IIIT Pet (both splits combined = 7349 images).
    Returns (DataLoader, n_images).
    """
    tf = make_transform()
    try:
        # Combine trainval + test for maximum diversity
        trainval = OxfordIIITPet(data_root, split="trainval",
                                  transform=tf, download=True)
        test     = OxfordIIITPet(data_root, split="test",
                                  transform=tf, download=True)
        full_ds  = ConcatDataset([trainval, test])
        # Subsample to max_images
        if len(full_ds) > max_images:
            indices = list(range(max_images))
            from torch.utils.data import Subset
            full_ds = Subset(full_ds, indices)
        print(f"  Oxford-IIIT Pet: {len(full_ds)} images  "
              f"(single centred pet, clean background)")
        loader = DataLoader(full_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            drop_last=False)
        return loader, len(full_ds)
    except Exception as e:
        print(f"  Oxford-IIIT Pet download failed: {e}")
        return None, 0


class FlatImageFolder(torch.utils.data.Dataset):
    """Recursively loads images from a directory (flat or with subdirs)."""
    def __init__(self, root, transform, max_images):
        exts = {".jpg", ".jpeg", ".png", ".webp", ".JPEG", ".JPG"}
        self.paths = []
        for dirpath, _, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1] in exts:
                    self.paths.append(os.path.join(dirpath, f))
        self.paths = sorted(self.paths)[:max_images]
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), 0   # dummy label
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), 0


def get_fallback_loader(batch_size, max_images, num_workers):
    """Try fallback directories if Pets not available."""
    for d in FALLBACK_DIRS:
        if os.path.isdir(d):
            n = sum(1 for _, _, fs in os.walk(d) for f in fs
                    if os.path.splitext(f)[1].lower() in {".jpg",".jpeg",".png"})
            if n > 10:
                print(f"  Fallback: {d}  ({n} images)")
                ds = FlatImageFolder(d, make_transform(), max_images)
                return (DataLoader(ds, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=True),
                        len(ds))
    return None, 0


# ==============================================================================
# 2. ATTENTION MATRIX EXTRACTOR
#    fused_attn=True (default) routes through F.scaled_dot_product_attention,
#    a fused CUDA kernel that never materialises the softmax matrix.
#    Setting fused_attn=False forces explicit computation and makes the
#    (B, H, N, N) weight matrix available to hooks.
#    We hook attn_drop which receives the softmax weights as its input.
# ==============================================================================

class AttentionMatrixExtractor:
    def __init__(self, model):
        self._weights = None

        # Disable fused attention on ALL blocks (needed for hook to fire)
        for block in model.blocks:
            block.attn.fused_attn = False

        # Hook attn_drop of the LAST block — input = (B, H, N, N) softmax weights
        def _hook(module, inputs, output):
            self._weights = inputs[0].detach().cpu()

        self._handle = model.blocks[-1].attn.attn_drop.register_forward_hook(_hook)

    def get(self):
        """Returns (B, H, N_total, N_total) CPU float32, or None if not fired."""
        return self._weights

    def remove(self):
        if self._handle:
            self._handle.remove()


# ==============================================================================
# 3. ACCUMULATE AVERAGE ATTENTION MAPS
#    Mathematical formulation:
#      M_i = (1/N) Σ_n  mean_h[ A^(n)_{h, i, N_prefix:} ]   ∈ R^{37×37}
#    where i ∈ {0=CLS, 1=reg₀, 2=reg₁, 3=reg₂, 4=reg₃, N_prefix+centre=patch}
# ==============================================================================

@torch.inference_mode()
def accumulate_maps(model, extractor, loader, device):
    keys = ["CLS", "reg_0", "reg_1", "reg_2", "reg_3", "patch"]
    sums = {k: np.zeros(N_PATCHES, dtype=np.float64) for k in keys}
    n    = 0

    # Absolute row indices in the attention matrix
    rows = {
        "CLS":   0,
        "reg_0": 1, "reg_1": 2, "reg_2": 3, "reg_3": 4,
        "patch": N_PREFIX + N_PATCHES // 2,   # centre patch (18,18)
    }
    patch_cols = slice(N_PREFIX, N_PREFIX + N_PATCHES)

    for batch in tqdm(loader, desc="  averaging", leave=True):
        # Handle both (imgs,) and (imgs, labels) batch formats
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        model(imgs.to(device, non_blocking=True))
        A = extractor.get()          # (B, H, N_total, N_total)
        if A is None: continue

        # Average across heads: (B, N_total, N_total)
        A_mean = A.mean(dim=1).numpy()
        B = A_mean.shape[0]

        for b in range(B):
            for key, row in rows.items():
                sums[key] += A_mean[b, row, patch_cols]
            n += 1

    print(f"  Averaged over {n} images")
    return {k: (sums[k] / max(n, 1)).reshape(GRID_SIZE, GRID_SIZE)
            for k in keys}, n


# ==============================================================================
# 4. PLOT  —  Figure 16 style
#    inferno colormap (matches paper's orange-red palette)
#    Shared vmax across CLS + registers; separate for patch (different scale)
# ==============================================================================

def plot_figure_16(maps, n_images, dataset_name, save_path):
    order  = ["CLS", "reg_0", "reg_1", "reg_2", "reg_3", "patch"]
    labels = ["[CLS]", r"reg$_0$", r"reg$_1$", r"reg$_2$", r"reg$_3$", "patch"]

    fig, axes = plt.subplots(1, 6, figsize=(13, 2.6),
                              gridspec_kw={"wspace": 0.04})
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Figure 16 — Positional Focus: Average Attention Maps (Appendix D.3)\n"
        f"DINOv2+reg (ViT-Large, 4 registers)  ·  {dataset_name}"
        f"  ·  {n_images} images",
        fontsize=10, fontweight="bold", y=1.05
    )

    # Shared scale for special tokens; independent for patch
    vmax_special = float(np.stack([maps[k] for k in order[:-1]]).max())
    vmax_patch   = float(maps["patch"].max())

    for ax, name, label in zip(axes, order, labels):
        vmax = vmax_patch if name == "patch" else vmax_special
        ax.imshow(maps[name], cmap="inferno", vmin=0, vmax=vmax,
                  interpolation="bilinear")
        ax.set_title(label, fontsize=10, fontweight="bold", pad=3)
        ax.axis("off")

    plt.savefig(save_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"\n[Figure 16] Saved → {save_path}")
    plt.show()


# ==============================================================================
# 5. DIAGNOSTICS
# ==============================================================================

def print_diagnostics(maps, n_images):
    g = GRID_SIZE
    border = np.zeros((g, g), dtype=bool)
    border[0,:] = border[-1,:] = border[:,0] = border[:,-1] = True
    upper  = np.zeros((g, g), dtype=bool);  upper[:g//3, :] = True

    print(f"\n── Spatial statistics (N={n_images} images) ────────────────────────")
    print(f"{'Token':<8}  {'entropy':>8}  {'border%':>9}  {'upper%':>8}  "
          f"{'support%':>10}")
    print("-" * 52)

    for name, label in zip(
        ["CLS","reg_0","reg_1","reg_2","reg_3","patch"],
        ["[CLS]","reg₀","reg₁","reg₂","reg₃","patch"]
    ):
        m   = maps[name]
        m_n = m / (m.sum() + 1e-9)
        entr  = -float((m_n * np.log(m_n + 1e-9)).sum())
        bord  = 100 * m[border].sum()  / (m.sum() + 1e-9)
        upp   = 100 * m[upper].sum()   / (m.sum() + 1e-9)
        supp  = 100 * (m > m.max()*0.1).mean()
        print(f"{label:<8}  {entr:>8.3f}  {bord:>8.1f}%  {upp:>7.1f}%  "
              f"{supp:>9.1f}%")

    print()
    print("Paper's observations (Figure 16):")
    print("  entropy : CLS ≈ registers >> patch  (global vs local)")
    print("  border% : reg₃ highest among registers")
    print("  upper%  : reg₂ slightly higher than others")
    print("  support%: patch << all others  (highly localised)")


# ==============================================================================
# 6. MAIN
# ==============================================================================

if __name__ == "__main__":
    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cpu"))
    print(f"Device: {device}")
    print(f"Model : {MODEL_NAME}")
    print(f"Grid  : {GRID_SIZE}×{GRID_SIZE} = {N_PATCHES} patches\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Loading dataset …")
    loader, n_images = get_pet_loader(DATA_ROOT, BATCH_SIZE, N_IMAGES, NUM_WORKERS)
    dataset_name = "Oxford-IIIT Pet"

    if loader is None:
        print("Trying fallback directories …")
        loader, n_images = get_fallback_loader(BATCH_SIZE, N_IMAGES, NUM_WORKERS)
        dataset_name = "fallback images"

    if loader is None or n_images == 0:
        raise FileNotFoundError(
            "No images found. Oxford-IIIT Pet download failed and no fallback "
            "directories are available. Check your internet connection.")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nLoading {MODEL_NAME} …")
    model     = timm.create_model(MODEL_NAME, pretrained=True).to(device).eval()
    extractor = AttentionMatrixExtractor(model)
    print(f"  heads={model.blocks[-1].attn.num_heads}  "
          f"fused_attn={model.blocks[-1].attn.fused_attn}  "
          f"(should be False)")

    # ── Accumulate ────────────────────────────────────────────────────────────
    print(f"\nAveraging attention over {n_images} images …")
    maps, n_used = accumulate_maps(model, extractor, loader, device)

    extractor.remove()
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    gc.collect()

    # ── Plot & diagnostics ────────────────────────────────────────────────────
    plot_figure_16(maps, n_used, dataset_name, SAVE_PATH)
    print_diagnostics(maps, n_used)