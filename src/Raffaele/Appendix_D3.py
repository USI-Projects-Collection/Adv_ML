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

# Config 
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
DATA_ROOT   = "./src/Raffaele/data"
SAVE_PATH   = "./src/Raffaele/img/figure_16_replication.png"

FALLBACK_DIRS = [
    "./coco/images/val2017",
    "./src/Raffo/datxa/coco/images/val2017",
    "./data/fgvc-aircraft-2013b/data/images",
]

# ==============================================================================
# 1. DATASET  —  Oxford-IIIT Pet 
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
            return self.transform(img), 0 
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
# ==============================================================================

class AttentionMatrixExtractor:
    def __init__(self, model):
        self._weights = None

        # Disable fused attention on ALL blocks (needed for hook to fire)
        for block in model.blocks:
            block.attn.fused_attn = False

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
# ==============================================================================

@torch.inference_mode()
def accumulate_maps(model, extractor, loader, device):
    keys = ["CLS", "reg_0", "reg_1", "reg_2", "reg_3", "patch"]
    sums = {k: np.zeros(N_PATCHES, dtype=np.float64) for k in keys}
    n    = 0

    rows = {
        "CLS":   0,
        "reg_0": 1, "reg_1": 2, "reg_2": 3, "reg_3": 4,
        "patch": N_PREFIX + N_PATCHES // 2,   # centre patch (18,18)
    }
    patch_cols = slice(N_PREFIX, N_PREFIX + N_PATCHES)

    for batch in tqdm(loader, desc="  averaging", leave=True):
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        model(imgs.to(device, non_blocking=True))
        A = extractor.get()          
        if A is None: continue

        # Average across heads
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
# 4. PLOT
# ==============================================================================

def plot_figure_16(maps, n_images, dataset_name, save_path):
    order  = ["CLS", "reg_0", "reg_1", "reg_2", "reg_3", "patch"]
    labels = ["[CLS]", r"reg$_0$", r"reg$_1$", r"reg$_2$", r"reg$_3$", "patch"]

    fig, axes = plt.subplots(1, 6, figsize=(13, 2.6),
                              gridspec_kw={"wspace": 0.04})
    fig.patch.set_facecolor("white")

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

    # Dataset
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
        )

    # Model
    print(f"\nLoading {MODEL_NAME} …")
    model     = timm.create_model(MODEL_NAME, pretrained=True).to(device).eval()
    extractor = AttentionMatrixExtractor(model)
    print(f"  heads={model.blocks[-1].attn.num_heads}  "
          f"fused_attn={model.blocks[-1].attn.fused_attn}  "
          f"(should be False)")

    # Accumulate
    print(f"\nAveraging attention over {n_images} images …")
    maps, n_used = accumulate_maps(model, extractor, loader, device)

    extractor.remove()
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    gc.collect()

    # Plot and diagnostics
    plot_figure_16(maps, n_used, dataset_name, SAVE_PATH)
    print_diagnostics(maps, n_used)