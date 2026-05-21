import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import timm
import open_clip
import os
import random

# ==============================================================================
# FIGURE 7 REPLICATION — Section 3.2
# "Vision Transformers Need Registers" (Darcet et al., ICLR 2024)
#
# The figure shows strip/scatter plots of OUTPUT token L2 norms for three
# model families (DINOv2 / OpenCLIP / DeiT-III), each compared w/o vs w/ reg.
#
# Layout (matching the paper exactly):
#   - 3 panels, one per model family
#   - Each panel: TWO adjacent narrow columns — "w/o reg" (left), "w/ reg" (right)
#   - Each column has TWO x-positions: CLS and patch
#     (the w/ reg column additionally shows individual register norms)
#   - Each panel has its OWN independent y-axis (scales differ across families)
#   - No shared y-axis — DINOv2 ~200, OpenCLIP ~300, DeiT-III ~1500
#
# ── CORRECT HOOK: post-LN output of model.norm / visual.ln_post ─────────────
# The paper plots norms of the FINAL output tokens (after the last LayerNorm).
# This is what "output token norms" means throughout Section 2 and Figure 3.
#
# The earlier confusion (inverted results for DINOv2) was caused by the
# open_clip_hf crash masking the actual data — the post-LN hook IS correct.
#
# ── open_clip_hf special case ────────────────────────────────────────────────
# The HF test-time registers model's ln_post receives (B*N, C) — already flat.
# Fix: hook the last resblock OUTPUT instead, which is (N, B, C) sequence-first,
# then transpose to (B, N, C). The norms are equivalent because LN is applied
# elementwise and the resblock output is the direct input to ln_post.
#
# ── Token layout ─────────────────────────────────────────────────────────────
#   timm no reg  → [CLS, p_0 … p_N]        at model.norm output
#   timm k reg   → [CLS, r_0 … r_{k-1}, p_0 … p_N]
#   open_clip    → [CLS, p_0 … p_N]        at visual.ln_post output
#   open_clip_hf → [CLS, p_0 … p_N] or [CLS, r_0…r_k, p_0…p_N] from resblock
# ==============================================================================

N_IMAGES       = 200
BATCH_SIZE     = 8
MAX_PATCH_DOTS = 20_000

IMAGE_DIR    = "./src/Raffo/data/coco/images/val2017"
FALLBACK_IMG = "./src/Raffo/img/Black_Labrador_Retriever_portrait.jpg"


# ==============================================================================
# 1. DATASET
# ==============================================================================
class ImageFolderSimple(Dataset):
    def __init__(self, folder, transform, max_images):
        exts  = {".jpg", ".jpeg", ".png", ".webp"}
        paths = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        ])
        random.seed(42)
        random.shuffle(paths)
        self.paths     = paths[:max_images]
        self.transform = transform

    def __len__(self):  return len(self.paths)

    def __getitem__(self, idx):
        try:   return self.transform(Image.open(self.paths[idx]).convert("RGB"))
        except: return torch.zeros(3, 224, 224)


# ==============================================================================
# 2. NORM EXTRACTOR
# ==============================================================================
class OutputNormExtractor:
    """
    Captures per-token L2 norms from the final output of each model.

    Hook targets:
      timm        → model.norm          output: (B, N, C)  [post-LN]
      open_clip   → model.visual.ln_post output: (B, N, C) [post-LN]  
      open_clip_hf→ model.model.visual.transformer.resblocks[-1]
                    output: (N, B, C) sequence-first → transposed to (B, N, C)
                    (ln_post receives flat (B*N, C) in this wrapper, so we hook
                     the last resblock output instead — equivalent norms)

    Token layout after hook (all backends):
      no reg : [CLS, p_0 … p_{N-1}]
      k  reg : [CLS, r_0 … r_{k-1}, p_0 … p_{N-1}]
    """

    def __init__(self, model, backend: str, n_regs: int):
        self.backend = backend
        self.n_regs  = n_regs
        self._data   = None

        if backend == "timm":
            hook_target = model.norm

        elif backend == "open_clip":
            hook_target = model.visual.ln_post

        elif backend == "open_clip_hf":
            # ln_post gets (B*N, C) — flat. Hook last resblock instead.
            hook_target = model.model.visual.transformer.resblocks[-1]

        else:
            raise ValueError(f"Unknown backend: {backend}")

        self._handle = hook_target.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        x = output.detach().float().cpu()

        # open_clip resblocks are (N, B, C) — sequence first
        if x.ndim == 3 and x.shape[0] > x.shape[1]:
            x = x.permute(1, 0, 2)   # → (B, N, C)

        self._data = x

    def remove(self):
        self._handle.remove()

    def get_norms(self) -> dict:
        """Return dict: 'cls', 'patch', and optionally 'reg_0'…'reg_k' — 1-D norm arrays."""
        x   = self._data   # (B, N_total, C)
        out = {}
        out["cls"] = x[:, 0, :].norm(dim=-1).numpy().ravel()
        if self.n_regs > 0:
            for i in range(self.n_regs):
                out[f"reg_{i}"] = x[:, 1 + i, :].norm(dim=-1).numpy().ravel()
            out["patch"] = x[:, 1 + self.n_regs:, :].norm(dim=-1).numpy().ravel()
        else:
            out["patch"] = x[:, 1:, :].norm(dim=-1).numpy().ravel()
        return out


# ==============================================================================
# 3. COLLECTION
# ==============================================================================
@torch.inference_mode()
def collect_from_loader(model, extractor, loader, device, backend) -> dict:
    accum = {}
    for batch in loader:
        batch = batch.to(device)
        if backend == "timm":            model(batch)
        elif backend == "open_clip":     model.encode_image(batch)
        elif backend == "open_clip_hf":  model.model.encode_image(batch)
        for k, v in extractor.get_norms().items():
            accum.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in accum.items()}


@torch.inference_mode()
def collect_single(model, extractor, image_path, img_size, device, backend) -> dict:
    tf  = T.Compose([
        T.Resize((img_size, img_size)), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    img = tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    if backend == "timm":            model(img)
    elif backend == "open_clip":     model.encode_image(img)
    elif backend == "open_clip_hf":  model.model.encode_image(img)
    return extractor.get_norms()


# ==============================================================================
# 4. PLOT  — matching the paper's style
# ==============================================================================
def plot_figure_7(all_norms: dict, save_path: str = "figure_7_replication.png"):
    """
    Reproduces Figure 7 layout:
      - 3 panels (one per model family), each with its own y-axis
      - Each panel: 2 condition columns (w/o reg, w/ reg) placed side-by-side
      - Within each column: CLS strip (left), patch strip (right)
        + register strips between CLS and patch for the w/-reg column
      - Dense jittered scatter, patch tokens subsampled for speed
    """
    rng = np.random.default_rng(42)

    # Panel config: (axes_title, key_noreg, key_reg, n_regs, color)
    panels = [
        ("DINOv2",   "dinov2_noreg",   "dinov2_reg",   4, "#4472C4"),
        ("OpenCLIP", "openclip_noreg", "openclip_reg",  4, "#4472C4"),
        ("DeiT-III", "deit3_noreg",    "deit3_reg",     4, "#4472C4"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.5),
                             constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.1, hspace=0.1)
    plt.subplots_adjust(top=0.82)
    
    def scatter_strip(ax, x_pos, vals, color, is_special=False):
        """Draw one jittered strip at x_pos."""
        if len(vals) == 0:
            return
        if not is_special and len(vals) > MAX_PATCH_DOTS:
            vals = rng.choice(vals, size=MAX_PATCH_DOTS, replace=False)
        jx = x_pos + rng.uniform(-0.25, 0.25, size=len(vals))
        ax.scatter(jx, vals,
                   s=0.3 if not is_special else 1.5,
                   alpha=0.08 if not is_special else 0.4,
                   color=color, linewidths=0, rasterized=True)

    for ax, (title, k_nr, k_r, n_regs, color) in zip(axes, panels):
        data_nr  = all_norms.get(k_nr)
        data_reg = all_norms.get(k_r)   # may be None

        # ── Build x-position layout ──────────────────────────────────────────
        # w/o reg: CLS at x=0, patch at x=1
        # gap between conditions: 0.5
        # w/ reg: CLS at x=2, reg_0..reg_k at x=3..2+k, patch at x=3+k
        gap      = 0.5
        x_cls_nr = 0
        x_pat_nr = 1
        x_cls_r  = x_pat_nr + 1 + gap
        x_regs   = [x_cls_r + 1 + i for i in range(n_regs)]
        x_pat_r  = x_cls_r + 1 + n_regs

        # ── w/o reg strips ───────────────────────────────────────────────────
        if data_nr is not None:
            scatter_strip(ax, x_cls_nr, data_nr["cls"],   color, is_special=True)
            scatter_strip(ax, x_pat_nr, data_nr["patch"], color, is_special=False)

        # ── w/ reg strips ────────────────────────────────────────────────────
        if data_reg is not None:
            scatter_strip(ax, x_cls_r, data_reg["cls"],   color, is_special=True)
            for i, xr in enumerate(x_regs):
                key = f"reg_{i}"
                if key in data_reg:
                    scatter_strip(ax, xr, data_reg[key], color, is_special=True)
            scatter_strip(ax, x_pat_r, data_reg["patch"], color, is_special=False)
        else:
            # Shade the w/ reg half and add a note
            ax.axvspan(x_cls_r - 0.6, x_pat_r + 0.5,
                       alpha=0.06, color="gray", zorder=0)
            ax.text((x_cls_r + x_pat_r) / 2, 0.5, "No public\ncheckpoint",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8,
                    color="#999999", style="italic")

        # ── x-ticks ──────────────────────────────────────────────────────────
        tick_pos    = [x_cls_nr, x_pat_nr, x_cls_r] + x_regs + [x_pat_r]
        tick_labels = ["CLS", "patch", "CLS"] \
                    + [f"reg_{i}" for i in range(n_regs)] \
                    + ["patch"]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=40, ha="right")
        ax.set_xlim(x_cls_nr - 0.7, x_pat_r + 0.6)

        # ── Vertical divider ─────────────────────────────────────────────────
        divider_x = (x_pat_nr + x_cls_r) / 2
        ax.axvline(divider_x, color="#BBBBBB", linewidth=0.8, linestyle="--")

        # ── Condition labels at top of plot ───────────────────────────────────
        ax.text((x_cls_nr + x_pat_nr) / 2, 1.02, "w/o REG",
                transform=ax.get_xaxis_transform(),
                ha="center", fontsize=8, color="#777777")
        ax.text((x_cls_r + x_pat_r) / 2, 1.02, "w/ REG",
                transform=ax.get_xaxis_transform(),
                ha="center", fontsize=8, color="#4472C4", fontweight="bold")

        # ── Per-panel y-axis (paper uses independent scales) ─────────────────
        # Compute from the actual data of this panel only
        panel_vals = []
        for nd in [data_nr, data_reg]:
            if nd is not None:
                panel_vals.append(nd["patch"])
        if panel_vals:
            p99 = np.percentile(np.concatenate(panel_vals), 99.5)
            y_max = max(p99 * 1.3, 100)
        else:
            y_max = 300

        ax.set_ylim(0, y_max)
        ax.yaxis.set_major_locator(ticker.AutoLocator())
        ax.set_ylabel("norm", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=28)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\n[Figure 7] Saved → {save_path}")
    plt.show()


# ==============================================================================
# 5. MODEL CONFIGS  (key, family, model_id, pretrained, backend, n_regs, img_size)
# ==============================================================================
MODEL_CONFIGS = [
    ("dinov2_noreg",   "timm",        "vit_base_patch14_dinov2.lvd142m",               True,                "timm",        0, 518),
    ("dinov2_reg",     "timm",        "vit_base_patch14_reg4_dinov2.lvd142m",          True,                "timm",        4, 518),
    ("openclip_noreg", "open_clip",   "ViT-B-16",                                      "laion2b_s34b_b88k", "open_clip",   0, 224),
    ("openclip_reg",   "open_clip_hf","amildravid4292/clip-vitb16-test-time-registers", True,               "open_clip_hf", 4, 224),
    ("deit3_noreg",    "timm",        "deit3_base_patch16_224.fb_in22k_ft_in1k",       True,                "timm",        0, 224),
    ("deit3_reg",    "timm",        "vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k", True,                "timm",        4, 256),
    # ("deit3_reg",      "timm",        [
    #     "vit_base_patch16_reg4_gap_256.sbb_in12k_ft_in1k",   # sbb v1 tag
    #     "vit_base_patch16_reg4_gap_256.sbb2_in12k_ft_in1k",  # sbb v2 tag
    #     "vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k",# medium fallback
    #     "vit_mediumd_patch16_reg4_gap_256.sbb_in12k",         # pretrain-only fallback
    # ]),
]


# ==============================================================================
# 6. MAIN
# ==============================================================================
def load_model(cfg, device):
    _, _, model_id, pretrained, backend, _, _ = cfg
    if backend == "timm":
        return timm.create_model(model_id, pretrained=pretrained).to(device).eval()
    elif backend == "open_clip":
        m, _, _ = open_clip.create_model_and_transforms(model_id, pretrained=pretrained)
        return m.to(device).eval()
    elif backend == "open_clip_hf":
        from transformers import AutoModel
        return AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
    raise ValueError(f"Unknown backend: {backend}")


if __name__ == "__main__":
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    use_folder = (
        os.path.isdir(IMAGE_DIR) and
        len([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]) > 10
    )

    all_norms = {}

    for cfg in MODEL_CONFIGS:
        key, _, model_id, pretrained, backend, n_regs, img_size = cfg
        print(f"\n{'='*60}")
        print(f"Loading {key}  [{model_id}]")
        print(f"{'='*60}")

        model     = load_model(cfg, device)
        extractor = OutputNormExtractor(model, backend=backend, n_regs=n_regs)

        if use_folder:
            tf = T.Compose([
                T.Resize((img_size, img_size)), T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
            dataset = ImageFolderSimple(IMAGE_DIR, tf, max_images=N_IMAGES)
            loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                                 num_workers=0, shuffle=False)
            print(f"  Running on {len(dataset)} images from {IMAGE_DIR}")
            norms = collect_from_loader(model, extractor, loader, device, backend)
        else:
            print(f"  Using single image: {FALLBACK_IMG}")
            norms = collect_single(model, extractor, FALLBACK_IMG,
                                   img_size, device, backend)

        extractor.remove()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        all_norms[key] = norms

        p = norms["patch"]
        print(f"  patch : n={len(p):,}  mean={p.mean():.1f}  "
              f"outliers>150: {(p>150).sum():,} ({100*(p>150).mean():.2f}%)")
        print(f"  cls   : mean={norms['cls'].mean():.1f}")
        for k in sorted(r for r in norms if r.startswith("reg_")):
            print(f"  {k}  : mean={norms[k].mean():.1f}  std={norms[k].std():.1f}")

    # DeiT-III+reg has no public checkpoint — None triggers the placeholder
    if "deit3_reg" not in all_norms:
        all_norms["deit3_reg"] = None

    save_path = "./src/Raffo/img/figure_7_replication.png"
    plot_figure_7(all_norms, save_path=save_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FIGURE 7 — SUMMARY")
    print("=" * 60)
    for key in ["dinov2_noreg","dinov2_reg","openclip_noreg","openclip_reg",
                "deit3_noreg","deit3_reg"]:
        nd = all_norms.get(key)
        if nd is None:
            print(f"\n{key:<22}: no data"); continue
        p = nd["patch"]
        print(f"\n{key:<22}: n={len(p):>7,}  "
              f"outliers>150={( p>150).sum():>6,} ({100*(p>150).mean():.2f}%)  "
              f"cls_mean={nd['cls'].mean():.1f}")
        for k in sorted(r for r in nd if r.startswith("reg_")):
            print(f"  {k}: mean={nd[k].mean():.1f}  std={nd[k].std():.1f}")
    print("=" * 60)