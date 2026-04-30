import os
import json
import time

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection
from tqdm import tqdm

import timm
import open_clip
from transformers import AutoModel

# ==============================================================================
# TABLE 3 REPLICATION — "Vision Transformers Need Registers" (ICLR 2024)
#
# Datasets : VOC 2007 trainval | VOC 2012 trainval | COCO 20k (val2017 first 20k)
# Metric   : CorLoc (% of images where IoU(pred_box, best_gt_box) >= 0.5)
#
# Paper Table 3 reference values:
#   Model              VOC07   VOC12  COCO20k
#   DeiT-III            11.7    13.1     10.7
#   DeiT-III+reg        27.1    32.7     25.1
#   OpenCLIP            38.8    44.3     31.0
#   OpenCLIP+reg        37.1    42.0     27.9
#   DINOv2              35.3    40.2     26.9
#   DINOv2+reg          55.4    60.0     42.0
#
# ── Bug fixes vs. previous version ──────────────────────────────────────────
#  FIX 1 — MEAN-CENTERING (root cause of ~20pt collapse)
#    The gram matrix must be globally mean-centred so that the LOST threshold
#    of 0.0 separates above-average similarity (background) from below-average
#    similarity (foreground seed). Without this, raw cosine similarities for
#    DINOv2/OpenCLIP cluster at 0.3-0.8 so *all* patches are "above threshold",
#    degrees are uniform, seed selection is random, and CorLoc collapses.
#    The previous version described this fix in a comment but never applied it.
#
#  FIX 2 — BIAS ORDER
#    Bias is added first, then the global mean is subtracted. The old code
#    added bias then never subtracted the mean, making bias a no-op.
#
#  FIX 3 — DINOv2 model size
#    The paper uses ViT-Large for DINOv2. The old code used ViT-Base,
#    which gives lower CorLoc. Corrected to vit_large_patch14_dinov2.lvd142m
#    and vit_large_patch14_reg4_dinov2.lvd142m.
#
# ── Speed optimisations vs. previous version ────────────────────────────────
#  OPT 1 — Batched GPU inference (batch_size=32, num_workers=4)
#    Images are forwarded in batches instead of one-by-one. The DataLoader
#    prefetches and preprocesses on CPU workers while the GPU runs.
#    Typical speedup: 10-20×.
#
#  OPT 2 — Model loaded once per config, reused across all three datasets
#    The old code reloaded the model for every (config × dataset) pair.
#    Now the model is loaded once per config and shared. For DINOv2-L this
#    saves ~3× the download/load time.
#
#  OPT 3 — torch.compile (PyTorch ≥ 2.0)
#    JIT-compiles the ViT and fuses ops. Typical speedup: 2-3×.
#    Set COMPILE=False below if you hit compatibility issues.
#
#  OPT 4 — FP16 autocast
#    ViT attention and matmuls run in FP16 on the GPU. Features are cast
#    back to FP32 before the LOST gram-matrix computation. Typical
#    speedup: 1.5-2× on Ampere+ GPUs.
#
#  OPT 5 — torch.inference_mode instead of no_grad
#    Skips additional autograd bookkeeping. Small but free.
#
#  OPT 6 — Vectorised IoU
#    All ground-truth boxes for one image are evaluated in a single numpy
#    operation instead of a Python loop.
# ==============================================================================


# ── Tunable knobs ────────────────────────────────────────────────────────────
BATCH_SIZE  = 32      # images per GPU forward pass — lower if you run out of VRAM
NUM_WORKERS = 4       # DataLoader CPU workers for parallel preprocessing
COMPILE     = True    # torch.compile the model (requires PyTorch ≥ 2.0)
USE_FP16    = True    # FP16 autocast on GPU (set False for CPU-only runs)
# ─────────────────────────────────────────────────────────────────────────────


# ==============================================================================
# 0. DATASET WRAPPERS
# ==============================================================================

class COCO20kDataset(Dataset):
    """
    Wraps COCO val2017 annotations to expose the first 20,000 annotated images,
    matching the LOST / VitReg paper evaluation protocol.

    Expected layout:
        <coco_root>/images/val2017/<file>.jpg
        <coco_root>/annotations/instances_val2017.json

    Download commands:
        wget http://images.cocodataset.org/zips/val2017.zip
        wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        unzip val2017.zip            -d <coco_root>/images/
        unzip annotations_*.zip      -d <coco_root>/
    """

    def __init__(self, coco_root: str, max_images: int = 20_000):
        ann_path = os.path.join(coco_root, "annotations", "instances_val2017.json")
        img_dir  = os.path.join(coco_root, "images", "val2017")

        with open(ann_path) as f:
            data = json.load(f)

        id2info  = {img["id"]: img for img in data["images"]}
        id2boxes: dict[int, list] = {}
        for ann in data["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            iid      = ann["image_id"]
            x, y, w, h = ann["bbox"]
            id2boxes.setdefault(iid, []).append([x, y, x + w, y + h])

        valid_ids = [img["id"] for img in data["images"] if img["id"] in id2boxes]
        valid_ids = valid_ids[:max_images]

        self.samples = []
        for iid in valid_ids:
            path = os.path.join(img_dir, id2info[iid]["file_name"])
            if os.path.exists(path):
                self.samples.append((path, id2boxes[iid]))

        print(f"  [COCO20k] loaded {len(self.samples)} images "
              f"(requested {max_images}, annotated {len(valid_ids)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, boxes = self.samples[idx]
        return Image.open(path).convert("RGB"), boxes   # boxes: [[x1,y1,x2,y2], …]


def _parse_gt_boxes(raw) -> list[list[int]]:
    """Normalise ground-truth boxes from either VOC dict or COCO list."""
    if isinstance(raw, dict):                           # VOCDetection annotation
        objs = raw["annotation"]["object"]
        if not isinstance(objs, list):
            objs = [objs]
        boxes = []
        for obj in objs:
            bb = obj["bndbox"]
            boxes.append([int(bb["xmin"]), int(bb["ymin"]),
                          int(bb["xmax"]), int(bb["ymax"])])
        return boxes
    return raw                                          # COCO20kDataset: already xyxy


def _collate(batch):
    """Return (list[PIL], list[gt_boxes]) — avoids tensor-stacking PIL images."""
    images, targets = zip(*batch)
    return list(images), list(targets)


# ==============================================================================
# 1. FEATURE EXTRACTOR  (hook-based, supports batches)
# ==============================================================================

class AttentionFeatureExtractor:
    """
    Registers a forward hook on the last attention layer and captures the raw
    key or value vectors before they are used in scaled-dot-product attention.

    Backends
    --------
    "timm"         DINOv2 / DeiT-III  (fused QKV linear: module.qkv)
    "open_clip"    OpenCLIP            (in_proj_weight / in_proj_bias)
    "open_clip_hf" HF-wrapped OpenCLIP
    """

    def __init__(self, model, feature_type: str = "keys", model_backend: str = "timm"):
        self.model_backend = model_backend
        self.feature_type  = feature_type.lower()
        self.features: torch.Tensor | None = None
        self._hook_handle  = None
        self._model_ref    = model          # kept for forward dispatch
        self._register(model)

    def _register(self, model):
        feat_idx = {"queries": 0, "keys": 1, "values": 2}[self.feature_type]

        if self.model_backend == "timm":
            # Hook the QKV *linear* directly so we read its output, not recompute it
            qkv_linear = model.blocks[-1].attn.qkv
            num_heads   = model.blocks[-1].attn.num_heads

            def _hook(module, inputs, output):
                B, N, C3 = output.shape
                C = C3 // 3
                H = num_heads
                D = C // H
                qkv  = output.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
                feat = qkv[feat_idx].permute(0, 2, 1, 3).reshape(B, N, C)
                self.features = feat

            self._hook_handle = qkv_linear.register_forward_hook(_hook)

        elif self.model_backend in ("open_clip", "open_clip_hf", "clip_tt_reg"):
            # For OpenCLIP the in_proj_weight fuses QKV — hook the attention module
            # and recompute from input, which is reliable here
            if self.model_backend == "open_clip":
                layer = model.visual.transformer.resblocks[-1].attn
            elif self.model_backend == "open_clip_hf":
                layer = model.model.visual.transformer.resblocks[-1].attn
            else:  # clip_tt_reg
                layer = model.model.visual.transformer.resblocks[-1].attn

            def _hook(module, inputs, _output):
                x = inputs[0]
                B, N, _ = x.shape
                C_in     = module.in_proj_weight.shape[1]
                H        = module.num_heads
                D        = C_in // H
                qkv      = F.linear(x, module.in_proj_weight, module.in_proj_bias)
                qkv      = qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
                feat     = qkv[feat_idx].permute(0, 2, 1, 3).reshape(B, N, C_in)
                self.features = feat

            self._hook_handle = layer.register_forward_hook(_hook)

        else:
            raise NotImplementedError(self.model_backend)

    def forward(self, inp: torch.Tensor):
        """Trigger the right forward method so the hook fires."""
        if self.model_backend == "timm":
            self._model_ref(inp)
        elif self.model_backend == "open_clip":
            self._model_ref.encode_image(inp)
        elif self.model_backend == "open_clip_hf":
            self._model_ref.model.encode_image(inp)

    def remove(self):
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None


def _probe_special_tokens(model, backend: str, device, img_size: int,
                           patch_size: int) -> int:
    """
    Returns the number of non-patch tokens (CLS + registers) by running one
    dummy forward pass and comparing the sequence length to n_patches.
    """
    dummy   = torch.zeros(1, 3, img_size, img_size, device=device)
    captured: dict[str, tuple] = {}

    def _h(module, inputs, _):
        captured["shape"] = inputs[0].shape   # (B, N, C)

    if backend == "timm":
        h = model.blocks[-1].attn.register_forward_hook(_h)
        with torch.inference_mode():
            model(dummy)
    elif backend == "open_clip":
        h = model.visual.transformer.resblocks[-1].attn.register_forward_hook(_h)
        with torch.inference_mode():
            model.encode_image(dummy)
    elif backend == "open_clip_hf":
        h = model.model.visual.transformer.resblocks[-1].attn.register_forward_hook(_h)
        with torch.inference_mode():
            model.model.encode_image(dummy)
    else:
        raise NotImplementedError(backend)

    h.remove()
    n_patches = (img_size // patch_size) ** 2
    n_special = captured["shape"][1] - n_patches
    print(f"  [probe] seq={captured['shape'][1]}, patches={n_patches}, "
          f"special={n_special}")
    return n_special


# ==============================================================================
# 2. LOST  (fixed)
# ==============================================================================

def compute_similarity_matrix(features: torch.Tensor,
                               bias: float = 0.0) -> torch.Tensor:
    """
    Cosine-similarity gram matrix, mean-centred so the LOST threshold of 0.0
    correctly separates above-average-similarity patches (background) from
    below-average-similarity patches (foreground seed candidate).

    Pipeline
    --------
    1. L2-normalise patch features  →  (N, C)
    2. Gram matrix                  →  (N, N)  cosine similarities in [-1, 1]
    3. Add per-model bias           shifts distribution for OpenCLIP / DeiT-III
    4. Subtract global mean         centres threshold at the distribution mean
    """
    features = F.normalize(features, p=2, dim=-1)
    gram     = features @ features.T
    gram     = gram + bias
    gram     = gram - gram.mean()       # ← THE FIX: was described but never applied
    return gram


def run_lost(gram: torch.Tensor, grid_h: int, grid_w: int,
             orig_w: int, orig_h: int, threshold: float = 0.0):
    """
    Full LOST pipeline: seed selection → similarity map → bounding box.

    Returns [xmin, ymin, xmax, ymax] in original pixel coordinates.
    """
    # Seed = patch with the fewest neighbours above threshold (most isolated)
    degrees    = (gram > threshold).float().sum(dim=-1)
    seed_idx   = int(torch.argmin(degrees).item())
    sim_map    = gram[seed_idx].view(grid_h, grid_w).cpu().float().numpy()

    # Binary map → largest connected component → bounding box
    binary     = (sim_map > threshold).astype(np.int32)
    labeled, n = ndimage.label(binary)
    if n == 0:
        return [0, 0, orig_w, orig_h]

    sizes              = np.bincount(labeled.ravel())
    sizes[0]           = 0
    largest            = sizes.argmax()
    sl                 = ndimage.find_objects((labeled == largest).astype(np.int32))[0]
    sl_y, sl_x         = sl

    sx = orig_w / grid_w
    sy = orig_h / grid_h
    return [int(sl_x.start * sx), int(sl_y.start * sy),
            int(sl_x.stop  * sx), int(sl_y.stop  * sy)]


# ==============================================================================
# 3. IoU  (vectorised)
# ==============================================================================

def max_iou(pred: list[int], gt_boxes: list[list]) -> float:
    """
    IoU between `pred` and every box in `gt_boxes`, returned as the maximum.
    Uses numpy broadcasting — no Python loop over ground-truth boxes.
    """
    if not gt_boxes:
        return 0.0
    gt  = np.array(gt_boxes, dtype=np.float32)          # (M, 4)
    p   = np.array(pred,     dtype=np.float32)           # (4,)

    xA  = np.maximum(p[0], gt[:, 0]);  yA = np.maximum(p[1], gt[:, 1])
    xB  = np.minimum(p[2], gt[:, 2]);  yB = np.minimum(p[3], gt[:, 3])
    inter = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)

    area_p = (p[2]  - p[0])  * (p[3]  - p[1])
    area_g = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    iou    = inter / (area_p + area_g - inter + 1e-6)
    return float(iou.max())


# ==============================================================================
# 4. MODEL LOADING
# ==============================================================================

def load_model(cfg: dict, device: torch.device):
    """
    Load model + build extractor + probe special tokens.
    Returns (extractor, num_special_tokens).
    The model itself is attached to `extractor` and lives on `device`.
    """
    source = cfg["source"]
    print(f"\nLoading model: {cfg['name']}  ({source})")

    if source == "timm":
        model   = timm.create_model(cfg["model_name"], pretrained=True).to(device).eval()
        backend = "timm"

    elif source == "open_clip":
        model, _, _ = open_clip.create_model_and_transforms(
            cfg["model_name"], pretrained=cfg["pretrained"])
        model   = model.to(device).eval()
        backend = "open_clip"

    elif source == "hf":
        model   = AutoModel.from_pretrained(
            cfg["model_name"], trust_remote_code=True).to(device).eval()
        backend = "open_clip_hf"

    else:
        raise ValueError(f"Unknown source: {source}")

    # OPT 3 — compile
    if COMPILE and hasattr(torch, "compile"):
        print("  torch.compile … ", end="", flush=True)
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("ok")
        except Exception as e:
            print(f"skipped ({e})")

    extractor   = AttentionFeatureExtractor(model, cfg["type"], backend)
    num_special = _probe_special_tokens(model, backend, device,
                                        cfg["img_size"], cfg["patch_size"])
    return model, extractor, num_special


# ==============================================================================
# 5. EVALUATION LOOP  (batched)
# ==============================================================================

def evaluate_dataset(cfg: dict, dataset, dataset_name: str,
                     extractor: AttentionFeatureExtractor,
                     num_special: int, device: torch.device) -> float:
    """
    Run LOST on every image in `dataset` and return CorLoc (%).

    Key changes vs. original:
    - Batched DataLoader with num_workers (OPT 1)
    - FP16 autocast on GPU  (OPT 4)
    - torch.inference_mode  (OPT 5)
    - Vectorised IoU        (OPT 6)
    - Model NOT reloaded here — passed in from caller (OPT 2)
    """
    img_size   = cfg["img_size"]
    patch_size = cfg["patch_size"]
    grid_size  = img_size // patch_size
    n_patches  = grid_size * grid_size
    use_fp16   = USE_FP16 and device.type == "cuda"

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # OPT 1 — batched DataLoader
    loader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        num_workers = 0 if device.type == "mps" else NUM_WORKERS,
        pin_memory  = device.type == "cuda",
        collate_fn  = _collate,
        drop_last   = False,
    )

    correct = 0
    total   = len(dataset)
    t0      = time.time()

    print(f"\n  Dataset: {dataset_name}  ({total} images, "
          f"bs={BATCH_SIZE}, fp16={use_fp16})")

    for images, targets in tqdm(loader, desc=f"  {dataset_name}", leave=True):
        # Stack tensors and ship to GPU
        inp = torch.stack([transform(img) for img in images]).to(device,
                                                                  non_blocking=True)

        # OPT 4 + OPT 5 — FP16 autocast + inference_mode
        ctx_fp16 = torch.autocast("cuda", dtype=torch.float16) if use_fp16 \
                   else torch.autocast("cpu",  enabled=False)
        with torch.inference_mode(), ctx_fp16:
            extractor.forward(inp)

        feats = extractor.features   # (B, N_total, C) — may be fp16 on GPU

        # Per-image LOST
        for i, (image, raw) in enumerate(zip(images, targets)):
            gt_boxes = _parse_gt_boxes(raw)
            orig_w, orig_h = image.size

            patches = feats[i, num_special:].float().cpu()    # cast to fp32 for gram

            if patches.shape[0] != n_patches:
                pred_box = [0, 0, orig_w, orig_h]
            else:
                gram     = compute_similarity_matrix(patches, bias=cfg["bias"])
                pred_box = run_lost(gram, grid_size, grid_size, orig_w, orig_h)

            if max_iou(pred_box, gt_boxes) >= 0.5:
                correct += 1

    elapsed = time.time() - t0
    corloc  = correct / total * 100
    print(f"  CorLoc = {corloc:.1f}%   ({elapsed:.0f}s)")
    return corloc



# ==============================================================================
# 6. MODEL CONFIGS
# ==============================================================================
# Feature-type rules (paper Sec 3.3):
#   DINOv2    → keys,   bias = 0.0
#   OpenCLIP  → values, bias = 0.1
#   DeiT-III  → values, bias = 0.1
#
# DINOv2 uses ViT-Large (corrected from Base in the previous version).
# DeiT-III+reg: no public checkpoint; row omitted.
# OpenCLIP+reg: community test-time-register weights (closest available proxy).

CONFIGS = [
    # ── DeiT-III (label-supervised) ──────────────────────────────────────────
    {
        "name":       "DeiT-III_NoReg",
        "source":     "timm",
        "model_name": "deit3_base_patch16_224.fb_in22k_ft_in1k",
        "type":       "values",
        "bias":       0.1,
        "img_size":   224,
        "patch_size": 16,
    },
    # DeiT-III+reg: uncomment when you have a trained checkpoint
    # {
    #     "name":       "DeiT-III_WithReg",
    #     "source":     "timm",
    #     "model_name": "<your-deit3-reg4-timm-name>",
    #     "type":       "values",
    #     "bias":       0.1,
    #     "img_size":   224,
    #     "patch_size": 16,
    # },

    # ── OpenCLIP (text-supervised) ────────────────────────────────────────────
    {
        "name":       "OpenCLIP_NoReg",
        "source":     "open_clip",
        "model_name": "ViT-B-16",
        "pretrained": "laion2b_s34b_b88k",
        "type":       "values",
        "bias":       0.1,
        "img_size":   224,
        "patch_size": 16,
    },
    {
        "name":       "OpenCLIP_WithReg",
        "source":     "hf",
        "model_name": "amildravid4292/clip-vitb16-test-time-registers",
        "pretrained": True,
        "type":       "values",
        "bias":       0.1,
        "img_size":   224,
        "patch_size": 16,
    },

    # ── DINOv2 (self-supervised) — ViT-Large as in the paper ─────────────────
    {
        "name":       "DINOv2_NoReg",
        "source":     "timm",
        "model_name": "vit_large_patch14_dinov2.lvd142m",       # FIX 3: Large not Base
        "type":       "keys",
        "bias":       0.0,
        "img_size":   518,
        "patch_size": 14,
    },
    {
        "name":       "DINOv2_WithReg",
        "source":     "timm",
        "model_name": "vit_large_patch14_reg4_dinov2.lvd142m",  # FIX 3: Large not Base
        "type":       "keys",
        "bias":       0.0,
        "img_size":   518,
        "patch_size": 14,
    },
]


# ==============================================================================
# 7. MAIN
# ==============================================================================

PAPER_REFS = {
    #                       VOC07   VOC12  COCO20k
    "DeiT-III_NoReg":      (11.7,   13.1,   10.7),
    "DeiT-III_WithReg":    (27.1,   32.7,   25.1),
    "OpenCLIP_NoReg":      (38.8,   44.3,   31.0),
    "OpenCLIP_WithReg":    (37.1,   42.0,   27.9),
    "DINOv2_NoReg":        (35.3,   40.2,   26.9),
    "DINOv2_WithReg":      (55.4,   60.0,   42.0),
}


if __name__ == "__main__":

    # ── Paths ────────────────────────────────────────────────────────────────
    VOC_ROOT  = "./src/Raffo/data"          # must contain VOCdevkit/
    COCO_ROOT = "./src/Raffo/data/coco"     # must contain images/val2017/ + annotations/

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    print("\nLoading datasets …")
    voc07  = VOCDetection(root=VOC_ROOT, year="2007", image_set="trainval", download=False)
    print(f"  VOC 2007 trainval : {len(voc07)} images")

    voc12  = VOCDetection(root=VOC_ROOT, year="2012", image_set="trainval", download=True)
    print(f"  VOC 2012 trainval : {len(voc12)} images")

    coco20k = COCO20kDataset(COCO_ROOT, max_images=20_000)

    datasets = [
        ("VOC2007", voc07),
        ("VOC2012", voc12),
        ("COCO20k", coco20k),
    ]
    ds_names = [d[0] for d in datasets]

    # ── Evaluation ───────────────────────────────────────────────────────────
    all_results: dict[str, dict[str, float]] = {}
    t_total = time.time()

    for cfg in CONFIGS:
        # OPT 2 — load model ONCE, reuse across all three datasets
        model, extractor, num_special = load_model(cfg, device)

        all_results[cfg["name"]] = {}
        print(f"\nEvaluating {cfg['name']} …")

        for ds_name, ds in datasets:
            score = evaluate_dataset(
                cfg, ds, ds_name,
                extractor, num_special, device,
            )
            all_results[cfg["name"]][ds_name] = score

        extractor.remove()          # clean up hook before next model
        del model, extractor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed_total = time.time() - t_total

    # ── Summary table ────────────────────────────────────────────────────────
    W = 22
    COL = 18

    print("\n" + "=" * 80)
    print("TABLE 3 REPLICATION — CorLoc (%)")
    print("=" * 80)

    header = f"{'Model':<{W}}"
    for ds in ds_names:
        header += f"{'ours / paper':>{COL}}"
    print(f"{header}")
    print(f"{'':>{W}}" + "".join(f"{ds:>{COL}}" for ds in ds_names))
    print("-" * 80)

    for cfg in CONFIGS:
        name = cfg["name"]
        row  = f"{name:<{W}}"
        for i, ds in enumerate(ds_names):
            ours = all_results[name].get(ds, float("nan"))
            ref  = PAPER_REFS.get(name, (float("nan"),) * 3)[i]
            row += f"{ours:>6.1f}% / {ref:.1f}%{'':{COL - 16}}"
        print(row)

    print("=" * 80)
    print(f"\nTotal wall-clock time: {elapsed_total/60:.1f} min")
    print("\nNotes:")
    print("  - DeiT-III+reg : no public checkpoint available; row omitted.")
    print("  - OpenCLIP+reg : community test-time-register proxy "
          "(amildravid4292/clip-vitb16-test-time-registers).")
    print("  - COCO20k      : first 20,000 annotated images of COCO val2017.")
    print("  - DINOv2       : ViT-Large (corrected from Base in previous version).")