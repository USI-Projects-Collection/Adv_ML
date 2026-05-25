import json
import os
import time

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCDetection
from tqdm import tqdm

import timm
import open_clip
from transformers import AutoModel

# =============================================================================
# CONFIGURATION
# =============================================================================
VOC_ROOT  = "./src/Raffaele/data"            # must contain VOCdevkit/VOC2007 and VOCdevkit/VOC2012
COCO_ROOT = "./src/Raffaele/data/coco"       # must contain images/val2017/ + annotations/instances_val2017.json

BATCH_SIZE  = 32    # images per GPU forward pass
NUM_WORKERS = 4     # DataLoader CPU workers
USE_FP16    = True  # mixed precision on GPU


# =============================================================================
# 1. DATASETS
# =============================================================================

class COCO20kDataset(Dataset):
    """
    First 20,000 annotated images of COCO val2017, matching the paper

    Expected directory layout:
      <COCO_ROOT>/images/val2017/<file>.jpg
      <COCO_ROOT>/annotations/instances_val2017.json
    """
    def __init__(self, coco_root: str, max_images: int = 20_000):
        ann_path = os.path.join(coco_root, "annotations", "instances_val2017.json")
        img_dir  = os.path.join(coco_root, "images", "val2017")

        with open(ann_path) as f:
            data = json.load(f)

        # Build image_id → boxes mapping (x1,y1,x2,y2 format, crowd excluded)
        id2boxes: dict[int, list] = {}
        for ann in data["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            id2boxes.setdefault(ann["image_id"], []).append(
                [x, y, x + w, y + h]
            )

        id2info = {img["id"]: img for img in data["images"]}

        # Keep only images that have at least one non-crowd annotation
        valid_ids = [
            img["id"] for img in data["images"] if img["id"] in id2boxes
        ][:max_images]

        self.samples = []
        for iid in valid_ids:
            path = os.path.join(img_dir, id2info[iid]["file_name"])
            if os.path.exists(path):
                self.samples.append((path, id2boxes[iid]))

        print(f"  [COCO20k] {len(self.samples)} images "
              f"(requested {max_images})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, boxes = self.samples[idx]
        return Image.open(path).convert("RGB"), boxes  # boxes already xyxy


def _parse_voc_boxes(annotation: dict) -> list[list[int]]:
    """Extract [x1,y1,x2,y2] boxes from a VOCDetection annotation dict."""
    objs = annotation["annotation"]["object"]
    if not isinstance(objs, list):
        objs = [objs]
    boxes = []
    for obj in objs:
        bb = obj["bndbox"]
        boxes.append([
            int(bb["xmin"]), int(bb["ymin"]),
            int(bb["xmax"]), int(bb["ymax"]),
        ])
    return boxes


def _collate_pil(batch):
    """Return (list[PIL], list[gt]) without tensor-stacking PIL images."""
    images, targets = zip(*batch)
    return list(images), list(targets)


# =============================================================================
# 2. FEATURE EXTRACTOR HOOK
# =============================================================================

class AttentionFeatureExtractor:
    """
    Registers a forward hook on the last attention layer and captures the raw
    key or value projections.
    """

    def __init__(self, model, feature_type: str, backend: str):
        self.backend      = backend
        self.feature_type = feature_type.lower()  # "keys" or "values"
        self.features     = None
        self._handle      = None
        self._register(model)

    def _register(self, model):
        feat_idx = {"keys": 1, "values": 2}[self.feature_type]

        if self.backend == "timm":
            # Hook the QKV linear directly — output is (B, N, 3C)
            qkv_linear = model.blocks[-1].attn.qkv
            num_heads  = model.blocks[-1].attn.num_heads

            def _hook(module, inputs, output):
                B, N, C3 = output.shape
                C = C3 // 3
                D = C  // num_heads
                qkv  = output.reshape(B, N, 3, num_heads, D).permute(2, 0, 3, 1, 4)
                feat = qkv[feat_idx].permute(0, 2, 1, 3).reshape(B, N, C)
                self.features = feat

            self._handle = qkv_linear.register_forward_hook(_hook)

        elif self.backend in ("open_clip", "open_clip_hf"):
            if self.backend == "open_clip":
                last_attn = model.visual.transformer.resblocks[-1].attn
            else:
                last_attn = model.model.visual.transformer.resblocks[-1].attn

            def _hook(module, inputs, output):
                x = inputs[0]
                B, N, _ = x.shape
                C_in      = module.in_proj_weight.shape[1]
                num_heads = module.num_heads
                D         = C_in // num_heads
                qkv  = F.linear(x, module.in_proj_weight, module.in_proj_bias)
                qkv  = qkv.reshape(B, N, 3, num_heads, D).permute(2, 0, 3, 1, 4)
                feat = qkv[feat_idx].permute(0, 2, 1, 3).reshape(B, N, C_in)
                self.features = feat

            self._handle = last_attn.register_forward_hook(_hook)

        else:
            raise NotImplementedError(f"Unknown backend: {self.backend}")

    def forward(self, model, x: torch.Tensor):
        """Run a forward pass and return (B, N_total, C) features."""
        with torch.no_grad():
            if self.backend == "timm":
                model(x)
            elif self.backend == "open_clip":
                model.encode_image(x)
            elif self.backend == "open_clip_hf":
                model.model.encode_image(x)
        return self.features

    def remove(self):
        if self._handle:
            self._handle.remove()


# =============================================================================
# 3. LOST ALGORITHM
# =============================================================================

def _gram_matrix(patch_feats: torch.Tensor, bias: float) -> torch.Tensor:
    """
    Compute the cosine similarity Gram matrix with bias and per-row centering: 
    "Is patch j MORE similar to patch i than patch i's average?"
    Without it, raw cosine similarities are all positive and 
    argmin(degree) picks index 0 always, and CorLoc collapses to ~30% (always predicting the top-left corner of the image).

    @params
        - patch_feats : (N, C) tensor of patch features (keys or values)
        - bias        : scalar bias to add to cosine similarities before centering - a higher bias makes the threshold more permissive
    
    @returns
        - G : (N, N) tensor of the centered cosine similarity matrix
    """
    feats = F.normalize(patch_feats.float(), p=2, dim=-1)   # (N, C)
    G     = feats @ feats.T + bias                           # (N, N)
    G     = G - G.mean(dim=1, keepdim=True)                  # per-row center
    return G


def _lost(patch_feats: torch.Tensor, bias: float,
          grid_h: int, grid_w: int,
          orig_w: int, orig_h: int) -> list[int]:
    """
    Run LOST on a single image's patch features and return a predicted
    bounding box [x1, y1, x2, y2] in original image pixel coordinates.

    @params
        - patch_feats : (N, C) tensor of patch features (keys or values)
        - bias        : scalar bias to add to cosine similarities before centering - a higher bias makes the threshold more permissive
        - grid_h, grid_w : height and width of the patch grid
        - orig_w, orig_h : original image width and height in pixels

    @returns
        - box : [x1, y1, x2, y2] in original image pixel coordinates
    """
    # 1. Build mean-centred gram matrix G.
    G = _gram_matrix(patch_feats, bias)   # (N, N) on CPU

    # 2. Threshold at 0 → binary adjacency matrix A.
    A      = (G > 0).float()
    # 3. Degree = number of neighbors for each patch.
    degree = A.sum(dim=-1)                     # (N,)
    # 4. Seed = patch with lowest degree (most isolated).
    seed   = int(degree.argmin().item())

    # Step 5: seed expansion
    expansion = (G[seed] > 0).numpy()         # (N,) bool

    # Step 6: reshape to spatial grid and find bounding box
    spatial = expansion.reshape(grid_h, grid_w).astype(np.int32)
    labeled, n_components = ndimage.label(spatial)

    if n_components == 0:
        return [0, 0, orig_w, orig_h]

    # Largest connected component
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # zero label is background, ignore it
    largest  = int(sizes.argmax())
    slices   = ndimage.find_objects(
        (labeled == largest).astype(np.int32)
    )[0]
    slice_y, slice_x = slices

    scale_x = orig_w / grid_w
    scale_y = orig_h / grid_h
    x1 = int(slice_x.start * scale_x)
    y1 = int(slice_y.start * scale_y)
    x2 = int(slice_x.stop  * scale_x)
    y2 = int(slice_y.stop  * scale_y)
    return [x1, y1, x2, y2]


def _iou(boxA: list, boxB: list) -> float:
    """
    Compute Intersection over Union (IoU) of two boxes in [x1,y1,x2,y2] format.

    @params
        - boxA, boxB : [x1, y1, x2, y2]

    @returns
        - iou : scalar IoU value between 0 and 1
    """
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(area_a + area_b - inter)


# =============================================================================
# 4. MODEL LOADING
# =============================================================================

def _resolve_timm_model(candidates: list[str]) -> str:
    """
    Return the first candidate name that exists as a pretrained timm checkpoint.
    Raises RuntimeError if none are found, with a helpful message.

    @params
        - candidates : list of timm model names in priority order (most preferred first)
    
    @returns
        - model_name : the first candidate that is available as a pretrained checkpoint in the current timm version
    """
    available = set(timm.list_models(pretrained=True))
    for name in candidates:
        if name in available:
            return name
    raise RuntimeError(
        f"None of the candidate timm models are available as pretrained "
        f"checkpoints in your timm version ({timm.__version__}):\n"
        + "\n".join(f"  - {n}" for n in candidates)
        + "\nRun  timm.list_models('*reg*patch16*', pretrained=True)  "
        "to find a valid alternative."
    )


def load_model(cfg: dict, device: torch.device):
    """
    Load model according to cfg['source'] and return
    (model, extractor, num_special_tokens, transform).

    @params
        - cfg : model config dict with keys

    @returns
        - model              : the loaded model, in eval mode on the specified device
        - extractor         : an AttentionFeatureExtractor hooked to the model's last attention layer, configured to extract the specified feature type (keys or values)
        - num_special_tokens : number of special tokens (CLS + registers) at the start of the sequence, which should be ignored when indexing patch features
        - transform         : a torchvision transform to preprocess input images for the model
        - n_patches          : number of patches (N) expected by the model's patch embedding, used for LOST's grid reshaping
        - grid_size          : height and width of the patch grid (sqrt(N)), used for LOST's grid reshaping
    """
    source     = cfg["source"]
    img_size   = cfg["img_size"]
    patch_size = cfg["patch_size"]
    n_patches  = (img_size // patch_size) ** 2

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    if source == "timm":
        model_name = cfg["model_name"]
        if isinstance(model_name, list):
            model_name = _resolve_timm_model(model_name)
            print(f"    [timm] resolved checkpoint → {model_name}")
        model   = timm.create_model(model_name, pretrained=True)
        model   = model.to(device).eval()
        backend = "timm"
        num_special = 1 + cfg.get("regs", 0)   # CLS + registers

        img_size   = model.default_cfg["input_size"][1]
        patch_size = model.patch_embed.patch_size[0]
        n_patches  = (img_size // patch_size) ** 2
        transform  = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        print(f"    [timm] img_size={img_size}, patch_size={patch_size}, "
              f"n_patches={n_patches}, num_special={num_special}")

    elif source == "open_clip":
        model, _, _ = open_clip.create_model_and_transforms(
            cfg["model_name"], pretrained=cfg["pretrained"]
        )
        model   = model.to(device).eval()
        backend = "open_clip"
        num_special = 1   # CLS only

    elif source == "hf":
        model   = AutoModel.from_pretrained(
            cfg["model_name"], trust_remote_code=True
        )
        model   = model.to(device).eval()
        backend = "open_clip_hf"
        # Test-time register model exposes num_register_tokens
        num_special = 1 + getattr(model, "num_register_tokens", 4)

    else:
        raise ValueError(f"Unknown source: {source}")

    extractor = AttentionFeatureExtractor(model, cfg["type"], backend)
    grid_size = int(n_patches ** 0.5)
    return model, extractor, num_special, transform, n_patches, grid_size


# =============================================================================
# 5. EVALUATION LOOP
# =============================================================================

def evaluate(cfg: dict, dataset, ds_name: str, device: torch.device) -> float:
    """
    Run LOST on all images in `dataset` and return CorLoc (%).
    Images are forwarded in batches for speed; LOST is run per-image on CPU.

    @params
        - cfg : model config dict, used to load the model and extractor
        - dataset : a PyTorch Dataset that returns (PIL image, targets) pairs
        - ds_name : name of the dataset, used for printing results

    @returns
        - corloc : CorLoc percentage over the dataset
    """
    print(f"\n  [{cfg['name']}] evaluating on {ds_name} ...")
    t0 = time.time()

    model, extractor, num_special, transform, n_patches, grid_size = load_model(cfg, device)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=_collate_pil,
        pin_memory=(device.type == "cuda"),
    )

    correct = 0
    total   = 0
    fp16_ctx = (torch.autocast("cuda", dtype=torch.float16)
                if USE_FP16 and device.type == "cuda"
                else torch.autocast("cpu", enabled=False))

    for images, targets in tqdm(loader, desc=f"    {ds_name}", leave=False):
        # Stack images into a batch tensor 
        batch = torch.stack([transform(img) for img in images]).to(device)

        # Forward pass
        with fp16_ctx:
            feats = extractor.forward(model, batch)

        feats_cpu = feats.float().cpu()

        for i, (image, raw) in enumerate(zip(images, targets)):
            orig_w, orig_h = image.size

            # Parse ground-truth boxes
            if isinstance(raw, dict):
                gt_boxes = _parse_voc_boxes(raw)
            else:
                gt_boxes = raw

            patch_feats = feats_cpu[i, num_special:]

            # Sanity check — skip malformed images
            if patch_feats.shape[0] != n_patches:
                pred_box = [0, 0, orig_w, orig_h]
            else:
                pred_box = _lost(
                    patch_feats, cfg["bias"],
                    grid_size, grid_size,
                    orig_w, orig_h,
                )

            max_iou = max(_iou(pred_box, gt) for gt in gt_boxes)
            if max_iou >= 0.5:
                correct += 1
            total += 1

    extractor.remove()
    del model, extractor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    corloc  = correct / total * 100
    elapsed = time.time() - t0
    print(f"    CorLoc = {corloc:.1f}%  ({elapsed:.0f}s, {total} images)")
    return corloc


# =============================================================================
# 6. MODEL CONFIGS
# =============================================================================
# Paper rules:
#   DINOv2    → keys,   bias=0.0,  ViT-Large
#   OpenCLIP  → values, bias=0.1,  ViT-B/16
#   DeiT-III  → values, bias=0.1,  ViT-B/16
# =============================================================================

CONFIGS = [
    # ── DeiT-III ─────────────────────────────────────────
    {
        "name":       "DeiT-III",
        "source":     "timm",
        "model_name": "deit3_base_patch16_224.fb_in22k_ft_in1k",
        "type":       "values",
        "bias":       0.1,
        "img_size":   224,
        "patch_size": 16,
        "regs":       0,
    },
    {
        "name":       "DeiT-III+reg (proxy)",
        "source":     "timm",
        "model_name": [
            "vit_base_patch16_reg4_gap_256.sbb_in12k_ft_in1k",
            "vit_base_patch16_reg8_gap_256.sbb2_in12k_ft_in1k",
            "vit_base_patch16_reg4_gap_256.sbb2_in12k_ft_in1k",
            "vit_base_patch14_reg4_dinov2.lvd142m",
        ],
        "type":       "values",
        "bias":       0.1,
        "img_size":   224,
        "patch_size": 16,
        "regs":       4,
    },

    # ── OpenCLIP ───────────────────────────────────────────
    {
        "name":       "OpenCLIP",
        "source":     "open_clip",
        "model_name": "ViT-B-16",
        "pretrained": "laion2b_s34b_b88k",
        "type":       "values",
        "bias":       0.1,
        "img_size":   224,
        "patch_size": 16,
    },
    {
        "name":       "OpenCLIP+reg (test-time proxy)",
        "source":     "hf",
        "model_name": "amildravid4292/clip-vitb16-test-time-registers",
        "type":       "values",
        "bias":       0.1,
        "img_size":   224,
        "patch_size": 16,
    },

    # ── DINOv2 ───────────────────────────────────────────
    {
        "name":       "DINOv2",
        "source":     "timm",
        "model_name": "vit_large_patch14_dinov2.lvd142m",
        "type":       "keys",
        "bias":       0.0,
        "img_size":   518,
        "patch_size": 14,
        "regs":       0,
    },
    {
        "name":       "DINOv2+reg",
        "source":     "timm",
        "model_name": "vit_large_patch14_reg4_dinov2.lvd142m",
        "type":       "keys",
        "bias":       0.0,
        "img_size":   518,
        "patch_size": 14,
        "regs":       4,
    },
]

# Paper reference numbers for the final comparison table
PAPER = {
    "DeiT-III":                       (11.7,  13.1,  10.7),
    "DeiT-III+reg (proxy)":           (27.1,  32.7,  25.1),
    "OpenCLIP":                        (38.8,  44.3,  31.0),
    "OpenCLIP+reg (test-time proxy)":  (37.1,  42.0,  27.9),
    "DINOv2":                          (35.3,  40.2,  26.9),
    "DINOv2+reg":                      (55.4,  60.0,  42.0),
}


# =============================================================================
# 7. MAIN
# =============================================================================

if __name__ == "__main__":

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Device : {device}")
    print(f"FP16   : {USE_FP16 and device.type == 'cuda'}")

    # ── Load datasets ────────────────────────────────────────────────────────
    print("\nLoading datasets ...")
    voc07   = VOCDetection(VOC_ROOT, year="2007", image_set="trainval",
                           download=False)
    voc12   = VOCDetection(VOC_ROOT, year="2012", image_set="trainval",
                           download=True)
    coco20k = COCO20kDataset(COCO_ROOT, max_images=20_000)

    print(f"  VOC 2007 trainval : {len(voc07)} images")
    print(f"  VOC 2012 trainval : {len(voc12)} images")
    print(f"  COCO 20k          : {len(coco20k)} images")

    DATASETS = [
        ("VOC2007", voc07),
        ("VOC2012", voc12),
        ("COCO20k", coco20k),
    ]
    DS_NAMES = [d[0] for d in DATASETS]

    # ── Evaluate ─────────────────────────────────────────────────────────────
    results: dict[str, dict[str, float]] = {}
    t_total = time.time()

    for cfg in CONFIGS:
        name = cfg["name"]
        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"{'='*60}")
        results[name] = {}
        for ds_name, ds in DATASETS:
            results[name][ds_name] = evaluate(cfg, ds, ds_name, device)

    elapsed_total = time.time() - t_total

    # ── Print results table ───────────────────────────────────────────────────
    W   = 35   # model name column width
    COL = 20   # dataset column width

    print("\n" + "=" * (W + COL * 3))
    print("TABLE 3 REPLICATION — CorLoc (%)")
    print("=" * (W + COL * 3))

    header = f"{'Model':<{W}}"
    for ds in DS_NAMES:
        header += f"{'ours / paper':>{COL}}"
    print(header)
    print(f"{'':>{W}}" + "".join(f"{ds:>{COL}}" for ds in DS_NAMES))
    print("-" * (W + COL * 3))

    for cfg in CONFIGS:
        name = cfg["name"]
        row  = f"{name:<{W}}"
        for i, ds in enumerate(DS_NAMES):
            ours = results[name].get(ds, float("nan"))
            ref  = PAPER.get(name, (float("nan"),) * 3)[i]
            cell = f"{ours:.1f} / {ref:.1f}"
            row += f"{cell:>{COL}}"
        print(row)

    print("=" * (W + COL * 3))
    print(f"\nTotal time: {elapsed_total / 60:.1f} min")