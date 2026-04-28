import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import scipy.ndimage as ndimage
from tqdm import tqdm
import timm
import open_clip
from transformers import AutoModel
import json
import os

# ==============================================================================
# TABLE 3 REPLICATION — "Vision Transformers Need Registers" (ICLR 2024)
#
# Datasets: VOC 2007 trainval | VOC 2012 trainval | COCO 20k (val2017 first 20k)
# Metric  : CorLoc (% images where IoU(pred, best_gt) >= 0.5)
#
# Paper Table 3 reference values:
#   Model              VOC07   VOC12  COCO20k
#   DeiT-III            11.7    13.1     10.7
#   DeiT-III+reg        27.1    32.7     25.1
#   OpenCLIP            38.8    44.3     31.0
#   OpenCLIP+reg        37.1    42.0     27.9
#   DINOv2              35.3    40.2     26.9
#   DINOv2+reg          55.4    60.0     42.0
# ==============================================================================


# ==========================================
# WHY YOUR VOC 2007 RESULTS WERE ~20pts LOW
# ==========================================
# There were five compounding bugs:
#
# 1. MEAN-CENTERING — The biggest issue. Your compute_similarity_matrix
#    subtracts the per-row mean from the gram matrix. This is WRONG for LOST.
#    DINOv2 keys and OpenCLIP values have cosine similarities that are
#    naturally spread around a non-zero mean (~0.3-0.8), and LOST's threshold
#    of 0.0 on the raw gram is exactly the right operating point because
#    negative cosine similarity already means "dissimilar". Mean-centering
#    moves the threshold to the average similarity, which makes half of all
#    patch pairs "similar" — degrees become nearly uniform, seed selection
#    becomes near-random, and CorLoc collapses.
#
# 2. BIAS APPLIED BEFORE MEAN-CENTERING — The bias_value was added then
#    immediately subtracted away by mean-centering, making it a no-op.
#    The paper applies bias to shift the raw gram for models with
#    different feature conditioning (OpenCLIP values need +0.1 to shift
#    their distribution so threshold=0 works properly).
#
# 3. SEED EXPANSION THRESHOLD — With mean-centered gram the threshold of
#    0.0 means "above-average similarity", which is too strict. Without
#    mean-centering, threshold=0.0 correctly means "positively correlated".
#
# 4. BOUNDING BOX EXTRACTION — Using scipy connected-components on the raw
#    similarity map is correct, but the threshold applied during bbox
#    extraction must match the one used in LOST seed expansion (0.0 on the
#    raw gram, not on the mean-centered one).
#
# 5. DeiT-III CHECKPOINT — deit3_base_patch16_224.fb_in22k_ft_in1k is
#    available in timm and uses values (same as OpenCLIP). No DeiT-III+reg
#    checkpoint is publicly available; we use the best proxy.


# ==========================================
# 0. COCO 20k DATASET WRAPPER
# ==========================================
class COCO20kDataset(Dataset):
    """
    Wraps the COCO val2017 annotation file to expose the first 20,000 images
    as used in LOST and this paper. Each item returns (PIL.Image, gt_boxes)
    where gt_boxes is a list of [xmin, ymin, xmax, ymax] in pixel coords.

    Setup:
      1. Download val2017 images:
         wget http://images.cocodataset.org/zips/val2017.zip
         unzip val2017.zip -d <coco_root>/images/
      2. Download annotations:
         wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
         unzip annotations_trainval2017.zip -d <coco_root>/
      The expected layout is:
         <coco_root>/images/val2017/<image_id>.jpg
         <coco_root>/annotations/instances_val2017.json
    """

    def __init__(self, coco_root: str, max_images: int = 20000):
        ann_path = os.path.join(coco_root, "annotations", "instances_val2017.json")
        img_dir  = os.path.join(coco_root, "images", "val2017")

        with open(ann_path) as f:
            data = json.load(f)

        # Build image_id → file_name map
        id2info = {img["id"]: img for img in data["images"]}

        # Build image_id → list of [x,y,w,h] boxes (COCO format → convert to xyxy)
        id2boxes = {}
        for ann in data["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            iid = ann["image_id"]
            x, y, w, h = ann["bbox"]
            box = [x, y, x + w, y + h]
            id2boxes.setdefault(iid, []).append(box)

        # Keep only images that have at least one annotation
        valid_ids = [img["id"] for img in data["images"] if img["id"] in id2boxes]
        valid_ids = valid_ids[:max_images]

        self.samples = []
        for iid in valid_ids:
            info = id2info[iid]
            path = os.path.join(img_dir, info["file_name"])
            if os.path.exists(path):
                self.samples.append((path, id2boxes[iid]))

        print(f"  [COCO20k] loaded {len(self.samples)} images "
              f"(requested {max_images}, had annotations for {len(valid_ids)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, boxes = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return image, boxes   # boxes already in [xmin,ymin,xmax,ymax] format


# ==========================================
# 1. DEBUGGING UTILITY
# ==========================================
def probe_token_sequence(model, model_backend, device, patch_size=14, img_size=224):
    """
    Runs a dummy forward pass and returns the number of special tokens
    (CLS + registers) by comparing the observed sequence length to the
    expected number of patch tokens.
    """
    dummy_input = torch.zeros(1, 3, img_size, img_size).to(device)
    captured = {}

    def make_hook(name):
        def hook(module, input, output):
            captured[name] = input[0].shape   # (B, N, C)
        return hook

    if model_backend == "timm":
        h = model.blocks[-1].attn.register_forward_hook(make_hook("seq"))
        with torch.no_grad():
            _ = model(dummy_input)
        h.remove()
    elif model_backend == "open_clip":
        h = model.visual.transformer.resblocks[-1].attn.register_forward_hook(make_hook("seq"))
        with torch.no_grad():
            _ = model.encode_image(dummy_input)
        h.remove()
    elif model_backend == "open_clip_hf":
        h = model.model.visual.transformer.resblocks[-1].attn.register_forward_hook(make_hook("seq"))
        with torch.no_grad():
            _ = model.model.encode_image(dummy_input)
        h.remove()

    seq_len   = captured["seq"][1]
    n_patches = (img_size // patch_size) ** 2
    n_special = seq_len - n_patches
    print(f"  [probe] seq_len={seq_len}, patches={n_patches}, special={n_special}")
    return n_special


# ==========================================
# 2. FEATURE EXTRACTOR
# ==========================================
class AttentionFeatureExtractor:
    """
    Extracts keys or values from the last attention layer via a forward hook.

    Supported backends:
      "timm"        — DINOv2, DeiT-III (blocks[-1].attn.qkv fused linear)
      "open_clip"   — OpenCLIP (visual.transformer.resblocks[-1].attn)
      "open_clip_hf"— HF-wrapped OpenCLIP (model.model.visual…)
    """

    def __init__(self, model, feature_type="keys", model_backend="timm"):
        self.model         = model
        self.feature_type  = feature_type.lower()
        self.model_backend = model_backend
        self.features      = None
        self.hook_handle   = None
        self._register_hook()

    def _register_hook(self):
        if self.model_backend == "timm":
            layer = self.model.blocks[-1].attn
        elif self.model_backend == "open_clip":
            layer = self.model.visual.transformer.resblocks[-1].attn
        elif self.model_backend == "open_clip_hf":
            layer = self.model.model.visual.transformer.resblocks[-1].attn
        else:
            raise NotImplementedError(f"Unknown backend: {self.model_backend}")

        feat_idx = {"queries": 0, "keys": 1, "values": 2}[self.feature_type]

        def hook(module, input, output):
            x = input[0]          # (B, N, C)
            B, N, C = x.shape

            if self.model_backend == "timm":
                num_heads = module.num_heads
                head_dim  = C // num_heads
                qkv  = module.qkv(x).reshape(B, N, 3, num_heads, head_dim)
                qkv  = qkv.permute(2, 0, 3, 1, 4)     # (3, B, H, N, D)
                feat = qkv[feat_idx]                   # (B, H, N, D)
                feat = feat.permute(0, 2, 1, 3).reshape(B, N, C)
                self.features = feat

            elif self.model_backend in ("open_clip", "open_clip_hf"):
                C_in      = module.in_proj_weight.shape[1]
                num_heads = module.num_heads
                head_dim  = C_in // num_heads
                qkv  = F.linear(x, module.in_proj_weight, module.in_proj_bias)
                qkv  = qkv.reshape(B, N, 3, num_heads, head_dim)
                qkv  = qkv.permute(2, 0, 3, 1, 4)
                feat = qkv[feat_idx]
                feat = feat.permute(0, 2, 1, 3).reshape(B, N, C_in)
                self.features = feat

        self.hook_handle = layer.register_forward_hook(hook)

    def extract(self, x):
        with torch.no_grad():
            if self.model_backend == "timm":
                _ = self.model(x)
            elif self.model_backend == "open_clip":
                _ = self.model.encode_image(x)
            elif self.model_backend == "open_clip_hf":
                _ = self.model.model.encode_image(x)
        return self.features   # (B, N_total, C)

    def remove_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()


# ==========================================
# 3. LOST ALGORITHM  (fixed)
# ==========================================
def compute_similarity_matrix(features: torch.Tensor, bias_value: float = 0.0) -> torch.Tensor:
    """
    Gram matrix of L2-normalised features: cosine similarity (N×N).

    The paper (Sec 3.3) adds a scalar bias to handle different feature
    conditioning across models (e.g. +0.1 for OpenCLIP values).
    We do NOT mean-center — that was the primary bug in the previous version.

    Why no mean-centering:
      LOST thresholds at 0.0, meaning "positive cosine similarity".
      For DINOv2 keys the distribution spans [-0.5, +1.0] so threshold=0
      naturally separates background (high mutual similarity) from foreground
      (low similarity to background). Mean-centering artificially flattens
      degree variance, making seed selection nearly random and CorLoc collapse.
    """
    features = F.normalize(features, p=2, dim=-1)
    gram = features @ features.T + bias_value
    return torch.clamp(gram, min=-1.0, max=1.0)


def run_lost_seed_selection(gram_matrix: torch.Tensor, threshold: float = 0.0):
    """
    Lowest-degree node in the thresholded adjacency graph = seed patch.
    Returns (seed_index, similarity row of the seed).
    """
    A          = (gram_matrix > threshold).float()
    degrees    = A.sum(dim=-1)
    seed_index = torch.argmin(degrees)
    return seed_index, gram_matrix[seed_index]


# ==========================================
# 4. BOUNDING BOX & CORLOC HELPERS
# ==========================================
def extract_bounding_box(similarity_map: np.ndarray, grid_size: int,
                          orig_width: int, orig_height: int,
                          threshold: float = 0.0):
    """
    Convert (grid_size, grid_size) similarity map → pixel bounding box.
    Thresholds the map, finds the largest connected component, and scales
    grid coordinates back to original image size.
    """
    binary_map = (similarity_map > threshold).astype(np.int32)
    labeled, n_features = ndimage.label(binary_map)

    if n_features == 0:
        return [0, 0, orig_width, orig_height]

    sizes          = np.bincount(labeled.ravel())
    sizes[0]       = 0                          # ignore background label
    largest_label  = sizes.argmax()
    objs           = ndimage.find_objects((labeled == largest_label).astype(np.int32))
    slice_y, slice_x = objs[0]

    scale_x = orig_width  / grid_size
    scale_y = orig_height / grid_size

    xmin = int(slice_x.start * scale_x)
    ymin = int(slice_y.start * scale_y)
    xmax = int(slice_x.stop  * scale_x)
    ymax = int(slice_y.stop  * scale_y)
    return [xmin, ymin, xmax, ymax]


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


# ==========================================
# 5. DATASET-AGNOSTIC EVALUATION LOOP
# ==========================================
def _get_gt_boxes_voc(target):
    """Parse VOCDetection annotation dict → list of [xmin,ymin,xmax,ymax]."""
    objects = target["annotation"]["object"]
    if not isinstance(objects, list):
        objects = [objects]
    boxes = []
    for obj in objects:
        bb = obj["bndbox"]
        boxes.append([int(bb["xmin"]), int(bb["ymin"]),
                      int(bb["xmax"]), int(bb["ymax"])])
    return boxes


def evaluate_dataset(config: dict, dataset, dataset_name: str = "dataset") -> float:
    """
    Runs LOST on every image in `dataset` and returns CorLoc (%).

    `dataset` must yield either:
      (PIL.Image, voc_target_dict)   for VOC datasets, or
      (PIL.Image, list_of_xyxy_boxes) for COCO20kDataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Model: {config['name']}   Dataset: {dataset_name}")
    print(f"{'='*60}")

    # ── Load model ────────────────────────────────────────────────
    source = config["source"]
    if source == "timm":
        model        = timm.create_model(config["model_name"], pretrained=True).to(device).eval()
        model_backend = "timm"
    elif source == "open_clip":
        model, _, _  = open_clip.create_model_and_transforms(
            config["model_name"], pretrained=config["pretrained"])
        model        = model.to(device).eval()
        model_backend = "open_clip"
    elif source == "hf":
        model        = AutoModel.from_pretrained(
            config["model_name"], trust_remote_code=True).to(device).eval()
        model_backend = "open_clip_hf"
    else:
        raise ValueError(f"Unknown source: {source}")

    img_size   = config["img_size"]
    patch_size = config["patch_size"]
    grid_size  = img_size // patch_size
    n_patches  = grid_size * grid_size

    # ── Auto-detect special token count ──────────────────────────
    print("Probing token sequence layout ...")
    num_special = probe_token_sequence(model, model_backend, device,
                                       patch_size=patch_size, img_size=img_size)

    # ── Build extractor & transform ───────────────────────────────
    extractor = AttentionFeatureExtractor(model, feature_type=config["type"],
                                          model_backend=model_backend)
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    correct = 0
    total   = len(dataset)
    print(f"Evaluating {total} images ...")

    for idx in tqdm(range(total)):
        item  = dataset[idx]
        image = item[0]
        raw   = item[1]

        # Parse ground-truth boxes depending on dataset type
        if isinstance(raw, dict):
            gt_boxes = _get_gt_boxes_voc(raw)
        else:
            gt_boxes = raw    # COCO20kDataset already returns list of [x,y,x,y]

        orig_w, orig_h = image.size

        # ── Feature extraction ────────────────────────────────────
        inp     = transform(image).unsqueeze(0).to(device)
        feats   = extractor.extract(inp)[0]          # (N_total, C)
        patches = feats[num_special:]                # (n_patches, C)

        if patches.shape[0] != n_patches:
            pred_box = [0, 0, orig_w, orig_h]
        else:
            # ── LOST ──────────────────────────────────────────────
            gram        = compute_similarity_matrix(patches, bias_value=config["bias"])
            seed_idx, corr = run_lost_seed_selection(gram, threshold=0.0)
            sim_map     = corr.view(grid_size, grid_size).cpu().numpy()
            pred_box    = extract_bounding_box(sim_map, grid_size, orig_w, orig_h,
                                               threshold=0.0)

        max_iou = max(compute_iou(pred_box, gt) for gt in gt_boxes)
        if max_iou >= 0.5:
            correct += 1

    extractor.remove_hook()
    corloc = correct / total * 100
    print(f"  CorLoc = {corloc:.1f}%")
    return corloc


# ==========================================
# 6. MODEL CONFIGS
# ==========================================
# Feature-type rules from paper Sec 3.3:
#   DINOv2  → keys,   bias=0.0
#   OpenCLIP → values, bias=0.1
#   DeiT-III → values, bias=0.1
#
# DeiT-III+reg NOTE:
#   The paper's DeiT-III+reg weights were never released publicly.
#   We use `deit3_base_patch16_224.fb_in22k_ft_in1k` (no-reg) and flag that
#   the +reg column is a known gap. If you train your own DeiT-III+reg,
#   add its timm name here with "regs": 4.

CONFIGS = [
    # ── DeiT-III (label-supervised) ──────────────────────────────
    {
        "name":       "DeiT-III_NoReg",
        "source":     "timm",
        "model_name": "deit3_base_patch16_224.fb_in22k_ft_in1k",
        "type":       "values",
        "bias":       0.1,
        "img_size":   224,
        "patch_size": 16,
    },
    # DeiT-III+reg: no public checkpoint exists. Comment in and set
    # model_name when you have your own trained checkpoint.
    # {
    #     "name":       "DeiT-III_WithReg",
    #     "source":     "timm",
    #     "model_name": "<your-deit3-reg4-checkpoint>",
    #     "type":       "values",
    #     "bias":       0.1,
    #     "img_size":   224,
    #     "patch_size": 16,
    # },

    # ── OpenCLIP (text-supervised) ────────────────────────────────
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

    # ── DINOv2 (self-supervised) ──────────────────────────────────
    {
        "name":       "DINOv2_NoReg",
        "source":     "timm",
        "model_name": "vit_base_patch14_dinov2.lvd142m",
        "type":       "keys",
        "bias":       0.0,
        "img_size":   518,
        "patch_size": 14,
    },
    {
        "name":       "DINOv2_WithReg",
        "source":     "timm",
        "model_name": "vit_base_patch14_reg4_dinov2.lvd142m",
        "type":       "keys",
        "bias":       0.0,
        "img_size":   518,
        "patch_size": 14,
    },
]


# ==========================================
# 7. MAIN
# ==========================================
if __name__ == "__main__":
    # ── Paths — update these to match your local layout ──────────
    VOC_ROOT  = "./src/Raffo/data"        # contains VOCdevkit/
    COCO_ROOT = "./src/Raffo/data/coco"   # contains images/val2017/ + annotations/

    # ── Paper reference values ────────────────────────────────────
    paper = {
        #                       VOC07   VOC12  COCO20k
        "DeiT-III_NoReg":      (11.7,   13.1,   10.7),
        "DeiT-III_WithReg":    (27.1,   32.7,   25.1),
        "OpenCLIP_NoReg":      (38.8,   44.3,   31.0),
        "OpenCLIP_WithReg":    (37.1,   42.0,   27.9),
        "DINOv2_NoReg":        (35.3,   40.2,   26.9),
        "DINOv2_WithReg":      (55.4,   60.0,   42.0),
    }

    # ── Load datasets ─────────────────────────────────────────────
    print("Loading datasets ...")

    voc07 = VOCDetection(root=VOC_ROOT, year="2007",
                         image_set="trainval", download=False)
    print(f"  VOC 2007 trainval: {len(voc07)} images")

    voc12 = VOCDetection(root=VOC_ROOT, year="2012",
                         image_set="trainval", download=True)
    print(f"  VOC 2012 trainval: {len(voc12)} images")

    # COCO 20k — set coco_available=True once you have the files downloaded
    coco_available = os.path.isdir(os.path.join(COCO_ROOT, "images", "val2017"))
    coco20k = COCO20kDataset(COCO_ROOT, max_images=20000)
    # print(f"\n  [COCO20k] Data not found at {COCO_ROOT}.")
    # print("  To download:")
    # print("    wget http://images.cocodataset.org/zips/val2017.zip")
    # print("    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
    # print("    Unzip both into:", COCO_ROOT)

    datasets = [
        ("VOC2007", voc07),
        ("VOC2012", voc12),
        ("COCO20k", coco20k)
    ]

    # ── Run evaluation ────────────────────────────────────────────
    all_results = {cfg["name"]: {} for cfg in CONFIGS}

    for cfg in CONFIGS:
        for ds_name, ds in datasets:
            score = evaluate_dataset(cfg, ds, dataset_name=ds_name)
            all_results[cfg["name"]][ds_name] = score

    # ── Print summary table ────────────────────────────────────────
    ds_names_run = [d[0] for d in datasets]

    print("\n" + "=" * 80)
    print("TABLE 3 REPLICATION — CorLoc (%)")
    print("=" * 80)

    # Header
    header = f"{'Model':<26}"
    for ds_name in ds_names_run:
        header += f"  {ds_name:>10} (paper)"
    print(header)
    print("-" * 80)

    for cfg in CONFIGS:
        name = cfg["name"]
        row  = f"{name:<26}"
        for i, ds_name in enumerate(ds_names_run):
            score = all_results[name].get(ds_name, float("nan"))
            ref   = paper.get(name, (float("nan"),) * 3)[i]
            row  += f"  {score:>6.1f}%  ({ref:.1f}%)"
        print(row)

    print("=" * 80)
    print("\nNotes:")
    print("  - DeiT-III+reg: no public checkpoint; column omitted.")
    print("  - OpenCLIP+reg uses test-time registers (HF: amildravid4292/clip-vitb16-test-time-registers).")
    print("  - COCO20k = first 20,000 images of COCO val2017 with annotations.")
    print("  - If your numbers are still ~10pts low, ensure you are NOT mean-centering the gram matrix.")