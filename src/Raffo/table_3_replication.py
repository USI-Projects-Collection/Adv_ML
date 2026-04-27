import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import VOCDetection
from PIL import Image
import numpy as np
import scipy.ndimage as ndimage
from tqdm import tqdm
import timm
import open_clip
from transformers import AutoModel

# ==================================================
# FINAL TABLE 3 RESULTS  (VOC 2007 trainval)
# ==================================================
# Model                            CorLoc      Paper
# DINOv2_NoReg                      14.5%      35.3%
# DINOv2_WithReg                    25.0%      55.4%
# OpenCLIP_NoReg                    14.2%      38.8%
# OpenCLIP_TestTimeReg              23.8%      37.1%
# ==========================================
# DEBUGGING UTILITY
# ==========================================
def probe_token_sequence(model, model_backend, device, patch_size=14, img_size=224):
    """
    Runs a dummy forward pass and prints the full feature sequence shape
    so we know exactly how many tokens there are and in what order.
    """
    # Build dummy input directly as a tensor — no PIL transform needed here
    dummy_input = torch.zeros(1, 3, img_size, img_size).to(device)

    features_probe = {}

    # Hook every block to see sequence length evolution
    def make_hook(name):
        def hook(module, input, output):
            x = input[0]
            features_probe[name] = x.shape  # (B, N, C)
        return hook

    if model_backend == "timm":
        handle = model.blocks[-1].attn.register_forward_hook(make_hook("last_attn_input"))
        with torch.no_grad():
            _ = model(dummy_input)
        handle.remove()
    elif model_backend == "open_clip":
        handle = model.visual.transformer.resblocks[-1].attn.register_forward_hook(make_hook("last_attn_input"))
        with torch.no_grad():
            _ = model.encode_image(dummy_input)
        handle.remove()
    elif model_backend == "open_clip_hf":
        # The fix for the Custom Hugging Face wrapper
        handle = model.model.visual.transformer.resblocks[-1].attn.register_forward_hook(make_hook("last_attn_input"))
        with torch.no_grad():
            _ = model.model.encode_image(dummy_input)
        handle.remove()
    elif model_backend == "hf_clip":
        handle = model.vision_model.encoder.layers[-1].self_attn.register_forward_hook(make_hook("last_attn_input"))
        with torch.no_grad():
            _ = model.get_image_features(dummy_input)
        handle.remove()

    seq_len = features_probe["last_attn_input"][1]
    n_patches = (img_size // patch_size) ** 2
    n_special = seq_len - n_patches
    print(f"  [probe] sequence length at last attention: {seq_len}")
    print(f"  [probe] expected patch tokens: {n_patches} ({img_size//patch_size}x{img_size//patch_size})")
    print(f"  [probe] special tokens (CLS + registers): {n_special}")
    return n_special


# ==========================================
# 1. FEATURE EXTRACTOR
# ==========================================
class AttentionFeatureExtractor:
    """
    Extracts keys or values from the last attention layer.

    Token layout (verified by probe_token_sequence):
      - timm ViT (no reg):        [CLS, patch_0, ..., patch_N]
      - timm ViT (with reg):      [CLS, reg_0, ..., reg_R, patch_0, ..., patch_N]
      - open_clip ViT:            [CLS, patch_0, ..., patch_N]
      - HF CLIP (custom reg):     depends on implementation, use probe

    In ALL cases we drop the first `num_special_tokens` from the sequence,
    where num_special_tokens = 1 (CLS) + num_registers.
    For timm-with-registers the registers come right after CLS before the patches,
    so slicing [num_special:] correctly isolates patch tokens.
    """

    def __init__(self, model, feature_type="keys", model_backend="timm"):
        self.model = model
        self.feature_type = feature_type.lower()
        self.features = None
        self.hook_handle = None
        self.model_backend = model_backend
        self._register_hook()

    def _register_hook(self):
        if self.model_backend == "timm":
            last_attn_layer = self.model.blocks[-1].attn
        elif self.model_backend == "open_clip":
            last_attn_layer = self.model.visual.transformer.resblocks[-1].attn
        elif self.model_backend == "open_clip_hf":
            # Custom HF model that wraps open_clip internals
            last_attn_layer = self.model.model.visual.transformer.resblocks[-1].attn
        elif self.model_backend == "hf_clip":
            last_attn_layer = self.model.vision_model.encoder.layers[-1].self_attn
        else:
            raise NotImplementedError(f"Unknown backend: {self.model_backend}")

        def hook(module, input, output):
            x = input[0]  # (B, N, C)
            B, N, C = x.shape

            if self.model_backend in ("timm",):
                # timm attention: module.qkv is a single Linear(C, 3*C)
                qkv = module.qkv(x)  # (B, N, 3*C)
                num_heads = module.num_heads
                head_dim = C // num_heads
                # reshape to (3, B, num_heads, N, head_dim)
                qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                # qkv[0]=Q, qkv[1]=K, qkv[2]=V  shape: (B, num_heads, N, head_dim)
                if self.feature_type == "keys":
                    feat = qkv[1]  # (B, num_heads, N, head_dim)
                else:
                    feat = qkv[2]
                # Merge heads back: (B, N, C)
                feat = feat.permute(0, 2, 1, 3).reshape(B, N, C)
                self.features = feat

            elif self.model_backend in ("open_clip", "open_clip_hf"):
                # open_clip uses F.multi_head_attention_forward internally.
                # module.in_proj_weight shape: (3*C, C)
                # module.in_proj_bias   shape: (3*C,)
                C_in = module.in_proj_weight.shape[1]
                qkv = F.linear(x, module.in_proj_weight, module.in_proj_bias)  # (B, N, 3*C_in)
                num_heads = module.num_heads
                head_dim = C_in // num_heads
                qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                if self.feature_type == "keys":
                    feat = qkv[1]
                else:
                    feat = qkv[2]
                feat = feat.permute(0, 2, 1, 3).reshape(B, N, C_in)
                self.features = feat

            elif self.model_backend == "hf_clip":
                # HuggingFace CLIP SelfAttention has separate k_proj / v_proj
                if self.feature_type == "keys":
                    feat = module.k_proj(x)
                else:
                    feat = module.v_proj(x)
                self.features = feat

        self.hook_handle = last_attn_layer.register_forward_hook(hook)

    def extract(self, x):
        with torch.no_grad():
            if self.model_backend == "timm":
                _ = self.model(x)
            elif self.model_backend == "open_clip":
                _ = self.model.encode_image(x)
            elif self.model_backend == "open_clip_hf":
                _ = self.model.model.encode_image(x)
            elif self.model_backend == "hf_clip":
                _ = self.model.get_image_features(x)
        return self.features  # (B, N_total, C)

    def remove_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()


# ==========================================
# 2. LOST ALGORITHM
# ==========================================
def compute_similarity_matrix(features, bias_value=0.0):
    """
    Gram matrix of L2-normalised features, shape (N, N).

    After normalisation, cosine similarities for DINOv2 keys are all
    positive and tightly clustered near their mean. Thresholding at 0
    therefore makes every patch 'similar' to every other — degrees are
    all equal and seed selection is random.

    The fix is mean-centering: subtract the per-row mean so that patches
    more similar than average get positive values and patches less similar
    than average get negative values. Now threshold=0 is meaningful.

    `bias_value` is applied BEFORE mean-centering as the additive
    correction mentioned in Sec 3.3 for models like OpenCLIP whose
    features have different baseline conditioning.
    """
    features = F.normalize(features, p=2, dim=-1)
    gram = features @ features.T   # (N, N), cosine similarities
    gram = gram + bias_value
    # Mean-center each row: above-average similarity → positive
    gram = gram - gram.mean(dim=-1, keepdim=True)
    return gram


def run_lost_seed_selection(gram_matrix, threshold=0.0):
    """
    Select the seed patch: lowest degree in the thresholded adjacency
    graph (most isolated = most likely foreground).
    With mean-centered gram, threshold=0 means 'above-average similarity',
    so uniform background patches get HIGH degree and distinctive
    foreground patches get LOW degree — exactly what LOST needs.
    Returns (seed_index, mean-centered similarity row of the seed).
    """
    A = (gram_matrix > threshold).float()
    degrees = A.sum(dim=-1)
    seed_index = torch.argmin(degrees)
    return seed_index, gram_matrix[seed_index]


# ==========================================
# 3. BOUNDING BOX & CORLOC
# ==========================================
def extract_bounding_box(similarity_map, grid_size, orig_width, orig_height, threshold=0.0):
    """
    Convert a (grid_size, grid_size) similarity map into a bounding box
    in original image pixel coordinates.
    """
    binary_map = (similarity_map > threshold).astype(int)
    labeled_array, num_features = ndimage.label(binary_map)

    if num_features == 0:
        # Fallback: whole image
        return [0, 0, orig_width, orig_height]

    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0  # ignore background
    largest_label = sizes.argmax()

    objects = ndimage.find_objects((labeled_array == largest_label).astype(int))
    slice_y, slice_x = objects[0]
    grid_ymin, grid_ymax = slice_y.start, slice_y.stop
    grid_xmin, grid_xmax = slice_x.start, slice_x.stop

    scale_x = orig_width  / grid_size
    scale_y = orig_height / grid_size

    xmin = int(grid_xmin * scale_x)
    ymin = int(grid_ymin * scale_y)
    xmax = int(grid_xmax * scale_x)
    ymax = int(grid_ymax * scale_y)

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
# 4. DATASET EVALUATION LOOP
# ==========================================
def evaluate_dataset(config, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Loading Model: {config['name']}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if config["source"] == "timm":
        model = timm.create_model(
            config["model_name"], pretrained=config["pretrained"]
        ).to(device).eval()
        model_backend = "timm"

    elif config["source"] == "open_clip":
        model, _, _ = open_clip.create_model_and_transforms(
            config["model_name"], pretrained=config["pretrained"]
        )
        model = model.to(device).eval()
        model_backend = "open_clip"

    elif config["source"] == "hf":
        model = AutoModel.from_pretrained(
            config["model_name"], trust_remote_code=True
        ).to(device).eval()
        model_backend = "open_clip_hf"

    # ------------------------------------------------------------------
    # Probe to find the true number of special tokens (CLS + registers)
    # This is the KEY fix: we do not hard-code; we measure it.
    # ------------------------------------------------------------------
    img_size   = config["img_size"]
    patch_size = config["patch_size"]
    grid_size  = img_size // patch_size
    n_patches  = grid_size * grid_size

    print("Probing token sequence layout ...")
    num_special_tokens = probe_token_sequence(
        model, model_backend, device,
        patch_size=patch_size, img_size=img_size
    )
    num_registers = num_special_tokens - 1  # subtract the CLS token
    print(f"  => num_special_tokens={num_special_tokens}  "
          f"(1 CLS + {num_registers} register(s))")

    # Sanity check: after dropping special tokens we must have n_patches left
    assert num_special_tokens >= 1, "At least the CLS token must be present"

    # ------------------------------------------------------------------
    # Build extractor
    # ------------------------------------------------------------------
    extractor = AttentionFeatureExtractor(model, feature_type=config["type"],
                                          model_backend=model_backend)

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    correct_localizations = 0
    total_images = len(dataset)
    print(f"Evaluating CorLoc on {total_images} images ...")

    for idx in tqdm(range(total_images)):
        image, target = dataset[idx]
        orig_width, orig_height = image.size

        # Ground-truth boxes
        objects = target['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]
        gt_boxes = []
        for obj in objects:
            bb = obj['bndbox']
            gt_boxes.append([
                int(bb['xmin']), int(bb['ymin']),
                int(bb['xmax']), int(bb['ymax'])
            ])

        # Feature extraction
        input_tensor = transform(image).unsqueeze(0).to(device)
        all_features = extractor.extract(input_tensor)  # (1, N_total, C)
        all_features = all_features[0]                  # (N_total, C)

        # Verify sequence length on first image only
        if idx == 0:
            expected_total = num_special_tokens + n_patches
            actual_total   = all_features.shape[0]
            print(f"\n[First image debug]")
            print(f"  all_features.shape        : {all_features.shape}")
            print(f"  expected total tokens     : {expected_total}")
            print(f"  special tokens to drop    : {num_special_tokens}")
            if actual_total != expected_total:
                print(f"  WARNING: mismatch! {actual_total} != {expected_total}")
            else:
                print(f"  OK: token count matches.")
            # Gram diagnostic on first image to verify mean-centering is working
            pf_d = all_features[num_special_tokens:]
            g_d  = compute_similarity_matrix(pf_d, bias_value=config["bias"])
            print(f"  gram (mean-centered) min={g_d.min():.4f} max={g_d.max():.4f} "
                  f"mean={g_d.mean():.4f} std={g_d.std():.4f}")
            deg_d = (g_d > 0).float().sum(dim=-1)
            print(f"  degrees  min={deg_d.min():.0f} max={deg_d.max():.0f} "
                  f"mean={deg_d.mean():.1f}")
            seed_d = torch.argmin(deg_d).item()
            print(f"  seed={seed_d} "
                  f"(row={seed_d // grid_size}, col={seed_d % grid_size})")

        # -------------------------------------------------------
        # THE KEY FIX:
        # Token layout for timm (both with and without registers):
        #   index 0           : [CLS]
        #   index 1..R        : [REG_1]..[REG_R]   (if registers exist)
        #   index R+1..R+N    : patch tokens
        #
        # So patch_features = all_features[num_special_tokens:]
        # -------------------------------------------------------
        patch_features = all_features[num_special_tokens:]  # (n_patches, C)

        if patch_features.shape[0] != n_patches:
            # Something is still wrong; fall back to whole image prediction
            pred_box = [0, 0, orig_width, orig_height]
        else:
            # LOST algorithm
            gram       = compute_similarity_matrix(patch_features, bias_value=config["bias"])
            seed_idx, corr = run_lost_seed_selection(gram, threshold=0.0)
            map_result = corr.view(grid_size, grid_size).cpu().numpy()
            pred_box   = extract_bounding_box(
                map_result, grid_size, orig_width, orig_height, threshold=0.0
            )

        # CorLoc evaluation
        max_iou = max(compute_iou(pred_box, gt) for gt in gt_boxes)
        if max_iou >= 0.5:
            correct_localizations += 1

    extractor.remove_hook()

    corloc = (correct_localizations / total_images) * 100
    print(f"\nResult for {config['name']}: CorLoc = {corloc:.2f}%")
    return corloc


# ==========================================
# 5. CONFIGS  (matching Table 3 in the paper)
# ==========================================
#
# Paper Table 3 – VOC 2007 reference values:
#   DeiT-III           : 11.7   DeiT-III+reg    : 27.1
#   OpenCLIP           : 38.8   OpenCLIP+reg    : 37.1
#   DINOv2             : 35.3   DINOv2+reg      : 55.4
#
# We replicate DINOv2 (no-reg vs with-reg) and OpenCLIP (no-reg vs test-time reg).
#
# Feature type choices come directly from the paper (Sec 3.3):
#   "For DINOv2, we use KEYS; for DeiT and OpenCLIP, we use VALUES."
# Bias: the paper adds a scalar bias to the gram matrix to handle
#   conditioning differences across models.  0.0 for DINOv2-keys;
#   0.1 is a common value used for OpenCLIP-values.
# ==========================================

CONFIGS = [
    # ---- DINOv2 without registers ----
    {
        "name":       "DINOv2_NoReg",
        "source":     "timm",
        "model_name": "vit_base_patch14_dinov2.lvd142m",
        "pretrained": True,
        "type":       "keys",
        "bias":       0.0,
        "img_size":   518,
        "patch_size": 14,
        # num_special_tokens is now AUTO-DETECTED via probe; ignore this field
    },

    # ---- DINOv2 WITH registers (4 registers, trained-in) ----
    {
        "name":       "DINOv2_WithReg",
        "source":     "timm",
        "model_name": "vit_base_patch14_reg4_dinov2.lvd142m",
        "pretrained": True,
        "type":       "keys",
        "bias":       0.0,
        "img_size":   518,
        "patch_size": 14,
    },

    # ---- OpenCLIP without registers ----
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

    # ---- OpenCLIP with test-time registers (HuggingFace custom model) ----
    {
        "name":       "OpenCLIP_TestTimeReg",
        "source":     "hf",
        "model_name": "amildravid4292/clip-vitb16-test-time-registers",
        "pretrained": True,
        "type":       "values",
        "bias":       0.1,
        "img_size":   224,
        "patch_size": 16,
    },
]


# ==========================================
# 6. MAIN
# ==========================================
if __name__ == "__main__":
    print("Preparing PASCAL VOC 2007 Dataset ...")
    # NOTE: Set download=True the FIRST time you run this (~450 MB).
    voc_dataset = VOCDetection(
        root="./src/Raffo/data",
        year="2007",
        image_set="trainval",
        download=False,   # set True first run
    )

    final_results = {}
    for cfg in CONFIGS:
        score = evaluate_dataset(cfg, voc_dataset)
        final_results[cfg["name"]] = score

    print("\n" + "="*50)
    print("FINAL TABLE 3 RESULTS  (VOC 2007 trainval)")
    print("="*50)
    print(f"{'Model':<30} {'CorLoc':>8}   {'Paper':>8}")
    paper = {
        "DINOv2_NoReg":        35.3,
        "DINOv2_WithReg":      55.4,
        "OpenCLIP_NoReg":      38.8,
        "OpenCLIP_TestTimeReg": 37.1,
    }
    for name, score in final_results.items():
        ref = paper.get(name, float("nan"))
        print(f"{name:<30} {score:>7.1f}%   {ref:>7.1f}%")