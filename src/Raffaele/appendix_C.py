import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import timm
import open_clip
from transformers import AutoModel

# ==========================================
# 1. FEATURE EXTRACTOR  (keys / queries / values)
# ==========================================
class AttentionFeatureExtractor:
    """
    Extracts Q, K, or V features from the last attention layer.
    Supports timm (DINOv2/DeiT-III) and open_clip (OpenCLIP) backends.
    """

    def __init__(self, model, feature_type: str, model_backend: str):
        self.model = model
        self.feature_type = feature_type.lower()   # "keys" | "queries" | "values"
        self.model_backend = model_backend
        self.features = None
        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        if self.model_backend == "timm":
            last_attn = self.model.blocks[-1].attn

            def hook(module, input, output):
                x = input[0]
                B, N, C = x.shape
                num_heads = module.num_heads
                head_dim  = C // num_heads
                qkv = module.qkv(x)
                qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                idx = {"queries": 0, "keys": 1, "values": 2}[self.feature_type]
                feat = qkv[idx].permute(0, 2, 1, 3).reshape(B, N, C)
                self.features = feat

            self.hook_handle = last_attn.register_forward_hook(hook)

        elif self.model_backend in ("open_clip", "open_clip_hf"):
            if self.model_backend == "open_clip":
                last_attn = self.model.visual.transformer.resblocks[-1].attn
            else:
                last_attn = self.model.model.visual.transformer.resblocks[-1].attn

            if self.feature_type == "values":
                # For values: recompute the full attention-weighted output
                # softmax(QK^T / sqrt(d)) @ V
                def hook_attn_out(module, input, output):
                    x = input[0]
                    B, N, _ = x.shape
                    C_in      = module.in_proj_weight.shape[1]
                    num_heads = module.num_heads
                    head_dim  = C_in // num_heads
                    scale     = head_dim ** -0.5

                    # Project to Q, K, V
                    qkv = F.linear(x, module.in_proj_weight, module.in_proj_bias)
                    qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]

                    # Attention weights: softmax(QK^T / sqrt(d))
                    attn = (q @ k.transpose(-2, -1)) * scale 
                    attn = attn.softmax(dim=-1)

                    # Attended output: softmax(QK^T) @ V → (B, H, N, D)
                    out = (attn @ v)
                    # Reshape to (B, N, C)
                    out = out.permute(0, 2, 1, 3).reshape(B, N, C_in)
                    self.features = out

                self.hook_handle = last_attn.register_forward_hook(hook_attn_out)

            else:
                # Keys and queries: raw projection from in_proj_weight
                def hook_kq(module, input, output):
                    x = input[0]
                    B, N, _ = x.shape
                    C_in      = module.in_proj_weight.shape[1]
                    num_heads = module.num_heads
                    head_dim  = C_in // num_heads
                    qkv = F.linear(x, module.in_proj_weight, module.in_proj_bias)
                    qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    idx = {"queries": 0, "keys": 1}[self.feature_type]
                    feat = qkv[idx].permute(0, 2, 1, 3).reshape(B, N, C_in)
                    self.features = feat

                self.hook_handle = last_attn.register_forward_hook(hook_kq)

        else:
            raise NotImplementedError(f"Unknown backend: {self.model_backend}")

    def extract(self, x):
        """
        Runs a forward pass to trigger the hook and extract features.
        """
        with torch.no_grad():
            if self.model_backend == "timm":
                _ = self.model(x)
            elif self.model_backend == "open_clip":
                _ = self.model.encode_image(x)
            elif self.model_backend == "open_clip_hf":
                _ = self.model.model.encode_image(x)
        return self.features   # (B, N_total, C)

    def remove_hook(self):
        """
        Removes the forward hook to prevent side effects on future inferences.
        """
        if self.hook_handle:
            self.hook_handle.remove()


# ==========================================
# 2. LOST ALGORITHM — three intermediate outputs
# ==========================================
def compute_similarity_matrix(features: torch.Tensor, bias: float = 0.0) -> torch.Tensor:
    """
    Returns mean-centred cosine-similarity Gram matrix (N, N).
    Mean-centering makes threshold=0 meaningful: patches more similar than
    the average get positive values; distinctive patches get negative values.

    @params
        - features: (N, C) tensor of patch features.
        - bias: small positive value to add to the Gram matrix to prevent zero entries

    @returns
        - gram: (N, N) mean-centred cosine similarity matrix.
    """
    features = F.normalize(features, p=2, dim=-1)
    gram = features @ features.T + bias
    gram = gram - gram.mean(dim=-1, keepdim=True)   # mean-centre per row
    return gram


def lost_intermediate_stages(patch_features: torch.Tensor, bias: float = 0.0):
    """
    Runs LOST and returns ALL three intermediate maps:
      1. lost_score      — inverse-degree map: value = 1 / (degree + 1e-6)
                           High = few similar neighbours = candidate foreground.
      2. dot_prod_seed   — raw similarity row of the selected seed patch.
      3. seed_expansion  — binary mask: patches above the threshold.

    @params
        - patch_features: (N, C) tensor of patch features.
        - bias: small positive value to add to the Gram matrix to prevent zero entries

    @returns
        - lost_score      (N,)  float
        - dot_prod_seed   (N,)  float
        - seed_expansion  (N,)  float  (0/1)
        - seed_idx        int
    """
    gram = compute_similarity_matrix(patch_features, bias=bias)

    # --- Stage 1: LOST score (inverse degree) ---
    A = (gram > 0.0).float()
    degrees = A.sum(dim=-1)                        
    # Invert so that "most isolated" = highest score
    lost_score = 1.0 / (degrees + 1e-6)

    # --- Stage 2: seed selection ---
    seed_idx = int(torch.argmin(degrees).item())
    dot_prod_seed = gram[seed_idx]                  

    # --- Stage 3: seed expansion ---
    seed_expansion = (dot_prod_seed > 0.0).float()  

    return (
        lost_score.cpu().numpy(),
        dot_prod_seed.cpu().numpy(),
        seed_expansion.cpu().numpy(),
        seed_idx,
    )


# ==========================================
# 3. MODEL LOADING HELPERS
# ==========================================

def _resolve_timm_model(candidates: list[str]) -> str:
    """
    Given an ordered list of timm model names, return the first one that is
    available as a pretrained checkpoint in the installed version of timm.
    Raises RuntimeError if none are found.
    """
    available = set(timm.list_models(pretrained=True))
    for name in candidates:
        if name in available:
            return name
    raise RuntimeError(
        f"None of the candidate timm models are available as pretrained "
        f"checkpoints in your timm version ({timm.__version__}):\n"
        + "\n".join(f"  - {n}" for n in candidates)
    )


def load_model(cfg: dict, device: torch.device):
    """
    Loads a model and returns (model, model_backend, num_special_tokens).

    @params
        - cfg: dictionary containing model configuration, including:

    @returns
        - model: the loaded PyTorch model, in eval mode and moved to the specified device.
        - model_backend: string indicating the backend type ("timm", "open_clip", or "open_clip_hf").
        - num_special_tokens: int, number of special tokens (CLS + registers) to skip when extracting patch features.
    """
    source = cfg["source"]

    if source == "timm":
        model_name = cfg["model_name"]
        if isinstance(model_name, list):
            model_name = _resolve_timm_model(model_name)
            print(f"    [timm] resolved checkpoint → {model_name}")
        model = timm.create_model(model_name, pretrained=True).to(device).eval()
        backend = "timm"
        num_special = 1 + cfg.get("regs", 0)      # CLS + registers

    elif source == "open_clip":
        model, _, _ = open_clip.create_model_and_transforms(
            cfg["model_name"], pretrained=cfg["pretrained"]
        )
        model = model.to(device).eval()
        backend = "open_clip"
        num_special = 1                            # CLS only

    elif source == "hf":
        model = AutoModel.from_pretrained(cfg["model_name"], trust_remote_code=True)
        model = model.to(device).eval()
        backend = "open_clip_hf"
        num_special = 1 + getattr(model, "num_register_tokens", 4)

    else:
        raise ValueError(f"Unknown source: {source}")

    return model, backend, num_special


def get_patch_features(model, backend: str, num_special: int,
                       input_tensor: torch.Tensor,
                       feature_type: str, bias: float):
    """
    Extract patch-only features for a given feature type.
    
    @params
        - model: the loaded PyTorch model.
        - backend: string indicating the model backend ("timm", "open_clip", or "open_clip_hf").
        - num_special: number of special tokens (CLS + registers) to skip when extracting patch features.
        - input_tensor: preprocessed input image tensor of shape (1, 3, H, W).
        - feature_type: string indicating which feature type to extract ("keys", "queries", or "values").
        - bias: small positive value to add to the Gram matrix to prevent zero entries in LOST computations.

    @returns
        - patch_feats: the extracted patch features of shape (N_patches, C).
    """
    extractor = AttentionFeatureExtractor(model, feature_type, backend)
    feats = extractor.extract(input_tensor)[0]     # (N_total, C)
    extractor.remove_hook()
    patch_feats = feats[num_special:]              # drop CLS + registers
    return patch_feats


# ==========================================
# 4. FIGURE 13 — LOST INTERMEDIATE STEPS
# ==========================================
def run_figure_13(image_path: str, save_path: str = "figure_13_replication.png"):
    """
    Replicates Figure 13 (Appendix C):
    Rows = [LOST score, dot-prod w/ seed, seed expansion]
    Cols = [DeiT-III w/o, DeiT-III w/, OpenCLIP w/o, OpenCLIP w/, DINOv2 w/o, DINOv2 w/]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Figure 13] Using device: {device}")

    # ---- Model configs ----
    configs = [
        # DeiT-III
        {
            "label": "DeiT-III\nw/o REG",
            "source": "timm",
            # Priority-ordered list
            "model_name": [
                "deit3_base_patch16_224.fb_in22k_ft_in1k",
                "deit3_base_patch16_224.fb_in1k",
                "deit3_base_patch16_384.fb_in22k_ft_in1k",
            ],
            "pretrained": True,
            "type": "values",
            "bias": 0.1,
            "img_size": 224,
            "patch_size": 16,
            "regs": 0,
        },
        {
            "label": "ViT-B/16\nw/ REG\n(proxy)",
            "source": "timm",
            "model_name": [
                "vit_base_patch16_reg4_gap_256.sbb_in12k_ft_in1k",
                "vit_base_patch16_reg8_gap_256.sbb2_in12k_ft_in1k",
                "vit_base_patch16_reg4_gap_256.sbb2_in12k_ft_in1k",
                # Fall back to DINOv2-B with registers
                "vit_base_patch14_reg4_dinov2.lvd142m",
            ],
            "pretrained": True,
            "type": "values",
            "bias": 0.1,
            "img_size": 224,
            "patch_size": 16,
            "regs": 4,
        },
        # OpenCLIP
        {
            "label": "OpenCLIP\nw/o REG",
            "source": "open_clip",
            "model_name": "ViT-B-16",
            "pretrained": "laion2b_s34b_b88k",
            "type": "values",
            "bias": 0.1,
            "img_size": 224,
            "patch_size": 16,
            "regs": 0,
        },
        {
            "label": "OpenCLIP\nw/ REG",
            "source": "hf",
            "model_name": "amildravid4292/clip-vitb16-test-time-registers",
            "pretrained": True,
            "type": "values",
            "bias": 0.1,
            "img_size": 224,
            "patch_size": 16,
            "regs": "dynamic",
        },
        # DINOv2
        {
            "label": "DINOv2\nw/o REG",
            "source": "timm",
            "model_name": "vit_base_patch14_dinov2.lvd142m",
            "pretrained": True,
            "type": "keys",
            "bias": 0.0,
            "img_size": 518,
            "patch_size": 14,
            "regs": 0,
        },
        {
            "label": "DINOv2\nw/ REG",
            "source": "timm",
            "model_name": "vit_base_patch14_reg4_dinov2.lvd142m",
            "pretrained": True,
            "type": "keys",
            "bias": 0.0,
            "img_size": 518,
            "patch_size": 14,
            "regs": 4,
        },
    ]

    n_cols = len(configs)
    n_rows = 3
    row_labels = ["", "LOST\nscore", "Dot prod.\nw/ seed", "Seed\nexpansion"]
    cmaps = ["viridis", "viridis", "gray"]

    # Collect results
    all_results = []
    all_images  = []

    for cfg in configs:
        print(f"  Loading {cfg['label'].replace(chr(10), ' ')} ...")
        
        # 1. Load the model
        model, backend, num_special = load_model(cfg, device)
        
        # 2. Dynamically fetch required resolution to prevent crashes
        if backend == "timm":
            img_size = model.default_cfg['input_size'][1]
            patch_size = model.patch_embed.patch_size[0]
        else:
            img_size = cfg["img_size"]
            patch_size = cfg["patch_size"]
            
        grid_size = img_size // patch_size

        # 3. Transform image to the model's native size
        image = Image.open(image_path).convert("RGB")
        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)
        all_images.append(image.resize((img_size, img_size)))

        # 4. Extract features
        patch_feats = get_patch_features(
            model, backend, num_special, input_tensor,
            feature_type=cfg["type"], bias=cfg["bias"]
        )
        del model  # free VRAM before loading next model

        # 5. Run LOST stages
        lost_score, dot_prod, seed_exp, seed_idx = lost_intermediate_stages(
            patch_feats, bias=cfg["bias"]
        )

        all_results.append((
            lost_score.reshape(grid_size, grid_size),
            dot_prod.reshape(grid_size, grid_size),
            seed_exp.reshape(grid_size, grid_size),
        ))
        print(f"    seed at flat index {seed_idx} "
              f"({seed_idx // grid_size}, {seed_idx % grid_size})")

    # ---- Plot ----
    fig = plt.figure(figsize=(3 * n_cols, 3 * (n_rows + 1)))
    gs  = gridspec.GridSpec(
        n_rows + 1, n_cols,
        hspace=0.05, wspace=0.05,
        top=0.92, bottom=0.02, left=0.10, right=0.99
    )

    # Row 0: original images + column headers
    for col, (cfg, img) in enumerate(zip(configs, all_images)):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(cfg["label"], fontsize=9, fontweight="bold", pad=4)

    # Rows 1–3: LOST intermediate maps
    for row_i, (row_label, cmap) in enumerate(zip(row_labels, cmaps)):
        for col, maps in enumerate(all_results):
            ax = fig.add_subplot(gs[row_i + 1, col])
            ax.imshow(maps[row_i], cmap=cmap, interpolation="nearest")
            ax.axis("off")

    # Row labels
    total_height = 0.92 - 0.02          # 0.90
    row_height   = total_height / (n_rows + 1)

    for row_i, row_label in enumerate(row_labels):
        row_bottom = 0.02 + (n_rows - row_i) * row_height
        y_centre   = row_bottom + row_height / 2
        fig.text(
            0.09, y_centre,          
            row_label,
            fontsize=9,
            fontweight="bold",       
            ha="center", va="center",
            rotation=90,             
            multialignment="center",
        )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[Figure 13] Saved → {save_path}")
    plt.show()


# ==========================================
# 5. FIGURE 14 — OPENCLIP KEYS / QUERIES / VALUES
# ==========================================
def run_figure_14(image_path: str, save_path: str = "figure_14_replication.png"):
    """
    Replicates Figure 14 (Appendix C):
    Rows = [w/o REG, w/ REG]
    Cols = [Input, keys, queries, values]
    Shows the dot-product-with-seed map (continuous, not binary) for each
    feature type, matching the paper's smooth viridis visualisation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Figure 14] Using device: {device}")

    feature_types = ["keys", "queries", "values"]
    img_size   = 224
    patch_size = 16
    grid_size  = img_size // patch_size
    bias       = 0.1

    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_img_display = image.resize((img_size, img_size))

    # ---- Load OpenCLIP without registers ----
    print("  Loading OpenCLIP w/o registers ...")
    model_no_reg, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k"
    )
    model_no_reg = model_no_reg.to(device).eval()
    results_no_reg = {}
    for ft in feature_types:
        patch_feats = get_patch_features(
            model_no_reg, "open_clip", 1, input_tensor,
            feature_type=ft, bias=bias
        )
        _, dot_prod, _, _ = lost_intermediate_stages(patch_feats, bias=bias)
        results_no_reg[ft] = dot_prod.reshape(grid_size, grid_size)
    del model_no_reg

    # ---- Load OpenCLIP with test-time registers ----
    print("  Loading OpenCLIP w/ registers (test-time) ...")
    model_reg = AutoModel.from_pretrained(
        "amildravid4292/clip-vitb16-test-time-registers", trust_remote_code=True
    ).to(device).eval()
    num_special_reg = 1 + getattr(model_reg, "num_register_tokens", 4)
    results_reg = {}
    for ft in feature_types:
        patch_feats = get_patch_features(
            model_reg, "open_clip_hf", num_special_reg, input_tensor,
            feature_type=ft, bias=bias
        )
        _, dot_prod, _, _ = lost_intermediate_stages(patch_feats, bias=bias)
        results_reg[ft] = dot_prod.reshape(grid_size, grid_size)
    del model_reg

    # ---- Plot ----
    rows = [
        ("w/o REG", results_no_reg),
        ("w/ REG",  results_reg),
    ]
    col_labels = ["Input", "Keys", "Queries", "Values"]

    n_rows      = 2
    n_cols      = 4
    fig, axes   = plt.subplots(n_rows, n_cols, figsize=(12, 6))
    fig.subplots_adjust(hspace=0.08, wspace=0.05, top=0.88, bottom=0.04,
                        left=0.12, right=0.99)

    for row_i, (_, result_dict) in enumerate(rows):
        # Col 0: input image
        axes[row_i, 0].imshow(input_img_display)
        axes[row_i, 0].axis("off")

        # Cols 1-3: dot-product maps for each feature type
        for col_i, ft in enumerate(feature_types):
            ax = axes[row_i, col_i + 1]
            ax.imshow(result_dict[ft], cmap="viridis", interpolation="nearest")
            ax.axis("off")

        # Column headers on first row only
        if row_i == 0:
            for col_i, label in enumerate(col_labels):
                axes[row_i, col_i].set_title(label, fontsize=11, fontweight="bold")

    # --- Row labels via fig.text ---
    total_height = 0.88 - 0.04
    row_height   = total_height / n_rows

    for row_i, (row_label, _) in enumerate(rows):
        row_bottom = 0.04 + (n_rows - 1 - row_i) * row_height
        y_centre   = row_bottom + row_height / 2
        fig.text(
            0.07, y_centre,
            row_label,
            fontsize=10,
            fontweight="bold",
            ha="center", va="center",
            rotation=90,
            multialignment="center",
        )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[Figure 14] Saved → {save_path}")
    plt.show()


# ==========================================
# 6. MAIN
# ==========================================
if __name__ == "__main__":
    IMAGE_PATH = "./src/Raffaele/img/Black_Labrador_Retriever_portrait.jpg"

    run_figure_13(IMAGE_PATH, save_path="figure_13_replication.png")
    run_figure_14(IMAGE_PATH, save_path="figure_14_replication.png")