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

# ==============================================================================
# APPENDIX C REPLICATION
#
# This script replicates two figures from Appendix C of
# "Vision Transformers Need Registers" (Darcet et al., ICLR 2024):
#
#   Figure 13: Intermediate LOST computation stages for all three model
#              families (DeiT-III, OpenCLIP, DINOv2) with and without
#              registers. Rows = [LOST score, dot prod. w/ seed, seed expansion]
#
#   Figure 14: LOST seed expansion score for OpenCLIP with and without
#              registers, broken down by feature type: keys, queries, values.
#              The key finding is that VALUES suppress artifacts even without
#              registers, while keys and queries do not.
#
# Design choices that match the paper:
#   - DINOv2  → keys   (Sec. 3.3)
#   - OpenCLIP/DeiT-III → values (Sec. 3.3)
#   - Gram matrix bias: 0.0 for DINOv2-keys, 0.1 for OpenCLIP/DeiT-III-values
#   - Seed expansion threshold: 0.0 (mean-centred gram, same as table_3_replication)
# ==============================================================================

# ==========================================
# 1. FEATURE EXTRACTOR  (keys / queries / values)
# ==========================================
class AttentionFeatureExtractor:
    """
    Extracts Q, K, or V features from the last attention layer.
    Supports timm (DINOv2/DeiT-III) and open_clip (OpenCLIP) backends.

    Token layout at last attention input:
      timm, no reg  : [CLS, patch_0 … patch_N]
      timm, 4 reg   : [CLS, reg_0 … reg_3, patch_0 … patch_N]
      open_clip     : [CLS, patch_0 … patch_N]
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
            # timm: hook the QKV Linear directly (reads its output, no recompute)
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
            # For Keys and Queries: recompute from in_proj_weight on the
            # pre-LN input — this correctly shows the raw projection differences.
            #
            # For Values: DO NOT recompute from in_proj_weight.
            # Reason: inputs[0] to MultiheadAttention is already LayerNorm-normalised
            # (OpenCLIP uses pre-norm architecture). LayerNorm erases the norm
            # difference between outlier and normal patches, so V = W_V @ LN(x)
            # looks uniform across all patches — the null-space effect disappears.
            # The paper's claim ("values suppress artifacts") refers to the
            # ATTENDED output — softmax(QK^T/√d) · V — which is the full block
            # output. We hook the ResidualAttentionBlock output for values only.

            if self.feature_type == "values":
                # Hook the full ResidualAttentionBlock output (post-attention + residual)
                if self.model_backend == "open_clip":
                    block = self.model.visual.transformer.resblocks[-1]
                else:  # open_clip_hf
                    block = self.model.model.visual.transformer.resblocks[-1]

                def hook_values(module, input, output):
                    # output: (B, N, C) — attended representation after residual add
                    self.features = output

                self.hook_handle = block.register_forward_hook(hook_values)

            else:
                # Keys or Queries: recompute from in_proj_weight (correct for these)
                if self.model_backend == "open_clip":
                    last_attn = self.model.visual.transformer.resblocks[-1].attn
                else:
                    last_attn = self.model.model.visual.transformer.resblocks[-1].attn

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
# 2. LOST ALGORITHM — three intermediate outputs
# ==========================================
def compute_similarity_matrix(features: torch.Tensor, bias: float = 0.0) -> torch.Tensor:
    """
    Returns mean-centred cosine-similarity Gram matrix (N, N).
    Mean-centering makes threshold=0 meaningful: patches more similar than
    the average get positive values; distinctive patches get negative values.
    """
    features = F.normalize(features, p=2, dim=-1)
    gram = features @ features.T + bias
    gram = gram - gram.mean(dim=-1, keepdim=True)   # mean-centre per row
    return gram


def lost_intermediate_stages(patch_features: torch.Tensor, bias: float = 0.0):
    """
    Runs LOST and returns ALL three intermediate maps that Figure 13 shows:

      1. lost_score      — inverse-degree map: value = 1 / (degree + 1e-6)
                           High = few similar neighbours = candidate foreground.
      2. dot_prod_seed   — raw similarity row of the selected seed patch.
      3. seed_expansion  — binary mask: patches above the threshold.

    Returns:
        lost_score      (N,)  float
        dot_prod_seed   (N,)  float
        seed_expansion  (N,)  float  (0/1)
        seed_idx        int
    """
    gram = compute_similarity_matrix(patch_features, bias=bias)

    # --- Stage 1: LOST score (inverse degree) ---
    A = (gram > 0.0).float()
    degrees = A.sum(dim=-1)                         # (N,)
    # Invert so that "most isolated" = highest score (matches paper's Fig 13 row 1)
    lost_score = 1.0 / (degrees + 1e-6)

    # --- Stage 2: seed selection ---
    seed_idx = int(torch.argmin(degrees).item())
    dot_prod_seed = gram[seed_idx]                  # (N,)

    # --- Stage 3: seed expansion ---
    seed_expansion = (dot_prod_seed > 0.0).float()  # (N,)

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
        + "\nRun  timm.list_models('*reg*patch16*', pretrained=True)  "
        "to find a valid alternative and add it to the candidates list."
    )


def load_model(cfg: dict, device: torch.device):
    """Loads a model and returns (model, model_backend, num_special_tokens)."""
    source = cfg["source"]

    if source == "timm":
        # Support an ordered list of candidate names so the script stays
        # robust across timm versions (the pretrained tag changes between releases).
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
    """Extract patch-only features for a given feature type."""
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
    # NOTE: The paper used its OWN retrained DeiT-III with/without registers.
    # We approximate DeiT-III here with the publicly available timm checkpoint.
    configs = [
        # DeiT-III (label-supervised)
        {
            "label": "DeiT-III\nw/o REG",
            "source": "timm",
            # Priority-ordered list — first available checkpoint wins.
            "model_name": [
                "deit3_base_patch16_224.fb_in22k_ft_in1k",
                "deit3_base_patch16_224.fb_in1k",
                "deit3_base_patch16_384.fb_in22k_ft_in1k",
            ],
            "pretrained": True,
            "type": "values",    # paper: values for DeiT/OpenCLIP
            "bias": 0.1,
            "img_size": 224,
            "patch_size": 16,
            "regs": 0,
        },
        {
            "label": "ViT-B/16\nw/ REG\n(proxy)",
            "source": "timm",
            # ----------------------------------------------------------------
            # NOTE ON DeiT-III+reg:
            # The paper trained its own DeiT-III+reg model and never released
            # the checkpoint publicly. No timm pretrained DeiT-III+reg exists.
            #
            # We use the best available PUBLIC timm ViT-B/16+reg4 checkpoint
            # as a proxy so that the "with registers" column is meaningful.
            # The model_name field is a priority-ordered list: the first name
            # found in your installed timm version will be used automatically.
            # ----------------------------------------------------------------
            "model_name": [
                # EVA-02 ViT-B with 4 registers — same patch size, same reg count
                # "eva02_base_patch16_clip_224.merged2b",
                # Generic ViT-B with registers from SBB training
                "vit_base_patch16_reg4_gap_256.sbb_in12k_ft_in1k",
                "vit_base_patch16_reg8_gap_256.sbb2_in12k_ft_in1k",
                "vit_base_patch16_reg4_gap_256.sbb2_in12k_ft_in1k",
                # Fall back to DINOv2-B with registers (different patch size but same mechanism)
                "vit_base_patch14_reg4_dinov2.lvd142m",
            ],
            "pretrained": True,
            "type": "values",
            "bias": 0.1,
            "img_size": 224,
            "patch_size": 16,
            "regs": 4,
        },
        # OpenCLIP (text-supervised)
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
        # DINOv2 (self-supervised)
        {
            "label": "DINOv2\nw/o REG",
            "source": "timm",
            "model_name": "vit_base_patch14_dinov2.lvd142m",
            "pretrained": True,
            "type": "keys",   # paper: keys for DINOv2
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
    row_labels = ["LOST\nscore", "Dot prod.\nw/ seed", "Seed\nexpansion"]
    cmaps = ["viridis", "viridis", "gray"]

    # Collect results: list of (lost_score_map, dot_prod_map, seed_exp_map)
    all_results = []
    all_images  = []

    for cfg in configs:
        print(f"  Loading {cfg['label'].replace(chr(10), ' ')} ...")
        
        # 1. Load the model FIRST
        model, backend, num_special = load_model(cfg, device)
        
        # 2. Dynamically fetch required resolution to prevent crashes
        if backend == "timm":
            # timm models store their required input size in default_cfg
            img_size = model.default_cfg['input_size'][1]
            patch_size = model.patch_embed.patch_size[0]
        else:
            # OpenCLIP and HF use standard 224x224
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
            if col == 0:
                ax.set_ylabel(row_label, fontsize=9, rotation=0,
                              labelpad=55, va="center")

    fig.suptitle(
        "Figure 13 – LOST Intermediate Computations (Appendix C)\n"
        "Adding registers improves all LOST stages for DeiT-III and DINOv2.\n"
        "The difference is less striking for OpenCLIP (values filter outliers).",
        fontsize=10, y=0.97
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
    Cols = [keys, queries, values]

    Key finding to verify:
      - Keys and queries w/o REG show clear artifact spots in the background.
      - Values w/o REG are already smooth — outliers live in the null space
        of the value projection layer (paper Appendix C, last paragraph).
      - With REG, all three feature types become clean.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Figure 14] Using device: {device}")

    feature_types = ["keys", "queries", "values"]
    img_size   = 224
    patch_size = 16
    grid_size  = img_size // patch_size
    bias       = 0.1      # same as OpenCLIP in Table 3

    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

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
        _, dot_prod, seed_exp, _ = lost_intermediate_stages(patch_feats, bias=bias)
        # Figure 14 shows seed expansion maps
        results_no_reg[ft] = seed_exp.reshape(grid_size, grid_size)
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
        _, dot_prod, seed_exp, _ = lost_intermediate_stages(patch_feats, bias=bias)
        results_reg[ft] = seed_exp.reshape(grid_size, grid_size)
    del model_reg

    # ---- Plot ----
    rows = [
        ("w/o REG", results_no_reg),
        ("w/ REG",  results_reg),
    ]
    col_labels = ["Keys", "Queries", "Values"]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    fig.subplots_adjust(hspace=0.08, wspace=0.05, top=0.88, bottom=0.04,
                        left=0.12, right=0.99)

    for row_i, (row_label, result_dict) in enumerate(rows):
        for col_i, ft in enumerate(feature_types):
            ax = axes[row_i, col_i]
            ax.imshow(result_dict[ft], cmap="viridis", interpolation="nearest",
                      vmin=0, vmax=1)
            ax.axis("off")
            if row_i == 0:
                ax.set_title(col_labels[col_i], fontsize=11, fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(row_label, fontsize=10, rotation=0,
                              labelpad=45, va="center")

    # Annotate the key finding from the paper
    fig.text(
        0.5, 0.01,
        "Values (right column) suppress artifacts even without registers —\n"
        "outliers appear to live in the null space of the value projection layer.",
        ha="center", va="bottom", fontsize=8.5, style="italic",
        color="#444444"
    )

    fig.suptitle(
        "Figure 14 – OpenCLIP Seed Expansion: Keys / Queries / Values (Appendix C)",
        fontsize=11, fontweight="bold"
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[Figure 14] Saved → {save_path}")
    plt.show()


# ==========================================
# 6. MAIN
# ==========================================
if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Update this path to point to your dog (or any other) image.
    # Both figures use the same input image, matching the paper's style.
    # ----------------------------------------------------------------
    IMAGE_PATH = "./src/Raffo/img/Black_Labrador_Retriever_portrait.jpg"

    print("=" * 60)
    print("Appendix C Replication")
    print("  Figure 13 — LOST intermediate steps (all 3 model families)")
    print("  Figure 14 — OpenCLIP keys / queries / values comparison")
    print("=" * 60)

    # Figure 13: ~6 model loads, each forward pass on a single image.
    # Expect ~2-5 min on CPU per model; use GPU for speed.
    run_figure_13(IMAGE_PATH, save_path="figure_13_replication.png")

    # Figure 14: 2 model loads × 3 feature types = 6 extractions.
    run_figure_14(IMAGE_PATH, save_path="figure_14_replication.png")