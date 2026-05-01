"""Reproduce Figure 2 attention-map examples from "Vision Transformers Need Registers".

The script creates a qualitative grid with four input images and last-layer
class-token attention maps for ViT model families discussed in the paper.
CPU mode uses one public checkpoint per family. Exact mode adds larger variants
where public checkpoints are available, but it is intended for a GPU machine.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_KUN_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_KUN_DIR / "data"
DEFAULT_RESULTS_DIR = REPO_KUN_DIR / "results"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SelectedImage:
    """Image and stable source metadata used in the Figure 2 grid."""

    image: Image.Image
    source: str
    label: str


@dataclass(frozen=True)
class AttentionResult:
    """Attention map and metadata produced by one model for one image."""

    map_2d: np.ndarray
    grid_size: tuple[int, int]
    prefix_tokens: int


class TimmAttentionModel:
    """Capture last-layer attention probabilities from a timm ViT model."""

    def __init__(self, label: str, candidate_names: Iterable[str], device: torch.device) -> None:
        try:
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
        except ImportError as exc:
            raise RuntimeError("Install timm to use DeiT-III models.") from exc

        self.label = label
        self.device = device
        self.model_name = None
        last_error: Exception | None = None
        for model_name in candidate_names:
            try:
                self.model = timm.create_model(model_name, pretrained=True).to(device).eval()
                self.model_name = model_name
                break
            except Exception as exc:  # pragma: no cover - depends on installed timm registry.
                last_error = exc
        if self.model_name is None:
            raise RuntimeError(f"Could not load any timm model for {label}: {list(candidate_names)}") from last_error

        for block in getattr(self.model, "blocks", []):
            if hasattr(block.attn, "fused_attn"):
                block.attn.fused_attn = False

        config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**config, is_training=False)

    @torch.no_grad()
    def attention(self, image: Image.Image) -> AttentionResult:
        attn_buffers: list[torch.Tensor] = []
        handle = self.model.blocks[-1].attn.attn_drop.register_forward_hook(
            lambda _module, _inp, out: attn_buffers.append(out.detach())
        )
        try:
            x = self.transform(image).unsqueeze(0).to(self.device)
            _ = self.model(x)
        finally:
            handle.remove()

        if not attn_buffers:
            raise RuntimeError(f"No attention captured for {self.label}.")
        cls_attention = attn_buffers[-1][0, :, 0].mean(dim=0)
        return attention_vector_to_map(cls_attention)


class TransformersAttentionModel:
    """Use Hugging Face vision models that expose attentions directly."""

    def __init__(self, label: str, model_id: str, device: torch.device) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("Install transformers to use DINO/DINOv2 models.") from exc

        self.label = label
        self.model_id = model_id
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, attn_implementation="eager").to(device).eval()

    @torch.no_grad()
    def attention(self, image: Image.Image) -> AttentionResult:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        outputs = self.model(**inputs, output_attentions=True)
        if outputs.attentions is None:
            raise RuntimeError(f"Model {self.model_id} did not return attentions.")
        cls_attention = outputs.attentions[-1][0, :, 0].mean(dim=0)
        return attention_vector_to_map(cls_attention)


class OpenClipAttentionModel:
    """Compute last-block class-token attention from an OpenCLIP ViT image encoder."""

    def __init__(self, label: str, model_name: str, pretrained: str, device: torch.device) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError("Install open_clip_torch to use OpenCLIP models.") from exc

        self.label = label
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.to(device).eval()
        self.visual = model.visual
        self.transform = preprocess

    @torch.no_grad()
    def attention(self, image: Image.Image) -> AttentionResult:
        captured_inputs: list[torch.Tensor] = []
        last_block = self.visual.transformer.resblocks[-1]
        handle = last_block.register_forward_pre_hook(lambda _module, inp: captured_inputs.append(inp[0].detach()))
        try:
            x = self.transform(image).unsqueeze(0).to(self.device)
            _ = self.model.encode_image(x)
        finally:
            handle.remove()

        if not captured_inputs:
            raise RuntimeError(f"No transformer input captured for {self.label}.")

        seq = self._as_batch_first(captured_inputs[-1])
        seq = last_block.ln_1(seq)
        attn = self._mha_attention_weights(last_block.attn, seq)
        cls_attention = attn[0, :, 0].mean(dim=0)
        return attention_vector_to_map(cls_attention)

    @staticmethod
    def _as_batch_first(seq: torch.Tensor) -> torch.Tensor:
        """Normalize OpenCLIP block inputs to [batch, tokens, channels].

        OpenCLIP versions differ here: some visual transformers use
        [tokens, batch, channels], while newer variants use batch-first tensors.
        The script runs one image at a time, so the singleton batch dimension
        makes this unambiguous.
        """

        if seq.ndim != 3:
            raise RuntimeError(f"Expected a 3D OpenCLIP sequence, got shape {tuple(seq.shape)}.")
        if seq.shape[0] == 1 and seq.shape[1] > 1:
            return seq
        if seq.shape[1] == 1 and seq.shape[0] > 1:
            return seq.permute(1, 0, 2)
        raise RuntimeError(f"Could not infer OpenCLIP batch dimension from shape {tuple(seq.shape)}.")

    @staticmethod
    def _mha_attention_weights(attn_module: torch.nn.MultiheadAttention, seq: torch.Tensor) -> torch.Tensor:
        """Return self-attention weights for an nn.MultiheadAttention module.

        The returned tensor has shape [batch, heads, tokens, tokens].
        """

        batch, tokens, channels = seq.shape
        heads = attn_module.num_heads
        head_dim = channels // heads
        qkv = F.linear(seq, attn_module.in_proj_weight, attn_module.in_proj_bias)
        query, key, _value = qkv.chunk(3, dim=-1)
        query = query.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
        key = key.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
        query = query * (head_dim**-0.5)
        return (query @ key.transpose(-2, -1)).softmax(dim=-1)


def attention_vector_to_map(attention: torch.Tensor) -> AttentionResult:
    """Convert a class-token attention vector into a square patch map."""

    attention = attention.detach().float().cpu()
    for prefix_tokens in range(1, 5):
        patch_count = attention.numel() - prefix_tokens
        if patch_count <= 0:
            continue
        side = int(math.sqrt(patch_count))
        if side * side == patch_count:
            patch_attention = attention[prefix_tokens:]
            map_2d = patch_attention.reshape(side, side).numpy()
            return AttentionResult(normalize_map(map_2d), (side, side), prefix_tokens)
    raise RuntimeError(f"Could not infer square patch grid from {attention.numel()} tokens.")


def normalize_map(values: np.ndarray) -> np.ndarray:
    """Normalize an attention map to [0, 1] for visualization."""

    values = values.astype(np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value <= min_value:
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def attention_heatmap(attention: np.ndarray, size: int = 224, colormap: str = "viridis") -> np.ndarray:
    """Render a standalone class-token attention heatmap."""

    attn_tensor = torch.from_numpy(attention)[None, None]
    upsampled = F.interpolate(attn_tensor, size=(size, size), mode="nearest")[0, 0].numpy()
    return plt.get_cmap(colormap)(upsampled)[..., :3]


def overlay_attention(image: Image.Image, attention: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend a small patch attention map over the input RGB image for debugging."""

    image_array = np.asarray(image.resize((224, 224))).astype(np.float32) / 255.0
    attn_tensor = torch.from_numpy(attention)[None, None]
    upsampled = F.interpolate(attn_tensor, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()
    heat = plt.get_cmap("magma")(upsampled)[..., :3]
    return np.clip((1.0 - alpha) * image_array + alpha * heat, 0.0, 1.0)


def load_local_images(image_dir: Path, max_images: int, seed: int) -> list[SelectedImage]:
    """Load deterministic RGB images from a local directory."""

    paths = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(paths) < max_images:
        raise RuntimeError(f"Need at least {max_images} images in {image_dir}, found {len(paths)}.")
    rng = random.Random(seed)
    selected_paths = paths[:]
    rng.shuffle(selected_paths)
    images: list[SelectedImage] = []
    for path in selected_paths[:max_images]:
        images.append(SelectedImage(Image.open(path).convert("RGB"), str(path), path.stem))
    return images


def load_caltech101_images(data_dir: Path, max_images: int, seed: int, download: bool) -> list[SelectedImage]:
    """Load deterministic Caltech101 examples via torchvision."""

    try:
        from torchvision.datasets import Caltech101
    except ImportError as exc:
        raise RuntimeError("Install torchvision to download/load Caltech101.") from exc

    dataset = Caltech101(root=str(data_dir), target_type="category", download=download)
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    images: list[SelectedImage] = []
    for index in indices:
        image, target = dataset[index]
        label = dataset.categories[target]
        if label == "BACKGROUND_Google":
            continue
        images.append(SelectedImage(image.convert("RGB"), f"Caltech101[{index}]", label))
        if len(images) == max_images:
            break

    if len(images) < max_images:
        raise RuntimeError(f"Could only load {len(images)} usable Caltech101 images.")
    return images


def load_cifar10_images(data_dir: Path, max_images: int, seed: int, download: bool) -> list[SelectedImage]:
    """Load deterministic CIFAR10 examples as a robust lightweight fallback."""

    try:
        from torchvision.datasets import CIFAR10
    except ImportError as exc:
        raise RuntimeError("Install torchvision to download/load CIFAR10.") from exc

    dataset = CIFAR10(root=str(data_dir), train=False, download=download)
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    images: list[SelectedImage] = []
    for index in indices[:max_images]:
        image, target = dataset[index]
        label = dataset.classes[target]
        images.append(SelectedImage(image.convert("RGB"), f"CIFAR10-test[{index}]", label))

    if len(images) < max_images:
        raise RuntimeError(f"Could only load {len(images)} usable CIFAR10 images.")
    return images


def load_default_images(data_dir: Path, max_images: int, seed: int, download: bool) -> tuple[list[SelectedImage], str]:
    """Try the preferred dataset first, then fall back to a stable small dataset."""

    try:
        images = load_caltech101_images(data_dir, max_images, seed, download)
        return images, "Caltech101 via torchvision"
    except Exception as caltech_error:
        print(f"Warning: Caltech101 unavailable ({caltech_error}). Falling back to CIFAR10.")

    try:
        images = load_cifar10_images(data_dir, max_images, seed, download)
        return images, "CIFAR10 test split via torchvision fallback"
    except Exception as cifar_error:
        raise RuntimeError(
            "Could not load Caltech101 or CIFAR10. Provide a directory with at least "
            f"{max_images} images using --image-dir."
        ) from cifar_error


def build_models(mode: str, device: torch.device) -> list[object]:
    """Create the ordered model list used as Figure 2 columns."""

    models: list[object] = [
        TimmAttentionModel(
            "DeiT-III-B",
            (
                "deit3_base_patch16_224.fb_in22k_ft_in1k",
                "deit3_base_patch16_224.fb_in1k",
                "deit3_base_patch16_224",
            ),
            device,
        ),
        OpenClipAttentionModel("OpenCLIP-B", "ViT-B-16", "laion2b_s34b_b88k", device),
        TransformersAttentionModel("DINO-B", "facebook/dino-vitb16", device),
        TransformersAttentionModel(
            "DINOv2-B" if mode == "cpu" else "DINOv2-g",
            "facebook/dinov2-base" if mode == "cpu" else "facebook/dinov2-giant",
            device,
        ),
    ]

    if mode == "exact":
        models.insert(
            1,
            TimmAttentionModel(
                "DeiT-III-L",
                (
                    "deit3_large_patch16_224.fb_in22k_ft_in1k",
                    "deit3_large_patch16_224.fb_in1k",
                    "deit3_large_patch16_224",
                ),
                device,
            ),
        )
        models.insert(3, OpenClipAttentionModel("OpenCLIP-L", "ViT-L-14", "laion2b_s32b_b82k", device))

    return models


def make_grid(
    images: list[SelectedImage],
    models: list[object],
    output_path: Path,
    overlay: bool = False,
    colormap: str = "viridis",
) -> dict:
    """Run all models and save the qualitative attention-map grid."""

    columns = ["Input"] + [model.label for model in models]
    figure, axes = plt.subplots(
        nrows=len(images),
        ncols=len(columns),
        figsize=(2.4 * len(columns), 2.4 * len(images)),
        squeeze=False,
    )

    metadata: dict = {"images": [], "models": []}
    for model in models:
        metadata["models"].append(
            {
                "label": model.label,
                "model_name": getattr(model, "model_name", getattr(model, "model_id", "")),
                "pretrained": getattr(model, "pretrained", ""),
            }
        )

    for row, selected in enumerate(images):
        metadata["images"].append({"row": row, "source": selected.source, "label": selected.label})
        axes[row, 0].imshow(selected.image)
        axes[row, 0].set_ylabel(selected.label, fontsize=9)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        for col, model in enumerate(models, start=1):
            result = model.attention(selected.image)
            if overlay:
                axes[row, col].imshow(overlay_attention(selected.image, result.map_2d))
            else:
                axes[row, col].imshow(attention_heatmap(result.map_2d, colormap=colormap))
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            metadata.setdefault("attention_maps", []).append(
                {
                    "image_row": row,
                    "model": model.label,
                    "grid_size": result.grid_size,
                    "prefix_tokens": result.prefix_tokens,
                }
            )

    for col, title in enumerate(columns):
        axes[0, col].set_title(title, fontsize=10)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "exact"), default="cpu", help="Model set to run.")
    parser.add_argument("--device", default="auto", help="Use 'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--max-images", type=int, default=4, help="Number of images in the grid.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic image selection.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Dataset/cache directory.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Optional local image directory.")
    parser.add_argument("--no-download", action="store_true", help="Do not download Caltech101.")
    parser.add_argument("--overlay", action="store_true", help="Blend attention maps over input images for debugging.")
    parser.add_argument("--colormap", default="viridis", help="Matplotlib colormap for standalone attention maps.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "figure2_attention_maps.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    if args.image_dir is not None:
        images = load_local_images(args.image_dir, args.max_images, args.seed)
        image_source = f"local directory: {args.image_dir}"
    else:
        images, image_source = load_default_images(args.data_dir, args.max_images, args.seed, download=not args.no_download)

    models = build_models(args.mode, device)
    metadata = make_grid(images, models, args.output, overlay=args.overlay, colormap=args.colormap)
    metadata.update(
        {
            "paper_result": "Figure 2",
            "mode": args.mode,
            "device": str(device),
            "seed": args.seed,
            "image_source": image_source,
            "attention_display": "overlay" if args.overlay else "standalone heatmap",
            "attention_colormap": args.colormap,
            "output": str(args.output),
        }
    )
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved Figure 2 reproduction to {args.output}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
