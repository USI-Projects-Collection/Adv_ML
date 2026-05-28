"""Visualize DINOv2-g attention maps for one outlier patch token and one normal patch token.

The script samples Caltech101 in deterministic random order, finds the first image
that contains at least one patch token with norm > 150 in the final pre-LayerNorm
encoder output, then saves:
1. the input image,
2. the attention map induced by one outlier patch token, and
3. the attention map induced by one normal patch token.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from make_figure2 import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, load_caltech101_images, overlay_attention, resolve_device
from make_figure3 import PAPER_NORM_CUTOFF, split_patch_tokens


@dataclass(frozen=True)
class TokenAttentionSample:
    """Per-image DINOv2-g outputs needed for token-attention visualization."""

    image: Image.Image
    source: str
    label: str
    patch_norms: np.ndarray
    attention: torch.Tensor
    prefix_tokens: int
    grid_size: tuple[int, int]


class DINOv2PatchAttentionModel:
    """Extract last-layer self-attention and final patch-token norms from DINOv2-g."""

    def __init__(self, device: torch.device, local_files_only: bool = False) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("Install transformers to use DINOv2 models.") from exc

        self.label = "DINOv2 ViT-g/14"
        self.model_id = "facebook/dinov2-giant"
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.model_id, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(
            self.model_id,
            attn_implementation="eager",
            local_files_only=local_files_only,
        ).to(device).eval()

    @torch.no_grad()
    def extract(self, image: Image.Image, source: str, label: str) -> TokenAttentionSample:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        output = self.model(**inputs, output_hidden_states=True, output_attentions=True)
        if output.hidden_states is None:
            raise RuntimeError(f"Model {self.model_id} did not return hidden states.")
        if output.attentions is None:
            raise RuntimeError(f"Model {self.model_id} did not return attentions.")

        final_tokens = output.hidden_states[-1][0].detach().float().cpu()
        patch_tokens, prefix_tokens, grid_size = split_patch_tokens(final_tokens)
        patch_norms = patch_tokens.norm(dim=-1).numpy()
        attention = output.attentions[-1][0].detach().float().cpu()
        return TokenAttentionSample(
            image=image,
            source=source,
            label=label,
            patch_norms=patch_norms,
            attention=attention,
            prefix_tokens=prefix_tokens,
            grid_size=grid_size,
        )


def pick_image_with_outlier(
    images,
    model: DINOv2PatchAttentionModel,
    seed: int,
) -> tuple[TokenAttentionSample, int, int]:
    """Return the first sampled image with at least one outlier and one normal patch."""

    rng = random.Random(seed)
    for selected in images:
        sample = model.extract(selected.image, selected.source, selected.label)
        outlier_indices = np.flatnonzero(sample.patch_norms > PAPER_NORM_CUTOFF)
        normal_indices = np.flatnonzero(sample.patch_norms <= PAPER_NORM_CUTOFF)
        if outlier_indices.size == 0 or normal_indices.size == 0:
            continue
        outlier_index = int(rng.choice(outlier_indices.tolist()))
        normal_index = int(rng.choice(normal_indices.tolist()))
        return sample, outlier_index, normal_index
    raise RuntimeError(
        "No sampled Caltech101 image contained both an outlier patch (norm > 150) "
        "and a normal patch. Try a different seed."
    )


def token_attention_map(sample: TokenAttentionSample, patch_index: int) -> np.ndarray:
    """Convert one patch token's last-layer attention row into a 2D patch map."""

    token_index = sample.prefix_tokens + patch_index
    attention_vector = sample.attention[:, token_index].mean(dim=0)
    patch_attention = attention_vector[sample.prefix_tokens :]
    return patch_attention.reshape(sample.grid_size).numpy()


def enhance_attention_map(values: np.ndarray, gamma: float = 0.55) -> np.ndarray:
    """Increase visual contrast for low-dynamic-range attention maps."""

    values = values.astype(np.float32)
    low = float(np.percentile(values, 5.0))
    high = float(np.percentile(values, 99.5))
    if high <= low:
        return np.zeros_like(values)
    clipped = np.clip(values, low, high)
    normalized = (clipped - low) / (high - low)
    return np.power(normalized, gamma)


def patch_rectangle(
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
    patch_index: int,
) -> tuple[float, float, float, float]:
    """Return x, y, width, height for one patch on the original image."""

    width, height = image_size
    grid_height, grid_width = grid_size
    row = patch_index // grid_width
    col = patch_index % grid_width
    patch_width = width / grid_width
    patch_height = height / grid_height
    return col * patch_width, row * patch_height, patch_width, patch_height


def save_figure(
    sample: TokenAttentionSample,
    outlier_map: np.ndarray,
    normal_map: np.ndarray,
    outlier_index: int,
    normal_index: int,
    output_path: Path,
) -> None:
    """Save a higher-contrast comparison of outlier and normal token attention maps."""

    figure, axes = plt.subplots(1, 4, figsize=(15.0, 4.2), constrained_layout=True)
    outlier_display = enhance_attention_map(outlier_map)
    normal_display = enhance_attention_map(normal_map)
    difference = outlier_display - normal_display
    image_size = sample.image.size
    outlier_rect = patch_rectangle(image_size, sample.grid_size, outlier_index)
    normal_rect = patch_rectangle(image_size, sample.grid_size, normal_index)

    axes[0].imshow(sample.image)
    axes[0].set_title(f"Input\n{sample.label}")
    for rect, color, text in (
        (outlier_rect, "#d62728", "outlier"),
        (normal_rect, "#1f77b4", "normal"),
    ):
        axes[0].add_patch(
            plt.Rectangle(
                (rect[0], rect[1]),
                rect[2],
                rect[3],
                fill=False,
                edgecolor=color,
                linewidth=2.0,
            )
        )
        axes[0].text(
            rect[0],
            max(rect[1] - 3.0, 2.0),
            text,
            color=color,
            fontsize=8,
            weight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
        )

    axes[1].imshow(overlay_attention(sample.image, outlier_display, alpha=0.75))
    axes[1].set_title(
        "Outlier token attention\n"
        f"patch {outlier_index}, norm={sample.patch_norms[outlier_index]:.1f}"
    )
    axes[2].imshow(overlay_attention(sample.image, normal_display, alpha=0.75))
    axes[2].set_title(
        "Normal token attention\n"
        f"patch {normal_index}, norm={sample.patch_norms[normal_index]:.1f}"
    )

    vmax = float(np.abs(difference).max())
    diff_plot = axes[3].imshow(difference, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[3].set_title("Outlier - normal\nattention difference")
    figure.colorbar(diff_plot, ax=axes[3], fraction=0.046, pad=0.04)

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="Use 'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic Caltech101 sampling.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Dataset/cache directory.")
    parser.add_argument("--no-download", action="store_true", help="Use only cached datasets and model files.")
    parser.add_argument(
        "--max-images",
        type=int,
        default=512,
        help="Maximum number of random Caltech101 images to scan for a valid outlier token.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "patch_token_attention_maps.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    images = load_caltech101_images(args.data_dir, args.max_images, args.seed, download=not args.no_download)
    model = DINOv2PatchAttentionModel(device, local_files_only=args.no_download)
    sample, outlier_index, normal_index = pick_image_with_outlier(images, model, args.seed)
    outlier_map = token_attention_map(sample, outlier_index)
    normal_map = token_attention_map(sample, normal_index)
    save_figure(sample, outlier_map, normal_map, outlier_index, normal_index, args.output)

    metadata = {
        "paper_result": "Patch-token attention maps",
        "device": str(device),
        "seed": args.seed,
        "cutoff": PAPER_NORM_CUTOFF,
        "model": {"label": model.label, "model_id": model.model_id},
        "selected_image": {"source": sample.source, "label": sample.label},
        "grid_size": list(sample.grid_size),
        "prefix_tokens": sample.prefix_tokens,
        "outlier_patch": {
            "index": outlier_index,
            "norm": float(sample.patch_norms[outlier_index]),
        },
        "normal_patch": {
            "index": normal_index,
            "norm": float(sample.patch_norms[normal_index]),
        },
        "max_patch_norm": float(sample.patch_norms.max()),
        "min_patch_norm": float(sample.patch_norms.min()),
        "output": str(args.output),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved patch-token attention maps to {args.output}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
