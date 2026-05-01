"""Reproduce Figure 3 from "Vision Transformers Need Registers".

Figure 3 compares local feature-token L2 norms for DINO and DINOv2 and plots a
distribution of patch-token norms. Exact mode uses DINOv2 giant, matching the
paper more closely. CPU mode uses DINOv2 large as a slower but more faithful
proxy because the paper reports high-norm outliers mainly for larger models.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from make_figure2 import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, load_default_images, load_local_images, resolve_device


PAPER_NORM_CUTOFF = 150.0
CPU_FALLBACK_PERCENTILE = 99.0
FIGURE2_IMAGE_COUNT = 4


@dataclass(frozen=True)
class NormMap:
    """Patch-token norm values for one image."""

    map_2d: np.ndarray
    values: np.ndarray
    grid_size: tuple[int, int]
    prefix_tokens: int


class PatchNormModel:
    """Extract local patch-token feature norms from a Hugging Face ViT model."""

    def __init__(self, label: str, model_id: str, device: torch.device, local_files_only: bool = False) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("Install transformers to use DINO and DINOv2 models.") from exc

        self.label = label
        self.model_id = model_id
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_id, local_files_only=local_files_only).to(device).eval()

    @torch.no_grad()
    def norm_map(self, image: Image.Image) -> NormMap:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        output = self.model(**inputs, output_hidden_states=True)
        if output.hidden_states is None:
            raise RuntimeError(f"Model {self.model_id} did not return hidden states.")
        # Figure 3 studies token norm outliers. The Hugging Face last_hidden_state
        # is post-final-LayerNorm for ViT models, which suppresses those norm
        # differences. Use the last encoder hidden state before the final norm.
        tokens = output.hidden_states[-1][0].detach().float().cpu()
        patch_tokens, prefix_tokens, grid_size = split_patch_tokens(tokens)
        values = patch_tokens.norm(dim=-1).numpy()
        return NormMap(values.reshape(grid_size), values, grid_size, prefix_tokens)


def split_patch_tokens(tokens: torch.Tensor) -> tuple[torch.Tensor, int, tuple[int, int]]:
    """Split prefix tokens from a square patch-token grid."""

    for prefix_tokens in range(1, 5):
        patch_count = tokens.shape[0] - prefix_tokens
        if patch_count <= 0:
            continue
        side = int(math.sqrt(patch_count))
        if side * side == patch_count:
            return tokens[prefix_tokens:], prefix_tokens, (side, side)
    raise RuntimeError(f"Could not infer square patch grid from {tokens.shape[0]} tokens.")


def build_models(mode: str, device: torch.device, local_files_only: bool) -> tuple[PatchNormModel, PatchNormModel]:
    """Build the DINO and DINOv2 models used for Figure 3."""

    dino = PatchNormModel("DINO ViT-B/16", "facebook/dino-vitb16", device, local_files_only=local_files_only)
    if mode == "exact":
        dinov2 = PatchNormModel(
            "DINOv2 ViT-g/14",
            "facebook/dinov2-giant",
            device,
            local_files_only=local_files_only,
        )
    else:
        dinov2 = PatchNormModel(
            "DINOv2 ViT-L/14",
            "facebook/dinov2-large",
            device,
            local_files_only=local_files_only,
        )
    return dino, dinov2


def load_images(args: argparse.Namespace) -> tuple[list, str]:
    """Load enough images for one reference image plus histogram sampling."""

    needed = max(args.max_hist_images + FIGURE2_IMAGE_COUNT + 1, args.reference_offset + 1)
    if args.image_dir is not None:
        images = load_local_images(args.image_dir, needed, args.seed)
        return images, f"local directory: {args.image_dir}"
    return load_default_images(args.data_dir, needed, args.seed, download=not args.no_download)


def collect_norms(model: PatchNormModel, images: list) -> np.ndarray:
    """Collect flattened patch-token norms for a list of images."""

    norms = [model.norm_map(selected.image).values for selected in images]
    return np.concatenate(norms, axis=0)


def cutoff_stats(norms: np.ndarray, mode: str) -> dict:
    """Compute paper-cutoff and CPU fallback statistics."""

    fraction_above_paper = float((norms > PAPER_NORM_CUTOFF).mean())
    stats = {
        "paper_cutoff": PAPER_NORM_CUTOFF,
        "fraction_above_paper_cutoff": fraction_above_paper,
        "max_norm": float(norms.max()),
        "mean_norm": float(norms.mean()),
        "median_norm": float(np.median(norms)),
    }
    if mode == "cpu":
        fallback_cutoff = float(np.percentile(norms, CPU_FALLBACK_PERCENTILE))
        stats.update(
            {
                "cpu_fallback_percentile": CPU_FALLBACK_PERCENTILE,
                "cpu_fallback_cutoff": fallback_cutoff,
                "fraction_above_cpu_fallback_cutoff": float((norms > fallback_cutoff).mean()),
            }
        )
    return stats


def plot_figure3(
    reference_image,
    dino_map: NormMap,
    dinov2_map: NormMap,
    dino_hist_norms: np.ndarray,
    dinov2_hist_norms: np.ndarray,
    stats: dict,
    output_path: Path,
) -> None:
    """Save Figure 3-style norm-map and histogram visualization."""

    figure = plt.figure(figsize=(11.5, 6.2), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        6,
        width_ratios=[1.0, 1.0, 0.06, 1.0, 1.0, 0.06],
        height_ratios=[1.0, 0.9],
    )
    input_axis = figure.add_subplot(grid[0, 0:2])
    dino_map_axis = figure.add_subplot(grid[0, 2:4])
    dinov2_map_axis = figure.add_subplot(grid[0, 4:6])
    colorbar_axis = dino_map_axis.inset_axes((1.04, 0.0, 0.045, 1.0))
    dino_hist_axis = figure.add_subplot(grid[1, 0:3])
    dinov2_hist_axis = figure.add_subplot(grid[1, 3:6])

    input_axis.imshow(reference_image.image)
    input_axis.set_title("Input")
    input_axis.set_ylabel(reference_image.label, fontsize=9)

    shared_vmax = max(float(dino_map.map_2d.max()), float(dinov2_map.map_2d.max()))
    dino_map_axis.imshow(dino_map.map_2d, cmap="viridis", vmin=0.0, vmax=shared_vmax)
    dino_map_axis.set_title("DINO norms")
    dinov2_plot = dinov2_map_axis.imshow(dinov2_map.map_2d, cmap="viridis", vmin=0.0, vmax=shared_vmax)
    dinov2_map_axis.set_title("DINOv2 norms")
    figure.colorbar(dinov2_plot, cax=colorbar_axis, label="L2 norm")

    hist_xmax = 600
    bins = np.linspace(0.0, hist_xmax, 80)
    dino_hist_axis.hist(dino_hist_norms, bins=bins, color="#3b528b", alpha=0.9)
    dino_hist_axis.axvline(PAPER_NORM_CUTOFF, color="#d62728", linestyle="--", linewidth=1.5)
    dino_hist_axis.set_title("DINO norm distribution")
    dino_hist_axis.set_xlabel("patch-token L2 norm")
    dino_hist_axis.set_ylabel("count")
    dino_hist_axis.set_yscale("log")
    dino_hist_axis.set_xlim(0.0, hist_xmax)

    dinov2_hist_axis.hist(dinov2_hist_norms, bins=bins, color="#3b528b", alpha=0.9)
    dinov2_hist_axis.axvline(PAPER_NORM_CUTOFF, color="#d62728", linestyle="--", linewidth=1.5, label="paper cutoff 150")
    if "cpu_fallback_cutoff" in stats:
        dinov2_hist_axis.axvline(
            stats["cpu_fallback_cutoff"],
            color="#f2c744",
            linestyle=":",
            linewidth=1.8,
            label=f"{CPU_FALLBACK_PERCENTILE:.0f}th pct.",
        )
    dinov2_hist_axis.set_title("DINOv2 norm distribution")
    dinov2_hist_axis.set_xlabel("patch-token L2 norm")
    dinov2_hist_axis.set_ylabel("count")
    dinov2_hist_axis.set_yscale("log")
    dinov2_hist_axis.set_xlim(0.0, hist_xmax)
    dinov2_hist_axis.legend(fontsize=8)

    for axis in (input_axis, dino_map_axis, dinov2_map_axis):
        axis.set_xticks([])
        axis.set_yticks([])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "exact"), default="cpu", help="Use lightweight or paper-scale DINOv2.")
    parser.add_argument("--device", default="auto", help="Use 'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic image selection.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Dataset/cache directory.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Optional local image directory.")
    parser.add_argument("--no-download", action="store_true", help="Use only cached datasets and model files.")
    parser.add_argument("--max-hist-images", type=int, default=16, help="Number of images for the norm histogram.")
    parser.add_argument(
        "--hist-source",
        choices=("sample", "reference"),
        default="sample",
        help="Use a sampled image set for histograms, or only the single reference image.",
    )
    parser.add_argument(
        "--reference-offset",
        type=int,
        default=FIGURE2_IMAGE_COUNT,
        help="Pick a reference image after the first Figure 2 images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "figure3_norms.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    images, image_source = load_images(args)
    breakpoint()
    reference = images[args.reference_offset]
    if args.hist_source == "reference":
        hist_images = [reference]
    else:
        hist_images = images[FIGURE2_IMAGE_COUNT : FIGURE2_IMAGE_COUNT + args.max_hist_images]

    dino, dinov2 = build_models(args.mode, device, local_files_only=args.no_download)
    dino_map = dino.norm_map(reference.image)
    dinov2_map = dinov2.norm_map(reference.image)
    dino_hist_norms = collect_norms(dino, hist_images)
    dinov2_hist_norms = collect_norms(dinov2, hist_images)
    stats = cutoff_stats(dinov2_hist_norms, args.mode)

    plot_figure3(reference, dino_map, dinov2_map, dino_hist_norms, dinov2_hist_norms, stats, args.output)

    metadata = {
        "paper_result": "Figure 3",
        "mode": args.mode,
        "device": str(device),
        "seed": args.seed,
        "image_source": image_source,
        "reference_image": {
            "source": reference.source,
            "label": reference.label,
            "offset": args.reference_offset,
        },
        "histogram_source": args.hist_source,
        "histogram_image_count": len(hist_images),
        "models": [
            {"label": dino.label, "model_id": dino.model_id, "grid_size": dino_map.grid_size},
            {"label": dinov2.label, "model_id": dinov2.model_id, "grid_size": dinov2_map.grid_size},
        ],
        "dino_histogram_stats": cutoff_stats(dino_hist_norms, args.mode),
        "dinov2_histogram_stats": stats,
        "output": str(args.output),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved Figure 3 reproduction to {args.output}")
    print(f"Saved metadata to {metadata_path}")
    print(
        "DINOv2 tokens above paper cutoff 150: "
        f"{100.0 * stats['fraction_above_paper_cutoff']:.2f}%"
    )


if __name__ == "__main__":
    main()
