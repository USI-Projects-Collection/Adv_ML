"""Reproduce Figure 5 from "Vision Transformers Need Registers".

This script implements Figure 5a and Figure 5b. Figure 5a compares cosine
similarity between input patch embeddings and their 4-neighborhood. Figure 5b
trains lightweight linear probes for local position prediction and pixel
reconstruction, then evaluates normal and outlier patch tokens separately.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from make_figure2 import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, load_default_images, load_local_images, resolve_device
from make_figure3 import FIGURE2_IMAGE_COUNT, PAPER_NORM_CUTOFF, split_patch_tokens


DEFAULT_CUTOFF_PERCENTILE = 99.0
EARLY_STOPPING_PATIENCE = 3


@dataclass(frozen=True)
class PatchFeatures:
    """Patch-level tensors used by Figure 5a and Figure 5b."""

    input_embeddings: torch.Tensor
    output_embeddings: torch.Tensor
    output_norms: np.ndarray
    pixel_targets: torch.Tensor
    position_targets: torch.Tensor
    grid_size: tuple[int, int]
    prefix_tokens: int


class DINOv2FeatureModel:
    """Extract DINOv2 patch embeddings needed for Figure 5a."""

    def __init__(self, mode: str, device: torch.device, local_files_only: bool = False) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("Install transformers to use DINOv2 models.") from exc

        self.label = "DINOv2 ViT-g/14" if mode == "exact" else "DINOv2 ViT-L/14"
        self.model_id = "facebook/dinov2-giant" if mode == "exact" else "facebook/dinov2-large"
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.model_id, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(self.model_id, local_files_only=local_files_only).to(device).eval()

    @torch.no_grad()
    def extract(self, image: Image.Image) -> PatchFeatures:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        output = self.model(**inputs, output_hidden_states=True)
        if output.hidden_states is None:
            raise RuntimeError(f"Model {self.model_id} did not return hidden states.")

        # hidden_states[0] is the embedding output before transformer blocks.
        input_tokens = output.hidden_states[0][0].detach().float().cpu()
        # hidden_states[-1] is the final encoder output before the final ViT LayerNorm.
        output_tokens = output.hidden_states[-1][0].detach().float().cpu()

        input_patches, input_prefix_tokens, input_grid = split_patch_tokens(input_tokens)
        output_patches, output_prefix_tokens, output_grid = split_patch_tokens(output_tokens)
        if input_grid != output_grid:
            raise RuntimeError(f"Input/output patch grids differ: {input_grid} vs {output_grid}.")
        if input_prefix_tokens != output_prefix_tokens:
            raise RuntimeError(f"Input/output prefix counts differ: {input_prefix_tokens} vs {output_prefix_tokens}.")

        output_norms = output_patches.norm(dim=-1).numpy()
        pixel_targets = patchify_pixels(inputs["pixel_values"][0].detach().float().cpu(), input_grid)
        position_targets = torch.arange(output_patches.shape[0], dtype=torch.long)
        return PatchFeatures(
            input_embeddings=input_patches,
            output_embeddings=output_patches,
            output_norms=output_norms,
            pixel_targets=pixel_targets,
            position_targets=position_targets,
            grid_size=input_grid,
            prefix_tokens=input_prefix_tokens,
        )


def patchify_pixels(pixel_values: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
    """Convert preprocessed image tensor [C, H, W] into flattened patch pixels."""

    channels, height, width = pixel_values.shape
    grid_height, grid_width = grid_size
    if height % grid_height or width % grid_width:
        raise RuntimeError(f"Image size {(height, width)} is not divisible by patch grid {grid_size}.")
    patch_height = height // grid_height
    patch_width = width // grid_width
    patches = pixel_values.reshape(channels, grid_height, patch_height, grid_width, patch_width)
    patches = patches.permute(1, 3, 0, 2, 4).reshape(grid_height * grid_width, -1)
    return patches


def load_images(args: argparse.Namespace) -> tuple[list, str]:
    """Load images for the Figure 5a distribution."""

    needed = args.image_offset + args.max_images
    if args.image_dir is not None:
        images = load_local_images(args.image_dir, needed, args.seed)
        return images[args.image_offset : needed], f"local directory: {args.image_dir}"
    images, source = load_default_images(args.data_dir, needed, args.seed, download=not args.no_download)
    return images[args.image_offset : needed], source


def percentile_arg(value: str) -> float:
    """Parse a percentile cutoff and reject values outside (0, 100)."""

    percentile = float(value)
    if not 0.0 < percentile < 100.0:
        raise argparse.ArgumentTypeError("cutoff percentile must be between 0 and 100.")
    return percentile


def choose_cutoff(
    all_norms: np.ndarray,
    mode: str,
    cutoff_mode: str,
    cutoff_percentile: float,
) -> tuple[float, str]:
    """Choose the norm cutoff used to define outlier tokens."""

    if cutoff_mode == "paper":
        return PAPER_NORM_CUTOFF, "paper"
    if cutoff_mode == "percentile":
        return float(np.percentile(all_norms, cutoff_percentile)), f"p{cutoff_percentile:g}"

    paper_count = int((all_norms > PAPER_NORM_CUTOFF).sum())
    if mode == "exact" or paper_count > 0:
        return PAPER_NORM_CUTOFF, "paper"
    return float(np.percentile(all_norms, cutoff_percentile)), f"p{cutoff_percentile:g}"


def neighbor_cosines(input_embeddings: torch.Tensor, grid_size: tuple[int, int], outlier_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute 4-neighbor cosine similarities split by center-patch group."""

    height, width = grid_size
    embeddings = F.normalize(input_embeddings, dim=-1).reshape(height, width, -1)
    mask = outlier_mask.reshape(height, width)
    normal_values: list[float] = []
    outlier_values: list[float] = []
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    for row in range(height):
        for col in range(width):
            target = outlier_values if mask[row, col] else normal_values
            center = embeddings[row, col]
            for drow, dcol in directions:
                neighbor_row = row + drow
                neighbor_col = col + dcol
                if 0 <= neighbor_row < height and 0 <= neighbor_col < width:
                    similarity = float(torch.dot(center, embeddings[neighbor_row, neighbor_col]))
                    target.append(similarity)

    return np.asarray(normal_values, dtype=np.float32), np.asarray(outlier_values, dtype=np.float32)


def collect_figure5a(
    model: DINOv2FeatureModel,
    images: list,
    mode: str,
    cutoff_mode: str,
    cutoff_percentile: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Collect Figure 5a normal/outlier cosine-similarity samples."""

    features = [model.extract(selected.image) for selected in images]
    all_norms = np.concatenate([feature.output_norms for feature in features], axis=0)
    cutoff, cutoff_source = choose_cutoff(all_norms, mode, cutoff_mode, cutoff_percentile)

    normal_parts: list[np.ndarray] = []
    outlier_parts: list[np.ndarray] = []
    outlier_token_count = 0
    total_token_count = 0
    for feature in features:
        outlier_mask = feature.output_norms > cutoff
        normal_values, outlier_values = neighbor_cosines(feature.input_embeddings, feature.grid_size, outlier_mask)
        normal_parts.append(normal_values)
        outlier_parts.append(outlier_values)
        outlier_token_count += int(outlier_mask.sum())
        total_token_count += int(outlier_mask.size)

    normal_cosines = np.concatenate(normal_parts, axis=0) if normal_parts else np.empty(0, dtype=np.float32)
    outlier_cosines = np.concatenate(outlier_parts, axis=0) if outlier_parts else np.empty(0, dtype=np.float32)
    if outlier_cosines.size == 0:
        raise RuntimeError(
            "No outlier patch neighbor comparisons were collected. "
            "Try --cutoff-mode percentile or increase --max-images."
        )

    metadata = {
        "cutoff": cutoff,
        "cutoff_source": cutoff_source,
        "paper_cutoff": PAPER_NORM_CUTOFF,
        "cutoff_percentile": cutoff_percentile,
        "patch_tokens": total_token_count,
        "outlier_patch_tokens": outlier_token_count,
        "outlier_patch_fraction": outlier_token_count / total_token_count,
        "normal_neighbor_pairs": int(normal_cosines.size),
        "outlier_neighbor_pairs": int(outlier_cosines.size),
        "normal_cosine_mean": float(normal_cosines.mean()),
        "outlier_cosine_mean": float(outlier_cosines.mean()),
        "normal_cosine_median": float(np.median(normal_cosines)),
        "outlier_cosine_median": float(np.median(outlier_cosines)),
    }
    return normal_cosines, outlier_cosines, metadata


def plot_figure5a(normal_cosines: np.ndarray, outlier_cosines: np.ndarray, output_path: Path) -> None:
    """Save Figure 5a-style cosine similarity distribution."""

    figure, axis = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    bins = np.linspace(-1.0, 1.0, 80)
    axis.hist(normal_cosines, bins=bins, density=True, histtype="step", linewidth=2.0, label="normal")
    axis.hist(outlier_cosines, bins=bins, density=True, histtype="step", linewidth=2.0, label="outlier")
    axis.set_title("Cosine similarity to 4-neighbor input patches")
    axis.set_xlabel("cosine similarity")
    axis.set_ylabel("density")
    axis.set_xlim(-1.0, 1.0)
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def collect_probe_dataset(
    model: DINOv2FeatureModel,
    images: list,
    mode: str,
    cutoff_mode: str,
    cutoff_percentile: float,
) -> tuple[dict, dict]:
    """Collect frozen patch embeddings and labels for Figure 5b probes."""

    features = [model.extract(selected.image) for selected in images]
    all_norms = np.concatenate([feature.output_norms for feature in features], axis=0)
    cutoff, cutoff_source = choose_cutoff(all_norms, mode, cutoff_mode, cutoff_percentile)

    embeddings: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    pixels: list[torch.Tensor] = []
    outlier_masks: list[torch.Tensor] = []
    fallback_outlier_images = 0
    fallback_normal_images = 0
    for feature in features:
        outlier_mask = feature.output_norms > cutoff
        if not np.any(outlier_mask):
            outlier_mask[int(feature.output_norms.argmax())] = True
            fallback_outlier_images += 1
        if np.all(outlier_mask):
            outlier_mask[int(feature.output_norms.argmin())] = False
            fallback_normal_images += 1

        embeddings.append(feature.output_embeddings)
        positions.append(feature.position_targets)
        pixels.append(feature.pixel_targets)
        outlier_masks.append(torch.from_numpy(outlier_mask))

    x = torch.cat(embeddings, dim=0).float()
    position_y = torch.cat(positions, dim=0).long()
    pixel_y = torch.cat(pixels, dim=0).float()
    outlier_mask = torch.cat(outlier_masks, dim=0).bool()

    metadata = {
        "cutoff": cutoff,
        "cutoff_source": cutoff_source,
        "paper_cutoff": PAPER_NORM_CUTOFF,
        "cutoff_percentile": cutoff_percentile,
        "image_count": len(images),
        "patch_tokens": int(x.shape[0]),
        "outlier_patch_tokens": int(outlier_mask.sum()),
        "outlier_patch_fraction": float(outlier_mask.float().mean()),
        "feature_dim": int(x.shape[1]),
        "pixel_target_dim": int(pixel_y.shape[1]),
        "num_positions": int(position_y.max().item() + 1),
        "fallback_outlier_images": fallback_outlier_images,
        "fallback_normal_images": fallback_normal_images,
    }
    dataset = {
        "x": x,
        "position_y": position_y,
        "pixel_y": pixel_y,
        "outlier_mask": outlier_mask,
    }
    return dataset, metadata


def make_stratified_split(outlier_mask: torch.Tensor, train_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create train/test indices while preserving normal and outlier groups."""

    generator = torch.Generator().manual_seed(seed)
    train_parts: list[torch.Tensor] = []
    test_parts: list[torch.Tensor] = []
    for group_value in (False, True):
        indices = torch.where(outlier_mask == group_value)[0]
        if indices.numel() < 2:
            raise RuntimeError(f"Need at least 2 samples for group outlier={group_value}, found {indices.numel()}.")
        shuffled = indices[torch.randperm(indices.numel(), generator=generator)]
        train_count = int(round(train_fraction * shuffled.numel()))
        train_count = min(max(train_count, 1), shuffled.numel() - 1)
        train_parts.append(shuffled[:train_count])
        test_parts.append(shuffled[train_count:])

    train_idx = torch.cat(train_parts)
    test_idx = torch.cat(test_parts)
    train_idx = train_idx[torch.randperm(train_idx.numel(), generator=generator)]
    test_idx = test_idx[torch.randperm(test_idx.numel(), generator=generator)]
    return train_idx, test_idx


def standardize_features(train_x: torch.Tensor, all_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardize probe features using train-set statistics."""

    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (all_x - mean) / std


def train_position_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    num_positions: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> torch.nn.Linear:
    """Train a linear classifier to predict patch position."""

    generator = torch.Generator().manual_seed(seed)
    model = torch.nn.Linear(train_x.shape[1], num_positions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    best_test_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        order = torch.randperm(train_x.shape[0], generator=generator)
        epoch_loss = 0.0
        batch_count = 0
        for start in range(0, order.numel(), batch_size):
            batch = order[start : start + batch_size].to(device)
            loss = F.cross_entropy(model(train_x[batch]), train_y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batch_count += 1
        mean_loss = epoch_loss / max(batch_count, 1)
        with torch.no_grad():
            logits = model(test_x)
            test_loss = F.cross_entropy(logits, test_y).item()

        improved = test_loss < best_test_loss
        if improved:
            best_test_loss = test_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"[position probe] epoch {epoch + 1}/{epochs} "
            f"train_loss={mean_loss:.4f} test_loss={test_loss:.4f} "
            f"best_test_loss={best_test_loss:.4f} wait={epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}"
        )
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"[position probe] early stop at epoch {epoch + 1}/{epochs}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_reconstruction_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> torch.nn.Linear:
    """Train a linear regressor to reconstruct preprocessed patch pixels."""

    generator = torch.Generator().manual_seed(seed)
    model = torch.nn.Linear(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    best_test_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        order = torch.randperm(train_x.shape[0], generator=generator)
        epoch_loss = 0.0
        batch_count = 0
        for start in range(0, order.numel(), batch_size):
            batch = order[start : start + batch_size].to(device)
            loss = F.mse_loss(model(train_x[batch]), train_y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batch_count += 1
        mean_loss = epoch_loss / max(batch_count, 1)
        with torch.no_grad():
            recon = model(test_x)
            test_loss = F.mse_loss(recon, test_y).item()

        improved = test_loss < best_test_loss
        if improved:
            best_test_loss = test_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"[reconstruction probe] epoch {epoch + 1}/{epochs} "
            f"train_loss={mean_loss:.4f} test_loss={test_loss:.4f} "
            f"best_test_loss={best_test_loss:.4f} wait={epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}"
        )
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"[reconstruction probe] early stop at epoch {epoch + 1}/{epochs}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def position_average_distance(predicted: torch.Tensor, target: torch.Tensor, grid_width: int) -> float:
    """Average Euclidean distance between predicted and true patch positions."""

    pred_row = torch.div(predicted, grid_width, rounding_mode="floor").float()
    pred_col = (predicted % grid_width).float()
    target_row = torch.div(target, grid_width, rounding_mode="floor").float()
    target_col = (target % grid_width).float()
    distance = torch.sqrt((pred_row - target_row) ** 2 + (pred_col - target_col) ** 2)
    return float(distance.mean())


def evaluate_figure5b(
    position_probe: torch.nn.Linear,
    reconstruction_probe: torch.nn.Linear,
    all_x: torch.Tensor,
    dataset: dict,
    test_idx: torch.Tensor,
    grid_size: tuple[int, int],
    device: torch.device,
) -> dict:
    """Evaluate Figure 5b metrics on normal and outlier test patches."""

    position_probe.eval()
    reconstruction_probe.eval()
    x_test = all_x[test_idx].to(device)
    position_y = dataset["position_y"][test_idx]
    pixel_y = dataset["pixel_y"][test_idx]
    outlier_mask = dataset["outlier_mask"][test_idx]

    with torch.no_grad():
        position_pred = position_probe(x_test).argmax(dim=1).cpu()
        pixel_pred = reconstruction_probe(x_test).cpu()

    metrics: dict[str, dict] = {}
    for label, group_value in (("normal", False), ("outlier", True)):
        group = outlier_mask == group_value
        if int(group.sum()) == 0:
            raise RuntimeError(f"No {label} samples in the Figure 5b test split.")
        group_pred = position_pred[group]
        group_position_y = position_y[group]
        group_pixel_pred = pixel_pred[group]
        group_pixel_y = pixel_y[group]
        l2_errors = torch.linalg.vector_norm(group_pixel_pred - group_pixel_y, dim=1)
        metrics[label] = {
            "position_top1_acc": float((group_pred == group_position_y).float().mean()),
            "position_avg_distance": position_average_distance(group_pred, group_position_y, grid_size[1]),
            "reconstruction_l2_error": float(l2_errors.mean()),
            "test_samples": int(group.sum()),
        }
    return metrics


def plot_figure5b(metrics: dict, output_path: Path) -> None:
    """Save a compact Figure 5b-style metrics table."""

    rows = ["normal", "outlier"]
    columns = ["position top-1 acc", "avg. distance", "reconstruction L2 error"]
    cell_text = []
    for row in rows:
        cell_text.append(
            [
                f"{100.0 * metrics[row]['position_top1_acc']:.1f}",
                f"{metrics[row]['position_avg_distance']:.2f}",
                f"{metrics[row]['reconstruction_l2_error']:.2f}",
            ]
        )

    figure, axis = plt.subplots(figsize=(7.4, 2.2), constrained_layout=True)
    axis.axis("off")
    table = axis.table(
        cellText=cell_text,
        rowLabels=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    axis.set_title("Figure 5b local information probing")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def run_figure5b(args: argparse.Namespace, images: list, model: DINOv2FeatureModel, device: torch.device) -> tuple[dict, dict]:
    """Train and evaluate Figure 5b local-information probes."""

    dataset, metadata = collect_probe_dataset(model, images, args.mode, args.cutoff_mode, args.cutoff_percentile)
    train_idx, test_idx = make_stratified_split(dataset["outlier_mask"], args.train_fraction, args.seed)
    train_x, all_x = standardize_features(dataset["x"][train_idx], dataset["x"])
    train_y_position = dataset["position_y"][train_idx]
    train_y_pixels = dataset["pixel_y"][train_idx]
    test_x = all_x[test_idx]
    test_y_position = dataset["position_y"][test_idx]
    test_y_pixels = dataset["pixel_y"][test_idx]

    position_probe = train_position_probe(
        train_x,
        train_y_position,
        test_x,
        test_y_position,
        metadata["num_positions"],
        device,
        args.probe_epochs,
        args.batch_size,
        args.probe_lr,
        args.seed,
    )
    reconstruction_probe = train_reconstruction_probe(
        train_x,
        train_y_pixels,
        test_x,
        test_y_pixels,
        device,
        args.probe_epochs,
        args.batch_size,
        args.probe_lr,
        args.seed + 1,
    )
    grid_side = int(round(metadata["num_positions"] ** 0.5))
    metrics = evaluate_figure5b(
        position_probe,
        reconstruction_probe,
        all_x,
        dataset,
        test_idx,
        (grid_side, grid_side),
        device,
    )
    metadata.update(
        {
            "train_samples": int(train_idx.numel()),
            "test_samples": int(test_idx.numel()),
            "train_fraction": args.train_fraction,
            "probe_epochs": args.probe_epochs,
            "batch_size": args.batch_size,
            "probe_lr": args.probe_lr,
        }
    )
    return metrics, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("5a", "5b"), default="5a", help="Figure 5 component to reproduce.")
    parser.add_argument("--mode", choices=("cpu", "exact"), default="cpu", help="Use DINOv2-Large or DINOv2-Giant.")
    parser.add_argument("--device", default="auto", help="Use 'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic image selection.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Dataset/cache directory.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Optional local image directory.")
    parser.add_argument("--no-download", action="store_true", help="Use only cached datasets and model files.")
    parser.add_argument("--max-images", type=int, default=16, help="Number of images for Figure 5.")
    parser.add_argument(
        "--image-offset",
        type=int,
        default=FIGURE2_IMAGE_COUNT,
        help="Skip the first images used for Figure 2 before sampling.",
    )
    parser.add_argument(
        "--cutoff-mode",
        choices=("auto", "paper", "percentile"),
        default="auto",
        help="Outlier norm cutoff: paper=150, percentile=<cutoff-percentile>, auto=paper when it yields outliers.",
    )
    parser.add_argument(
        "--cutoff-percentile",
        type=percentile_arg,
        default=DEFAULT_CUTOFF_PERCENTILE,
        help="Percentile used when cutoff mode is percentile, or when auto falls back to percentile.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.75, help="Train split fraction for Figure 5b probes.")
    parser.add_argument("--probe-epochs", type=int, default=80, help="Training epochs for Figure 5b linear probes.")
    parser.add_argument("--batch-size", type=int, default=512, help="Probe training batch size.")
    parser.add_argument("--probe-lr", type=float, default=1e-3, help="Probe optimizer learning rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    images, image_source = load_images(args)
    model = DINOv2FeatureModel(args.mode, device, local_files_only=args.no_download)
    if args.output is None:
        args.output = DEFAULT_RESULTS_DIR / (
            "figure5a_neighbor_cosine.png" if args.part == "5a" else "figure5b_local_probes.png"
        )

    metadata = {
        "paper_result": f"Figure {args.part}",
        "mode": args.mode,
        "device": str(device),
        "seed": args.seed,
        "image_source": image_source,
        "image_count": len(images),
        "model": {"label": model.label, "model_id": model.model_id},
        "output": str(args.output),
    }

    if args.part == "5a":
        normal_cosines, outlier_cosines, stats = collect_figure5a(
            model,
            images,
            args.mode,
            args.cutoff_mode,
            args.cutoff_percentile,
        )
        plot_figure5a(normal_cosines, outlier_cosines, args.output)
        metadata["figure5a_stats"] = stats
        print(f"Saved Figure 5a reproduction to {args.output}")
        print(
            "Outlier patch fraction: "
            f"{100.0 * stats['outlier_patch_fraction']:.2f}% "
            f"using cutoff {stats['cutoff']:.3f} ({stats['cutoff_source']})"
        )
    else:
        metrics, stats = run_figure5b(args, images, model, device)
        plot_figure5b(metrics, args.output)
        metadata["figure5b_stats"] = stats
        metadata["figure5b_metrics"] = metrics
        print(f"Saved Figure 5b reproduction to {args.output}")
        print(
            "Figure 5b metrics: "
            f"normal acc={100.0 * metrics['normal']['position_top1_acc']:.1f}, "
            f"outlier acc={100.0 * metrics['outlier']['position_top1_acc']:.1f}"
        )

    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
