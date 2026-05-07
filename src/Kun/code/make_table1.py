"""Reproduce Table 1 from "Vision Transformers Need Registers".

The table trains image-classification linear probes from three frozen DINOv2
representations per image: the class token, one normal patch token, and one
high-norm/outlier patch token. CPU mode runs the lightweight subset from the
project plan: CIFAR10 and Caltech101 when available.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from make_figure2 import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, resolve_device
from make_figure3 import CPU_FALLBACK_PERCENTILE, PAPER_NORM_CUTOFF, split_patch_tokens
from make_figure5 import choose_cutoff


@dataclass(frozen=True)
class LabeledImage:
    """Single labeled image example."""

    image: Image.Image
    label: int
    label_name: str
    source: str


class DINOv2TableModel:
    """Extract [CLS], normal patch, and outlier patch features from DINOv2."""

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
    def extract_tokens(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        output = self.model(**inputs, output_hidden_states=True)
        if output.hidden_states is None:
            raise RuntimeError(f"Model {self.model_id} did not return hidden states.")

        # Use the pre-final-LayerNorm encoder output to keep norm outliers visible.
        tokens = output.hidden_states[-1][0].detach().float().cpu()
        cls_token = tokens[0]
        patch_tokens, _prefix_tokens, _grid_size = split_patch_tokens(tokens)
        patch_norms = patch_tokens.norm(dim=-1).numpy()
        return cls_token, patch_tokens, patch_norms


def load_cifar10_split(data_dir: Path, split: str, max_images: int, seed: int, download: bool) -> tuple[list[LabeledImage], dict]:
    """Load a deterministic CIFAR10 train or test subset."""

    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(root=str(data_dir), train=(split == "train"), download=download)
    indices = stratified_indices(dataset.targets, max_images, seed)
    examples = [
        LabeledImage(dataset[index][0].convert("RGB"), int(dataset.targets[index]), dataset.classes[int(dataset.targets[index])], f"CIFAR10-{split}[{index}]")
        for index in indices
    ]
    metadata = {"name": "CIFAR10", "split": split, "classes": dataset.classes, "source": "torchvision.datasets.CIFAR10"}
    return examples, metadata


def load_caltech101_splits(data_dir: Path, max_train: int, max_test: int, seed: int, download: bool) -> tuple[list[LabeledImage], list[LabeledImage], dict]:
    """Load deterministic Caltech101 train/test subsets from one dataset."""

    from torchvision.datasets import Caltech101

    dataset = Caltech101(root=str(data_dir), target_type="category", download=download)
    label_to_indices: dict[int, list[int]] = {}
    for index, target in enumerate(dataset.y):
        label = int(target)
        if dataset.categories[label] == "BACKGROUND_Google":
            continue
        label_to_indices.setdefault(label, []).append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    per_class_train = max(1, max_train // max(len(label_to_indices), 1))
    per_class_test = max(1, max_test // max(len(label_to_indices), 1))
    for indices in label_to_indices.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        train_indices.extend(shuffled[:per_class_train])
        test_indices.extend(shuffled[per_class_train : per_class_train + per_class_test])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    train_indices = train_indices[:max_train]
    test_indices = test_indices[:max_test]

    def make_examples(indices: list[int], split: str) -> list[LabeledImage]:
        examples: list[LabeledImage] = []
        for index in indices:
            image, target = dataset[index]
            label = int(target)
            examples.append(
                LabeledImage(image.convert("RGB"), label, dataset.categories[label], f"Caltech101-{split}[{index}]")
            )
        return examples

    metadata = {"name": "Caltech101", "classes": dataset.categories, "source": "torchvision.datasets.Caltech101"}
    return make_examples(train_indices, "train"), make_examples(test_indices, "test"), metadata


def stratified_indices(labels: list[int], max_images: int, seed: int) -> list[int]:
    """Choose a roughly class-balanced deterministic subset."""

    label_to_indices: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        label_to_indices.setdefault(int(label), []).append(index)

    rng = random.Random(seed)
    per_class = max(1, max_images // max(len(label_to_indices), 1))
    selected: list[int] = []
    for indices in label_to_indices.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        selected.extend(shuffled[:per_class])
    if len(selected) < max_images:
        remaining = [index for index in range(len(labels)) if index not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_images - len(selected)])
    rng.shuffle(selected)
    return selected[:max_images]


def load_dataset_pair(args: argparse.Namespace, name: str) -> tuple[list[LabeledImage], list[LabeledImage], dict]:
    """Load train/test examples for a supported Table 1 dataset."""

    normalized = name.lower()
    if normalized == "cifar10":
        train, train_meta = load_cifar10_split(args.data_dir, "train", args.max_train_images, args.seed, not args.no_download)
        test, test_meta = load_cifar10_split(args.data_dir, "test", args.max_test_images, args.seed + 1, not args.no_download)
        return train, test, {"name": "CIFAR10", "train": train_meta, "test": test_meta}
    if normalized == "caltech101":
        train, test, metadata = load_caltech101_splits(
            args.data_dir,
            args.max_train_images,
            args.max_test_images,
            args.seed,
            not args.no_download,
        )
        return train, test, metadata
    raise ValueError(f"Unsupported dataset '{name}'. Supported: CIFAR10, Caltech101.")


def extract_split_features(
    model: DINOv2TableModel,
    examples: list[LabeledImage],
    cutoff: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict]:
    """Extract one image-level feature per token type for a split."""

    rng = random.Random(seed)
    features = {"cls": [], "normal": [], "outlier": []}
    labels: list[int] = []
    fallback_outlier_count = 0
    fallback_normal_count = 0

    for example in examples:
        cls_token, patch_tokens, patch_norms = model.extract_tokens(example.image)
        outlier_indices = np.flatnonzero(patch_norms > cutoff)
        normal_indices = np.flatnonzero(patch_norms <= cutoff)

        if outlier_indices.size == 0:
            outlier_index = int(patch_norms.argmax())
            fallback_outlier_count += 1
        else:
            outlier_index = int(rng.choice(outlier_indices.tolist()))

        if normal_indices.size == 0:
            normal_index = int(patch_norms.argmin())
            fallback_normal_count += 1
        else:
            normal_index = int(rng.choice(normal_indices.tolist()))

        features["cls"].append(cls_token)
        features["normal"].append(patch_tokens[normal_index])
        features["outlier"].append(patch_tokens[outlier_index])
        labels.append(example.label)

    stacked = {name: torch.stack(values, dim=0).float() for name, values in features.items()}
    metadata = {
        "examples": len(examples),
        "fallback_outlier_images": fallback_outlier_count,
        "fallback_normal_images": fallback_normal_count,
    }
    return stacked, torch.tensor(labels, dtype=torch.long), metadata


def estimate_cutoff(model: DINOv2TableModel, examples: list[LabeledImage], mode: str, cutoff_mode: str) -> tuple[float, str, dict]:
    """Estimate outlier cutoff from training examples."""

    all_norms = []
    for example in examples:
        _cls_token, _patch_tokens, patch_norms = model.extract_tokens(example.image)
        all_norms.append(patch_norms)
    norms = np.concatenate(all_norms, axis=0)
    cutoff, source = choose_cutoff(norms, mode, cutoff_mode)
    metadata = {
        "cutoff": cutoff,
        "cutoff_source": source,
        "paper_cutoff": PAPER_NORM_CUTOFF,
        "cpu_fallback_percentile": CPU_FALLBACK_PERCENTILE,
        "fraction_above_cutoff": float((norms > cutoff).mean()),
        "fraction_above_paper_cutoff": float((norms > PAPER_NORM_CUTOFF).mean()),
        "max_norm": float(norms.max()),
        "median_norm": float(np.median(norms)),
    }
    return cutoff, source, metadata


def remap_labels(train_y: torch.Tensor, test_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Map dataset labels to contiguous classifier targets."""

    classes = sorted(set(train_y.tolist()) | set(test_y.tolist()))
    mapping = {label: index for index, label in enumerate(classes)}
    mapped_train = torch.tensor([mapping[int(label)] for label in train_y], dtype=torch.long)
    mapped_test = torch.tensor([mapping[int(label)] for label in test_y], dtype=torch.long)
    return mapped_train, mapped_test, len(classes)


def standardize(train_x: torch.Tensor, test_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardize classifier features using train-set statistics."""

    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (test_x - mean) / std


def train_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_classes: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> torch.nn.Linear:
    """Train a linear image classifier."""

    generator = torch.Generator().manual_seed(seed)
    model = torch.nn.Linear(train_x.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    for _epoch in range(epochs):
        order = torch.randperm(train_x.shape[0], generator=generator)
        for start in range(0, order.numel(), batch_size):
            batch = order[start : start + batch_size].to(device)
            loss = F.cross_entropy(model(train_x[batch]), train_y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def accuracy(model: torch.nn.Linear, test_x: torch.Tensor, test_y: torch.Tensor, device: torch.device) -> float:
    """Compute top-1 accuracy."""

    model.eval()
    with torch.no_grad():
        prediction = model(test_x.to(device)).argmax(dim=1).cpu()
    return float((prediction == test_y).float().mean())


def run_dataset(args: argparse.Namespace, model: DINOv2TableModel, dataset_name: str) -> tuple[dict, dict]:
    """Run Table 1 probing for one dataset."""

    train_examples, test_examples, dataset_metadata = load_dataset_pair(args, dataset_name)
    cutoff, _source, cutoff_metadata = estimate_cutoff(model, train_examples, args.mode, args.cutoff_mode)
    train_features, train_labels, train_feature_metadata = extract_split_features(model, train_examples, cutoff, args.seed)
    test_features, test_labels, test_feature_metadata = extract_split_features(model, test_examples, cutoff, args.seed + 1)
    train_y, test_y, num_classes = remap_labels(train_labels, test_labels)

    results: dict[str, float] = {}
    for token_type in ("cls", "normal", "outlier"):
        train_x, test_x = standardize(train_features[token_type], test_features[token_type])
        classifier = train_classifier(
            train_x,
            train_y,
            num_classes,
            model.device,
            args.probe_epochs,
            args.batch_size,
            args.probe_lr,
            args.seed,
        )
        results[token_type] = accuracy(classifier, test_x, test_y, model.device)

    metadata = {
        "dataset": dataset_metadata,
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "num_classes": num_classes,
        "cutoff": cutoff_metadata,
        "train_feature_selection": train_feature_metadata,
        "test_feature_selection": test_feature_metadata,
        "probe_epochs": args.probe_epochs,
        "batch_size": args.batch_size,
        "probe_lr": args.probe_lr,
    }
    return results, metadata


def write_csv(results: dict[str, dict[str, float]], output_csv: Path) -> None:
    """Write Table 1-style results to CSV."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataset_names = list(results)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token"] + dataset_names)
        for token_type in ("cls", "normal", "outlier"):
            writer.writerow([token_type] + [f"{100.0 * results[name][token_type]:.2f}" for name in dataset_names])


def plot_table(results: dict[str, dict[str, float]], output_path: Path) -> None:
    """Render a compact Table 1-style image."""

    dataset_names = list(results)
    rows = ["[CLS]", "normal", "outlier"]
    keys = ["cls", "normal", "outlier"]
    cell_text = [[f"{100.0 * results[name][key]:.1f}" for name in dataset_names] for key in keys]

    figure_width = max(5.5, 1.2 * (len(dataset_names) + 1))
    figure, axis = plt.subplots(figsize=(figure_width, 2.4), constrained_layout=True)
    axis.axis("off")
    table = axis.table(
        cellText=cell_text,
        rowLabels=rows,
        colLabels=dataset_names,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    axis.set_title("Table 1 image classification linear probing")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "exact"), default="cpu", help="Use DINOv2-Large or DINOv2-Giant.")
    parser.add_argument("--device", default="auto", help="Use 'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic sampling.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Dataset/cache directory.")
    parser.add_argument("--no-download", action="store_true", help="Use only cached datasets and model files.")
    parser.add_argument(
        "--datasets",
        default="cifar10,caltech101",
        help="Comma-separated supported datasets. CPU default: cifar10,caltech101.",
    )
    parser.add_argument("--max-train-images", type=int, default=200, help="Max train images per dataset.")
    parser.add_argument("--max-test-images", type=int, default=100, help="Max test images per dataset.")
    parser.add_argument(
        "--cutoff-mode",
        choices=("auto", "paper", "percentile"),
        default="auto",
        help="Outlier norm cutoff: paper=150, percentile=99th percentile, auto=paper when it yields outliers.",
    )
    parser.add_argument("--probe-epochs", type=int, default=80, help="Training epochs for linear classifiers.")
    parser.add_argument("--batch-size", type=int, default=128, help="Classifier training batch size.")
    parser.add_argument("--probe-lr", type=float, default=1e-3, help="Classifier optimizer learning rate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "table1_linear_probe.png",
        help="Output table image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    model = DINOv2TableModel(args.mode, device, local_files_only=args.no_download)

    results: dict[str, dict[str, float]] = {}
    metadata: dict = {
        "paper_result": "Table 1",
        "mode": args.mode,
        "device": str(device),
        "seed": args.seed,
        "model": {"label": model.label, "model_id": model.model_id},
        "datasets": {},
        "skipped_datasets": {},
        "output": str(args.output),
    }

    for dataset_name in [name.strip() for name in args.datasets.split(",") if name.strip()]:
        try:
            dataset_results, dataset_metadata = run_dataset(args, model, dataset_name)
        except Exception as exc:
            print(f"Warning: skipping dataset {dataset_name}: {exc}")
            metadata["skipped_datasets"][dataset_name] = str(exc)
            continue
        canonical_name = dataset_metadata["dataset"]["name"]
        results[canonical_name] = dataset_results
        metadata["datasets"][canonical_name] = dataset_metadata

    if not results:
        raise RuntimeError("No datasets completed. Check dataset availability or use --no-download only with cached data.")

    plot_table(results, args.output)
    csv_path = args.output.with_suffix(".csv")
    write_csv(results, csv_path)
    metadata["csv_output"] = str(csv_path)
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved Table 1 reproduction to {args.output}")
    print(f"Saved CSV to {csv_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
