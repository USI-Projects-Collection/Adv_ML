"""Reproduce Table 1"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from make_figure2 import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, resolve_device
from make_figure3 import CPU_FALLBACK_PERCENTILE, PAPER_NORM_CUTOFF, split_patch_tokens
from make_figure5 import DEFAULT_CUTOFF_PERCENTILE, choose_cutoff


SUPPORTED_DATASETS = (
    "cifar10",
    "cifar100",
    "caltech101",
    "flowers102",
    "pets",
    "dtd",
    "aircraft",
    "cars",
    "food101",
    "sun397",
)
DEFAULT_SPLIT_FRACTION = 0.8


@dataclass(frozen=True)
class LabeledImage:
    """Single labeled image example."""

    image: Image.Image
    label: int
    label_name: str
    source: str


@dataclass(frozen=True)
class ExtractedImageTokens:
    """Cached DINOv2 outputs for one labeled image."""

    label: int
    cls_token: torch.Tensor
    patch_tokens: torch.Tensor
    patch_norms: np.ndarray


def parse_fraction(value: str) -> float:
    """Parse a dataset fraction in the interval (0, 1]."""

    fraction = float(value)
    if not 0.0 < fraction <= 1.0:
        raise argparse.ArgumentTypeError("dataset fraction must be in the interval (0, 1].")
    return fraction


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


def infer_label_names(dataset, labels: list[int]) -> list[str]:
    """Infer class names from a torchvision dataset, falling back to numeric labels."""

    classes = getattr(dataset, "classes", None)
    if classes is not None:
        return [str(name) for name in classes]
    categories = getattr(dataset, "categories", None)
    if categories is not None:
        return [str(name) for name in categories]
    return [str(index) for index in range(max(labels) + 1 if labels else 0)]


def stratified_subset_indices(
    labels: list[int],
    fraction: float,
    max_images: int | None,
    seed: int,
) -> list[int]:
    """Choose a roughly class-balanced deterministic subset."""

    target_count = int(math.ceil(len(labels) * fraction))
    if max_images is not None:
        target_count = min(target_count, max_images)
    target_count = min(target_count, len(labels))
    if target_count <= 0:
        raise RuntimeError("Requested subset is empty. Increase the dataset fraction or max image limit.")
    if target_count >= len(labels):
        return list(range(len(labels)))

    label_to_indices: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        label_to_indices.setdefault(int(label), []).append(index)

    rng = random.Random(seed)
    per_class = max(1, target_count // max(len(label_to_indices), 1))
    selected: list[int] = []
    selected_set: set[int] = set()
    for indices in label_to_indices.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        for index in shuffled[:per_class]:
            if index not in selected_set:
                selected.append(index)
                selected_set.add(index)
    if len(selected) < target_count:
        remaining = [index for index in range(len(labels)) if index not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[: target_count - len(selected)])
    rng.shuffle(selected)
    return selected[:target_count]


def stratified_train_test_indices(
    labels: list[int],
    train_split_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Create deterministic class-balanced train/test indices for datasets without official splits."""

    label_to_indices: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        label_to_indices.setdefault(int(label), []).append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for indices in label_to_indices.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        train_count = int(round(train_split_fraction * len(shuffled)))
        train_count = min(max(train_count, 1), len(shuffled) - 1)
        train_indices.extend(shuffled[:train_count])
        test_indices.extend(shuffled[train_count:])
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def make_labeled_examples(
    dataset,
    indices: list[int],
    dataset_name: str,
    split: str,
    labels: list[int],
    label_names: list[str],
) -> list[LabeledImage]:
    """Turn a torchvision dataset subset into project LabeledImage records."""

    examples: list[LabeledImage] = []
    for index in indices:
        image, target = dataset[index]
        label = int(target)
        label_name = label_names[label] if 0 <= label < len(label_names) else str(label)
        examples.append(LabeledImage(image.convert("RGB"), label, label_name, f"{dataset_name}-{split}[{index}]"))
    return examples


def load_indexed_split(
    dataset,
    dataset_name: str,
    split: str,
    labels: list[int],
    fraction: float,
    max_images: int | None,
    seed: int,
) -> tuple[list[LabeledImage], dict]:
    """Create a deterministic labeled subset from one torchvision split."""

    indices = stratified_subset_indices(labels, fraction, max_images, seed)
    label_names = infer_label_names(dataset, labels)
    examples = make_labeled_examples(dataset, indices, dataset_name, split, labels, label_names)
    metadata = {
        "name": dataset_name,
        "split": split,
        "classes": label_names,
        "source": f"{dataset.__class__.__module__}.{dataset.__class__.__name__}",
        "available_examples": len(labels),
        "selected_examples": len(examples),
        "selected_fraction": len(examples) / max(len(labels), 1),
    }
    return examples, metadata


def load_concat_splits(
    datasets: list,
    dataset_name: str,
    split_name: str,
    fraction: float,
    max_images: int | None,
    seed: int,
) -> tuple[list[LabeledImage], dict]:
    """Create one labeled subset from multiple torchvision splits concatenated together."""

    labels: list[int] = []
    offsets: list[int] = []
    total = 0
    label_names: list[str] = []
    for dataset in datasets:
        dataset_labels = [int(dataset[index][1]) for index in range(len(dataset))]
        labels.extend(dataset_labels)
        offsets.append(total)
        total += len(dataset_labels)
        if not label_names:
            label_names = infer_label_names(dataset, dataset_labels)

    indices = stratified_subset_indices(labels, fraction, max_images, seed)
    examples: list[LabeledImage] = []
    for global_index in indices:
        for dataset, offset in zip(datasets, offsets):
            if global_index < offset + len(dataset):
                local_index = global_index - offset
                image, target = dataset[local_index]
                label = int(target)
                label_name = label_names[label] if 0 <= label < len(label_names) else str(label)
                examples.append(
                    LabeledImage(image.convert("RGB"), label, label_name, f"{dataset_name}-{split_name}[{local_index}]")
                )
                break
    metadata = {
        "name": dataset_name,
        "split": split_name,
        "classes": label_names,
        "source": f"{datasets[0].__class__.__module__}.{datasets[0].__class__.__name__}",
        "available_examples": total,
        "selected_examples": len(examples),
        "selected_fraction": len(examples) / max(total, 1),
    }
    return examples, metadata


def load_cifar_split(dataset_cls, dataset_name: str, data_dir: Path, split: str, fraction: float, max_images: int | None, seed: int, download: bool) -> tuple[list[LabeledImage], dict]:
    """Load a deterministic CIFAR10/CIFAR100 train or test subset."""

    dataset = dataset_cls(root=str(data_dir), train=(split == "train"), download=download)
    labels = [int(label) for label in dataset.targets]
    return load_indexed_split(dataset, dataset_name, split, labels, fraction, max_images, seed)


def load_caltech101_splits(
    data_dir: Path,
    train_fraction: float,
    test_fraction: float,
    max_train: int | None,
    max_test: int | None,
    split_fraction: float,
    seed: int,
    download: bool,
) -> tuple[list[LabeledImage], list[LabeledImage], dict]:
    """Load deterministic Caltech101 train/test subsets from one dataset."""

    from torchvision.datasets import Caltech101

    dataset = Caltech101(root=str(data_dir), target_type="category", download=download)
    valid_indices: list[int] = []
    valid_labels: list[int] = []
    for index, target in enumerate(dataset.y):
        label = int(target)
        if dataset.categories[label] == "BACKGROUND_Google":
            continue
        valid_indices.append(index)
        valid_labels.append(label)

    train_base, test_base = stratified_train_test_indices(valid_labels, split_fraction, seed)
    train_source = [valid_indices[index] for index in train_base]
    test_source = [valid_indices[index] for index in test_base]
    train_labels = [valid_labels[index] for index in train_base]
    test_labels = [valid_labels[index] for index in test_base]
    train_selected = [train_source[index] for index in stratified_subset_indices(train_labels, train_fraction, max_train, seed)]
    test_selected = [test_source[index] for index in stratified_subset_indices(test_labels, test_fraction, max_test, seed + 1)]
    label_names = [name for name in dataset.categories if name != "BACKGROUND_Google"]
    metadata = {
        "name": "Caltech101",
        "classes": label_names,
        "source": "torchvision.datasets.Caltech101",
        "split_fraction": split_fraction,
        "available_train_examples": len(train_source),
        "available_test_examples": len(test_source),
    }
    return (
        make_labeled_examples(dataset, train_selected, "Caltech101", "train", valid_labels, infer_label_names(dataset, valid_labels)),
        make_labeled_examples(dataset, test_selected, "Caltech101", "test", valid_labels, infer_label_names(dataset, valid_labels)),
        metadata,
    )


def load_custom_split_dataset(
    dataset,
    dataset_name: str,
    train_fraction: float,
    test_fraction: float,
    max_train: int | None,
    max_test: int | None,
    split_fraction: float,
    seed: int,
    labels: list[int] | None = None,
) -> tuple[list[LabeledImage], list[LabeledImage], dict]:
    """Split a torchvision dataset into deterministic train/test subsets."""

    base_labels = labels if labels is not None else [int(dataset[index][1]) for index in range(len(dataset))]
    train_base, test_base = stratified_train_test_indices(base_labels, split_fraction, seed)
    train_labels = [base_labels[index] for index in train_base]
    test_labels = [base_labels[index] for index in test_base]
    train_selected = [train_base[index] for index in stratified_subset_indices(train_labels, train_fraction, max_train, seed)]
    test_selected = [test_base[index] for index in stratified_subset_indices(test_labels, test_fraction, max_test, seed + 1)]
    label_names = infer_label_names(dataset, base_labels)
    metadata = {
        "name": dataset_name,
        "classes": label_names,
        "source": f"{dataset.__class__.__module__}.{dataset.__class__.__name__}",
        "split_fraction": split_fraction,
        "available_train_examples": len(train_base),
        "available_test_examples": len(test_base),
    }
    return (
        make_labeled_examples(dataset, train_selected, dataset_name, "train", base_labels, label_names),
        make_labeled_examples(dataset, test_selected, dataset_name, "test", base_labels, label_names),
        metadata,
    )


def load_dataset_pair(args: argparse.Namespace, name: str) -> tuple[list[LabeledImage], list[LabeledImage], dict]:
    """Load train/test examples for a supported Table 1 dataset."""

    normalized = name.lower()
    if normalized == "cifar10":
        from torchvision.datasets import CIFAR10

        train, train_meta = load_cifar_split(
            CIFAR10,
            "CIFAR10",
            args.data_dir,
            "train",
            args.train_data_fraction,
            args.max_train_images,
            args.seed,
            not args.no_download,
        )
        test, test_meta = load_cifar_split(
            CIFAR10,
            "CIFAR10",
            args.data_dir,
            "test",
            args.test_data_fraction,
            args.max_test_images,
            args.seed + 1,
            not args.no_download,
        )
        return train, test, {"name": "CIFAR10", "train": train_meta, "test": test_meta}
    if normalized == "cifar100":
        from torchvision.datasets import CIFAR100

        train, train_meta = load_cifar_split(
            CIFAR100,
            "CIFAR100",
            args.data_dir,
            "train",
            args.train_data_fraction,
            args.max_train_images,
            args.seed,
            not args.no_download,
        )
        test, test_meta = load_cifar_split(
            CIFAR100,
            "CIFAR100",
            args.data_dir,
            "test",
            args.test_data_fraction,
            args.max_test_images,
            args.seed + 1,
            not args.no_download,
        )
        return train, test, {"name": "CIFAR100", "train": train_meta, "test": test_meta}
    if normalized == "caltech101":
        train, test, metadata = load_caltech101_splits(
            args.data_dir,
            args.train_data_fraction,
            args.test_data_fraction,
            args.max_train_images,
            args.max_test_images,
            args.custom_split_fraction,
            args.seed,
            not args.no_download,
        )
        return train, test, metadata
    if normalized == "flowers102":
        from torchvision.datasets import Flowers102

        test_dataset = Flowers102(root=str(args.data_dir), split="test", download=not args.no_download)
        train, train_meta = load_concat_splits(
            [
                Flowers102(root=str(args.data_dir), split="train", download=not args.no_download),
                Flowers102(root=str(args.data_dir), split="val", download=not args.no_download),
            ],
            "Flowers102",
            "trainval",
            args.train_data_fraction,
            args.max_train_images,
            args.seed,
        )
        test, test_meta = load_indexed_split(
            test_dataset,
            "Flowers102",
            "test",
            [int(label) for label in test_dataset._labels],
            args.test_data_fraction,
            args.max_test_images,
            args.seed + 1,
        )
        return train, test, {"name": "Flowers102", "train": train_meta, "test": test_meta}
    if normalized == "pets":
        from torchvision.datasets import OxfordIIITPet

        train_dataset = OxfordIIITPet(root=str(args.data_dir), split="trainval", target_types="category", download=not args.no_download)
        test_dataset = OxfordIIITPet(root=str(args.data_dir), split="test", target_types="category", download=not args.no_download)
        train, train_meta = load_indexed_split(
            train_dataset,
            "OxfordIIITPet",
            "trainval",
            [int(label) for label in train_dataset._labels],
            args.train_data_fraction,
            args.max_train_images,
            args.seed,
        )
        test, test_meta = load_indexed_split(
            test_dataset,
            "OxfordIIITPet",
            "test",
            [int(label) for label in test_dataset._labels],
            args.test_data_fraction,
            args.max_test_images,
            args.seed + 1,
        )
        return train, test, {"name": "OxfordIIITPet", "train": train_meta, "test": test_meta}
    if normalized == "dtd":
        from torchvision.datasets import DTD

        train, train_meta = load_concat_splits(
            [
                DTD(root=str(args.data_dir), split="train", download=not args.no_download),
                DTD(root=str(args.data_dir), split="val", download=not args.no_download),
            ],
            "DTD",
            "trainval",
            args.train_data_fraction,
            args.max_train_images,
            args.seed,
        )
        test_dataset = DTD(root=str(args.data_dir), split="test", download=not args.no_download)
        test, test_meta = load_indexed_split(
            test_dataset,
            "DTD",
            "test",
            [int(label) for label in test_dataset._labels],
            args.test_data_fraction,
            args.max_test_images,
            args.seed + 1,
        )
        return train, test, {"name": "DTD", "train": train_meta, "test": test_meta}
    if normalized == "aircraft":
        from torchvision.datasets import FGVCAircraft

        train_dataset = FGVCAircraft(root=str(args.data_dir), split="trainval", download=not args.no_download)
        test_dataset = FGVCAircraft(root=str(args.data_dir), split="test", download=not args.no_download)
        train, train_meta = load_indexed_split(
            train_dataset,
            "FGVCAircraft",
            "trainval",
            [int(label) for label in train_dataset._labels],
            args.train_data_fraction,
            args.max_train_images,
            args.seed,
        )
        test, test_meta = load_indexed_split(
            test_dataset,
            "FGVCAircraft",
            "test",
            [int(label) for label in test_dataset._labels],
            args.test_data_fraction,
            args.max_test_images,
            args.seed + 1,
        )
        return train, test, {"name": "FGVCAircraft", "train": train_meta, "test": test_meta}
    if normalized == "cars":
        from torchvision.datasets import StanfordCars

        train_dataset = StanfordCars(root=str(args.data_dir), split="train", download=not args.no_download)
        test_dataset = StanfordCars(root=str(args.data_dir), split="test", download=not args.no_download)
        train, train_meta = load_indexed_split(
            train_dataset,
            "StanfordCars",
            "train",
            [int(label) for _path, label in train_dataset._samples],
            args.train_data_fraction,
            args.max_train_images,
            args.seed,
        )
        test, test_meta = load_indexed_split(
            test_dataset,
            "StanfordCars",
            "test",
            [int(label) for _path, label in test_dataset._samples],
            args.test_data_fraction,
            args.max_test_images,
            args.seed + 1,
        )
        return train, test, {"name": "StanfordCars", "train": train_meta, "test": test_meta}
    if normalized == "food101":
        from torchvision.datasets import Food101

        train_dataset = Food101(root=str(args.data_dir), split="train", download=not args.no_download)
        test_dataset = Food101(root=str(args.data_dir), split="test", download=not args.no_download)
        train, train_meta = load_indexed_split(
            train_dataset,
            "Food101",
            "train",
            [int(label) for label in train_dataset._labels],
            args.train_data_fraction,
            args.max_train_images,
            args.seed,
        )
        test, test_meta = load_indexed_split(
            test_dataset,
            "Food101",
            "test",
            [int(label) for label in test_dataset._labels],
            args.test_data_fraction,
            args.max_test_images,
            args.seed + 1,
        )
        return train, test, {"name": "Food101", "train": train_meta, "test": test_meta}
    if normalized == "sun397":
        from torchvision.datasets import SUN397

        dataset = SUN397(root=str(args.data_dir), download=not args.no_download)
        return load_custom_split_dataset(
            dataset,
            "SUN397",
            args.train_data_fraction,
            args.test_data_fraction,
            args.max_train_images,
            args.max_test_images,
            args.custom_split_fraction,
            args.seed,
            [int(label) for label in dataset._labels],
        )
    raise ValueError(f"Unsupported dataset '{name}'. Supported: {', '.join(SUPPORTED_DATASETS)}.")


def extract_examples(model: DINOv2TableModel, examples: list[LabeledImage]) -> list[ExtractedImageTokens]:
    """Run the backbone once per image and cache token outputs."""

    extracted: list[ExtractedImageTokens] = []
    for example in examples:
        cls_token, patch_tokens, patch_norms = model.extract_tokens(example.image)
        extracted.append(
            ExtractedImageTokens(
                label=example.label,
                cls_token=cls_token,
                patch_tokens=patch_tokens,
                patch_norms=patch_norms,
            )
        )
    return extracted


def extract_split_features(
    extracted_examples: list[ExtractedImageTokens],
    cutoff: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict]:
    """Select one image-level feature per token type from cached outputs."""

    rng = random.Random(seed)
    features = {"cls": [], "normal": [], "outlier": []}
    labels: list[int] = []
    fallback_outlier_count = 0
    fallback_normal_count = 0

    for example in extracted_examples:
        outlier_indices = np.flatnonzero(example.patch_norms > cutoff)
        normal_indices = np.flatnonzero(example.patch_norms <= cutoff)

        if outlier_indices.size == 0:
            outlier_index = int(example.patch_norms.argmax())
            fallback_outlier_count += 1
        else:
            outlier_index = int(rng.choice(outlier_indices.tolist()))

        if normal_indices.size == 0:
            normal_index = int(example.patch_norms.argmin())
            fallback_normal_count += 1
        else:
            normal_index = int(rng.choice(normal_indices.tolist()))

        features["cls"].append(example.cls_token)
        features["normal"].append(example.patch_tokens[normal_index])
        features["outlier"].append(example.patch_tokens[outlier_index])
        labels.append(example.label)

    stacked = {name: torch.stack(values, dim=0).float() for name, values in features.items()}
    metadata = {
        "examples": len(extracted_examples),
        "fallback_outlier_images": fallback_outlier_count,
        "fallback_normal_images": fallback_normal_count,
    }
    return stacked, torch.tensor(labels, dtype=torch.long), metadata


def estimate_cutoff(extracted_examples: list[ExtractedImageTokens], mode: str, cutoff_mode: str) -> tuple[float, str, dict]:
    """Estimate outlier cutoff from cached training outputs."""

    norms = np.concatenate([example.patch_norms for example in extracted_examples], axis=0)
    cutoff, source = choose_cutoff(norms, mode, cutoff_mode, DEFAULT_CUTOFF_PERCENTILE)
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
    max_iter: int,
    seed: int,
):
    """Train a multiclass logistic regression classifier."""

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to run Table 1 logistic regression probes.") from exc

    classifier = LogisticRegression(
        max_iter=max_iter,
        multi_class="auto",
        solver="lbfgs",
        random_state=seed,
    )
    classifier.fit(train_x.numpy(), train_y.numpy())
    return classifier


def accuracy(model, test_x: torch.Tensor, test_y: torch.Tensor) -> float:
    """Compute top-1 accuracy."""

    prediction = torch.from_numpy(model.predict(test_x.numpy())).long()
    return float((prediction == test_y).float().mean())


def run_dataset(args: argparse.Namespace, model: DINOv2TableModel, dataset_name: str) -> tuple[dict, dict]:
    """Run Table 1 probing for one dataset."""

    print(f"[table1] loading dataset {dataset_name}")
    train_examples, test_examples, dataset_metadata = load_dataset_pair(args, dataset_name)
    print(f"[table1] loaded {len(train_examples)} train / {len(test_examples)} test images")
    print(f"[table1] extracting train tokens")
    train_extracted = extract_examples(model, train_examples)
    print(f"[table1] estimating outlier cutoff from cached train tokens")
    cutoff, _source, cutoff_metadata = estimate_cutoff(train_extracted, args.mode, args.cutoff_mode)
    print(
        f"[table1] cutoff={cutoff_metadata['cutoff']:.4f} "
        f"source={cutoff_metadata['cutoff_source']} "
        f"fraction_above_cutoff={100.0 * cutoff_metadata['fraction_above_cutoff']:.2f}%"
    )
    print(f"[table1] selecting cached train features")
    train_features, train_labels, train_feature_metadata = extract_split_features(train_extracted, cutoff, args.seed)
    print(f"[table1] extracting test tokens")
    test_extracted = extract_examples(model, test_examples)
    print(f"[table1] selecting test features")
    test_features, test_labels, test_feature_metadata = extract_split_features(test_extracted, cutoff, args.seed + 1)
    train_y, test_y, num_classes = remap_labels(train_labels, test_labels)
    print(f"[table1] fitting logistic regression probes for {num_classes} classes")

    results: dict[str, float] = {}
    for token_type in ("cls", "normal", "outlier"):
        print(f"[table1] training {token_type} probe")
        train_x, test_x = standardize(train_features[token_type], test_features[token_type])
        classifier = train_classifier(
            train_x,
            train_y,
            args.logreg_max_iter,
            args.seed,
        )
        results[token_type] = accuracy(classifier, test_x, test_y)
        print(f"[table1] {token_type} accuracy={100.0 * results[token_type]:.2f}%")

    metadata = {
        "dataset": dataset_metadata,
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "num_classes": num_classes,
        "cutoff": cutoff_metadata,
        "train_feature_selection": train_feature_metadata,
        "test_feature_selection": test_feature_metadata,
        "logreg_max_iter": args.logreg_max_iter,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "exact"), default="exact", help="Use DINOv2-Large or DINOv2-Giant.")
    parser.add_argument("--device", default="auto", help="Use 'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic sampling.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Dataset/cache directory.")
    parser.add_argument("--no-download", action="store_true", help="Use only cached datasets and model files.")
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="cifar10",
        help="One torchvision dataset to train/evaluate in this run.",
    )
    parser.add_argument(
        "--train-data-fraction",
        type=parse_fraction,
        default=1.0,
        help="Fraction of the training split to use before the hard max-train-images cap.",
    )
    parser.add_argument(
        "--test-data-fraction",
        type=parse_fraction,
        default=1.0,
        help="Fraction of the test split to use before the hard max-test-images cap.",
    )
    parser.add_argument("--max-train-images", type=int, default=200, help="Hard cap on selected training images.")
    parser.add_argument("--max-test-images", type=int, default=100, help="Hard cap on selected test images.")
    parser.add_argument(
        "--custom-split-fraction",
        type=parse_fraction,
        default=DEFAULT_SPLIT_FRACTION,
        help="Train split fraction for datasets without an official torchvision train/test split.",
    )
    parser.add_argument(
        "--cutoff-mode",
        choices=("auto", "paper", "percentile"),
        default="auto",
        help="Outlier norm cutoff: paper=150, percentile=99th percentile, auto=paper when it yields outliers.",
    )
    parser.add_argument("--logreg-max-iter", type=int, default=1000, help="Maximum iterations for sklearn logistic regression.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Optional output path prefix. Defaults to a dataset-specific file stem in src/Kun/results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    print(f"[table1] device={device} mode={args.mode} dataset={args.dataset}")
    print(f"[table1] loading model")
    model = DINOv2TableModel(args.mode, device, local_files_only=args.no_download)
    print(f"[table1] model loaded: {model.label} ({model.model_id})")
    if args.output_prefix is None:
        args.output_prefix = DEFAULT_RESULTS_DIR / f"table1_{args.dataset.lower()}_linear_probe"

    metadata: dict = {
        "paper_result": "Table 1",
        "mode": args.mode,
        "device": str(device),
        "seed": args.seed,
        "dataset": args.dataset,
        "train_data_fraction": args.train_data_fraction,
        "test_data_fraction": args.test_data_fraction,
        "max_train_images": args.max_train_images,
        "max_test_images": args.max_test_images,
        "custom_split_fraction": args.custom_split_fraction,
        "model": {"label": model.label, "model_id": model.model_id},
        "output_prefix": str(args.output_prefix),
    }

    results, dataset_metadata = run_dataset(args, model, args.dataset)
    dataset_name = dataset_metadata["dataset"]["name"]
    metadata["dataset_metadata"] = dataset_metadata

    print(f"[table1] writing outputs")
    csv_path = args.output_prefix.with_suffix(".csv")
    write_csv({dataset_name: results}, csv_path)
    metadata["csv_output"] = str(csv_path)
    metadata_path = args.output_prefix.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved CSV to {csv_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
