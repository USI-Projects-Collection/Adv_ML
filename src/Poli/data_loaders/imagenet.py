"""
ImageNet-1k subset loader.

We pick a small stratified subset (default 50 classes x 10 images) from the
official ImageNet validation split on HuggingFace, save the images to disk
under data/imagenet/{train,val}/, and provide a torch Dataset that reads them.

Why "validation" only: the official train split is too large to stream
through quickly; the validation split is 50,000 images with reliable labels
and is what the paper uses to report Top-1 accuracy.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset

# Project root (Adv_ML/src/Poli/)
_PKG_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = _PKG_ROOT / "data" / "imagenet"
META_FILE = DATA_DIR / "meta.json"


def download_subset(
    *,
    num_classes: int = 50,
    images_per_class: int = 10,
    train_ratio: float = 0.7,
    seed: int = 42,
    overwrite: bool = False,
) -> dict:
    """
    Download a stratified subset from HF ILSVRC/imagenet-1k validation split.

    Layout written to disk:
        data/imagenet/
            train/
                {class_idx}/
                    img_{i}.jpg
            val/
                {class_idx}/
                    img_{i}.jpg
            meta.json    # {classes: [...], train_count, val_count, ...}

    Returns the meta dict.
    """
    if META_FILE.exists() and not overwrite:
        with open(META_FILE) as f:
            meta = json.load(f)
        if (
            meta.get("num_classes") == num_classes
            and meta.get("images_per_class") == images_per_class
        ):
            print(f"[imagenet] subset already on disk at {DATA_DIR}")
            return meta

    from datasets import load_dataset

    rng = random.Random(seed)
    print(f"[imagenet] streaming validation split from HuggingFace ...")
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

    # Pick the first `num_classes` class indices we encounter (deterministic
    # via shuffled iteration). Using the lowest 50 class indices keeps things
    # reproducible and avoids the streaming shuffle buffer overhead.
    target_classes = sorted(rng.sample(range(1000), num_classes))
    target_set = set(target_classes)

    bucket: dict[int, list[Image.Image]] = defaultdict(list)
    needed = num_classes * images_per_class
    seen = 0

    for sample in ds:
        seen += 1
        label = sample["label"]
        if label not in target_set:
            continue
        if len(bucket[label]) >= images_per_class:
            continue
        # Convert lazily-loaded PIL image to RGB bytes (JpegImageFile may be
        # tied to an open file handle that closes when streaming advances).
        img = sample["image"].convert("RGB")
        bucket[label].append(img)

        total_collected = sum(len(v) for v in bucket.values())
        if total_collected >= needed:
            break

        if seen % 500 == 0:
            print(
                f"  scanned {seen} images, collected {total_collected}/{needed}"
            )

    print(f"[imagenet] collected {sum(len(v) for v in bucket.values())} images")

    # Stratified train/val split per class
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    if overwrite and DATA_DIR.exists():
        import shutil

        shutil.rmtree(DATA_DIR)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    n_train_per_class = round(images_per_class * train_ratio)
    train_count = val_count = 0
    for cls, imgs in bucket.items():
        rng.shuffle(imgs)
        train_imgs = imgs[:n_train_per_class]
        val_imgs = imgs[n_train_per_class:]
        (train_dir / str(cls)).mkdir(exist_ok=True)
        (val_dir / str(cls)).mkdir(exist_ok=True)
        for i, im in enumerate(train_imgs):
            im.save(train_dir / str(cls) / f"img_{i:03d}.jpg", quality=95)
            train_count += 1
        for i, im in enumerate(val_imgs):
            im.save(val_dir / str(cls) / f"img_{i:03d}.jpg", quality=95)
            val_count += 1

    meta = {
        "dataset": "ImageNet-1k val (HF ILSVRC/imagenet-1k)",
        "num_classes": num_classes,
        "images_per_class": images_per_class,
        "train_ratio": train_ratio,
        "seed": seed,
        "classes": target_classes,
        "train_count": train_count,
        "val_count": val_count,
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[imagenet] wrote {train_count} train + {val_count} val images to {DATA_DIR}")
    return meta


class ImageNetSubset(Dataset):
    """Reads images from data/imagenet/{split}/{class_idx}/*.jpg."""

    def __init__(self, split: str, transform: Optional[Callable] = None):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.root = DATA_DIR / split
        if not self.root.exists():
            raise FileNotFoundError(
                f"{self.root} not found — run data_loaders.imagenet.download_subset() first"
            )
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for cls_dir in sorted(self.root.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_idx = int(cls_dir.name)
            for img_path in sorted(cls_dir.glob("*.jpg")):
                self.samples.append((img_path, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


if __name__ == "__main__":
    download_subset()
