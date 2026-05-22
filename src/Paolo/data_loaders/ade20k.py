"""
ADE20k subset loader.

Reads the official ADE20k Scene Parsing Challenge 2016 release
(downloaded from data.csail.mit.edu) and extracts a small subset
into data/ade20k/{train,val}/{images, masks}/ for our linear
segmentation probe.

Layout of the official zip after extraction:
    ADEChallengeData2016/
        images/training/ADE_train_*.jpg     (20,210 images)
        images/validation/ADE_val_*.jpg     (2,000 images)
        annotations/training/ADE_train_*.png  (mask, uint8, values 0..150)
        annotations/validation/ADE_val_*.png

ADE20k has 150 classes + 1 background (class 0 = void / ignore).
"""
from __future__ import annotations

import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset

_PKG_ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = _PKG_ROOT / "data" / "ade20k_raw" / "ADEChallengeData2016.zip"
DATA_DIR = _PKG_ROOT / "data" / "ade20k"
META_FILE = DATA_DIR / "meta.json"

NUM_CLASSES = 150  # +1 for the ignore class 0
IGNORE_INDEX = 0


def build_subset(
    *,
    n_train: int = 140,
    n_val: int = 60,
    seed: int = 42,
    overwrite: bool = False,
) -> dict:
    """Extract a stratified random subset from the official zip."""
    if META_FILE.exists() and not overwrite:
        meta = json.loads(META_FILE.read_text())
        if meta.get("n_train") == n_train and meta.get("n_val") == n_val:
            print(f"[ade20k] subset already on disk at {DATA_DIR}")
            return meta

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Expected official zip at {ZIP_PATH} — download from "
            "https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
        )

    if overwrite and DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    for split, n in [("train", n_train), ("val", n_val)]:
        (DATA_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / split / "masks").mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    print(f"[ade20k] opening {ZIP_PATH.name} ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        names = z.namelist()
        train_imgs = sorted(
            n for n in names
            if n.startswith("ADEChallengeData2016/images/training/") and n.endswith(".jpg")
        )
        val_imgs = sorted(
            n for n in names
            if n.startswith("ADEChallengeData2016/images/validation/") and n.endswith(".jpg")
        )
        rng.shuffle(train_imgs)
        rng.shuffle(val_imgs)
        train_imgs = train_imgs[:n_train]
        val_imgs = val_imgs[:n_val]

        def extract(img_paths, split: str):
            for p in img_paths:
                fname = Path(p).name
                # Image
                with z.open(p) as src, open(DATA_DIR / split / "images" / fname, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                # Matching mask
                stem = Path(p).stem
                mask_name = f"ADEChallengeData2016/annotations/{'training' if split=='train' else 'validation'}/{stem}.png"
                with z.open(mask_name) as src, open(DATA_DIR / split / "masks" / f"{stem}.png", "wb") as dst:
                    shutil.copyfileobj(src, dst)

        print(f"[ade20k] extracting {len(train_imgs)} train + {len(val_imgs)} val ...")
        extract(train_imgs, "train")
        extract(val_imgs, "val")

    meta = {
        "dataset": "ADEChallengeData2016 (MIT scene parsing release)",
        "n_train": n_train,
        "n_val": n_val,
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
        "seed": seed,
    }
    META_FILE.write_text(json.dumps(meta, indent=2))
    print(f"[ade20k] wrote subset to {DATA_DIR}")
    return meta


class ADE20kSubset(Dataset):
    """Returns (PIL image, mask tensor [H, W] uint8, class indices 0..150)."""

    def __init__(
        self,
        split: str,
        image_transform: Optional[Callable] = None,
        mask_size: Optional[int] = None,
    ):
        if split not in ("train", "val"):
            raise ValueError(split)
        self.root = DATA_DIR / split
        if not self.root.exists():
            raise FileNotFoundError(f"{self.root} — call build_subset() first")
        self.image_transform = image_transform
        self.mask_size = mask_size
        self.img_files = sorted((self.root / "images").glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, i: int):
        import numpy as np
        import torch

        img_path = self.img_files[i]
        img = Image.open(img_path).convert("RGB")
        mask_path = self.root / "masks" / f"{img_path.stem}.png"
        mask = Image.open(mask_path)
        if self.mask_size is not None:
            mask = mask.resize((self.mask_size, self.mask_size), Image.NEAREST)
        mask_np = np.asarray(mask, dtype=np.int64)
        if self.image_transform is not None:
            img = self.image_transform(img)
        return img, torch.from_numpy(mask_np)


if __name__ == "__main__":
    build_subset()
