"""
NYU Depth v2 (labeled) subset loader.

The labeled split is a single ~2.9 GB HDF5 .mat file containing 1,449
indoor RGB images (480x640) and dense depth maps (in meters).
We extract a stratified random subset to PNG/NPY on disk for fast reload.

Layout after build_subset:
    data/nyud/{train,val}/
        images/img_{i:04d}.png
        depths/depth_{i:04d}.npy        # float32, meters
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

_PKG_ROOT = Path(__file__).resolve().parents[1]
MAT_PATH = _PKG_ROOT / "data" / "nyud_raw" / "nyu_depth_v2_labeled.mat"
DATA_DIR = _PKG_ROOT / "data" / "nyud"
META_FILE = DATA_DIR / "meta.json"


def build_subset(
    *,
    n_train: int = 140,
    n_val: int = 60,
    seed: int = 42,
    overwrite: bool = False,
) -> dict:
    if META_FILE.exists() and not overwrite:
        meta = json.loads(META_FILE.read_text())
        if meta.get("n_train") == n_train and meta.get("n_val") == n_val:
            print(f"[nyud] subset already on disk")
            return meta

    if not MAT_PATH.exists():
        raise FileNotFoundError(f"Need {MAT_PATH}")

    if overwrite and DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    for split in ("train", "val"):
        (DATA_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / split / "depths").mkdir(parents=True, exist_ok=True)

    import h5py

    print(f"[nyud] opening {MAT_PATH.name} ...")
    with h5py.File(MAT_PATH, "r") as f:
        images = f["images"]    # (N, 3, W, H) uint8 — note transpose
        depths = f["depths"]    # (N, W, H) float32
        N = images.shape[0]
        rng = random.Random(seed)
        idx = list(range(N))
        rng.shuffle(idx)
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]

        def write_split(indices, split):
            for k, i in enumerate(indices):
                img = images[i]                     # (3, W, H)
                img = np.transpose(img, (2, 1, 0))  # (H, W, 3)
                Image.fromarray(img).save(DATA_DIR / split / "images" / f"img_{k:04d}.png")
                d = depths[i]                       # (W, H)
                d = np.transpose(d, (1, 0)).astype(np.float32)  # (H, W)
                np.save(DATA_DIR / split / "depths" / f"depth_{k:04d}.npy", d)

        print(f"[nyud] writing {n_train} train + {n_val} val ...")
        write_split(train_idx, "train")
        write_split(val_idx, "val")

    meta = {
        "dataset": "NYU Depth v2 (labeled)",
        "n_train": n_train,
        "n_val": n_val,
        "seed": seed,
    }
    META_FILE.write_text(json.dumps(meta, indent=2))
    print(f"[nyud] done -> {DATA_DIR}")
    return meta


class NYUdSubset(Dataset):
    def __init__(
        self,
        split: str,
        image_transform: Optional[Callable] = None,
        depth_size: Optional[int] = None,
    ):
        if split not in ("train", "val"):
            raise ValueError(split)
        self.root = DATA_DIR / split
        self.image_transform = image_transform
        self.depth_size = depth_size
        self.img_files = sorted((self.root / "images").glob("*.png"))

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, i: int):
        import torch

        img_path = self.img_files[i]
        img = Image.open(img_path).convert("RGB")
        depth = np.load(self.root / "depths" / f"depth_{img_path.stem.split('_')[1]}.npy")
        if self.depth_size is not None:
            d_img = Image.fromarray(depth)
            d_img = d_img.resize((self.depth_size, self.depth_size), Image.BILINEAR)
            depth = np.asarray(d_img, dtype=np.float32)
        if self.image_transform is not None:
            img = self.image_transform(img)
        return img, torch.from_numpy(depth)


if __name__ == "__main__":
    build_subset()
