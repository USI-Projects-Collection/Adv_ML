"""
Linear probing for image classification (Tabella 2a — ImageNet column).

Frozen features → LogisticRegression → Top-1 accuracy.

The backbone is fully frozen. We extract one CLS feature vector per image
on train and val, fit a multinomial logistic regression with sklearn, and
report Top-1 accuracy on val.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import RegisteredViT


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def make_transform(img_size: int) -> Callable:
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


@torch.inference_mode()
def extract_cls_features(
    model: RegisteredViT,
    dataset,
    *,
    batch_size: int = 8,
    device: str = "cpu",
    desc: str = "extract",
) -> tuple[np.ndarray, np.ndarray]:
    """Run the backbone on every image and return (CLS features, labels)."""
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    feats: list[np.ndarray] = []
    labels: list[int] = []
    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        out = model(x)
        feats.append(out.cls.cpu().numpy())
        labels.extend(int(v) for v in y)
    X = np.concatenate(feats, axis=0)
    Y = np.asarray(labels, dtype=np.int64)
    return X, Y


def run_linear_probe(
    model: RegisteredViT,
    train_dataset,
    val_dataset,
    *,
    device: str = "cpu",
    batch_size: int = 8,
    C: float = 1.0,
    max_iter: int = 1000,
) -> dict:
    """
    Extract CLS features on train+val, fit LogisticRegression, return metrics.

    Returns: {"top1": float, "n_train": int, "n_val": int, "feature_dim": int}
    """
    X_train, y_train = extract_cls_features(
        model, train_dataset, batch_size=batch_size, device=device, desc=f"{model.name} train"
    )
    X_val, y_val = extract_cls_features(
        model, val_dataset, batch_size=batch_size, device=device, desc=f"{model.name} val"
    )

    clf = LogisticRegression(C=C, max_iter=max_iter, n_jobs=-1)
    clf.fit(X_train, y_train)
    top1 = clf.score(X_val, y_val)

    return {
        "top1": float(top1),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "feature_dim": int(X_train.shape[1]),
    }
