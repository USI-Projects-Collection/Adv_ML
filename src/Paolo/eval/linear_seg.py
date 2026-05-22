"""
Linear segmentation probe (Tabella 2a — ADE20k mIoU column).

Backbone is frozen. We extract per-patch features for each image, reshape
them into a (C, H_patch, W_patch) feature map, and train a tiny 1×1 Conv2d
that maps C → num_classes. Pixel-wise cross-entropy loss, AdamW, a few
epochs. Then we report mean IoU on the val set.

Mirror of the protocol used by the paper (frozen features + linear decoder).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loaders.ade20k import IGNORE_INDEX, NUM_CLASSES
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


def _patches_to_grid(patches: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """(B, P, C) -> (B, C, H, W)."""
    B, P, C = patches.shape
    assert P == grid_h * grid_w, f"{P} != {grid_h}*{grid_w}"
    return patches.transpose(1, 2).reshape(B, C, grid_h, grid_w)


@torch.inference_mode()
def _extract_feature_maps(
    model: RegisteredViT,
    dataset,
    *,
    batch_size: int,
    device: str,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        feats: (N, C, H_patch, W_patch) on CPU
        masks: (N, H_mask, W_mask) on CPU, int64
    """
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_feats, all_masks = [], []
    gh, gw = model.patch_grid
    for x, m in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        out = model(x)
        fmap = _patches_to_grid(out.patches, gh, gw).cpu()
        all_feats.append(fmap)
        all_masks.append(m)
    return torch.cat(all_feats, dim=0), torch.cat(all_masks, dim=0)


def _miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """preds, targets: (N, H, W) int64. Returns macro-mean IoU."""
    ious = []
    for c in range(1, num_classes + 1):  # skip ignore class 0
        p = preds == c
        t = targets == c
        union = (p | t).sum().item()
        if union == 0:
            continue
        inter = (p & t).sum().item()
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def run_linear_seg(
    model: RegisteredViT,
    train_dataset,
    val_dataset,
    *,
    device: str = "cpu",
    batch_size: int = 4,
    epochs: int = 20,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> dict:
    """
    Frozen backbone -> 1x1 Conv2d head trained pixel-wise.
    Returns {"miou": float, "n_train": int, "n_val": int, "feature_dim": int}.
    """
    feats_tr, masks_tr = _extract_feature_maps(
        model, train_dataset, batch_size=batch_size, device=device,
        desc=f"{model.name} train feats",
    )
    feats_va, masks_va = _extract_feature_maps(
        model, val_dataset, batch_size=batch_size, device=device,
        desc=f"{model.name} val feats",
    )
    C = feats_tr.shape[1]
    H_mask = masks_tr.shape[1]
    W_mask = masks_tr.shape[2]

    head = nn.Conv2d(C, NUM_CLASSES + 1, kernel_size=1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    feats_tr_dev = feats_tr.to(device)
    masks_tr_dev = masks_tr.to(device)

    head.train()
    for epoch in range(epochs):
        perm = torch.randperm(feats_tr_dev.shape[0])
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            f = feats_tr_dev[idx]
            m = masks_tr_dev[idx]
            logits = head(f)  # (B, K, H_patch, W_patch)
            logits = F.interpolate(logits, size=(H_mask, W_mask), mode="bilinear", align_corners=False)
            loss = loss_fn(logits, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/max(1,n_batches):.4f}")

    head.eval()
    with torch.inference_mode():
        feats_va_dev = feats_va.to(device)
        all_preds = []
        for i in range(0, feats_va_dev.shape[0], batch_size):
            f = feats_va_dev[i : i + batch_size]
            logits = head(f)
            logits = F.interpolate(logits, size=(H_mask, W_mask), mode="bilinear", align_corners=False)
            all_preds.append(logits.argmax(dim=1).cpu())
        preds = torch.cat(all_preds, dim=0)
    miou = _miou(preds, masks_va, NUM_CLASSES)

    return {
        "miou": miou,
        "n_train": int(feats_tr.shape[0]),
        "n_val": int(feats_va.shape[0]),
        "feature_dim": int(C),
    }
