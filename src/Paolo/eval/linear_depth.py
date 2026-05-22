"""
Linear depth probe (Tabella 2a — NYUd RMSE column).

Frozen backbone -> 1x1 Conv2d head: C -> 1.
Trained pixel-wise with MSE on log-depth (more stable than raw meters).
We report RMSE in meters on val.
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


def _patches_to_grid(patches, gh, gw):
    B, P, C = patches.shape
    return patches.transpose(1, 2).reshape(B, C, gh, gw)


@torch.inference_mode()
def _extract(model, dataset, *, batch_size, device, desc):
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, depths = [], []
    gh, gw = model.patch_grid
    for x, d in tqdm(loader, desc=desc, leave=False):
        x = x.to(device)
        out = model(x)
        feats.append(_patches_to_grid(out.patches, gh, gw).cpu())
        depths.append(d)
    return torch.cat(feats, dim=0), torch.cat(depths, dim=0)


def run_linear_depth(
    model: RegisteredViT,
    train_dataset,
    val_dataset,
    *,
    device: str = "cpu",
    batch_size: int = 4,
    epochs: int = 30,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    min_depth: float = 0.5,
    max_depth: float = 10.0,
) -> dict:
    feats_tr, d_tr = _extract(model, train_dataset, batch_size=batch_size, device=device, desc=f"{model.name} train")
    feats_va, d_va = _extract(model, val_dataset,   batch_size=batch_size, device=device, desc=f"{model.name} val")
    C = feats_tr.shape[1]
    H, W = d_tr.shape[1], d_tr.shape[2]

    head = nn.Conv2d(C, 1, kernel_size=1).to(device)
    # Zero-init keeps initial predictions at log(1m) = 0, avoiding exp() overflow
    # before AdamW finds a sensible scale.
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    feats_tr = feats_tr.to(device)
    d_tr = d_tr.to(device)

    head.train()
    for epoch in range(epochs):
        perm = torch.randperm(feats_tr.shape[0])
        total = 0.0
        nb = 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            f = feats_tr[idx]
            d = d_tr[idx]                                  # (B, H, W) meters
            pred = head(f)                                 # (B, 1, h, w)
            pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
            mask = (d > min_depth) & (d < max_depth)
            log_pred = pred                                # predicting log-depth directly
            log_gt = torch.log(d.clamp(min=1e-3))
            loss = F.mse_loss(log_pred[mask], log_gt[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{epochs}  log-mse={total/max(1,nb):.4f}")

    head.eval()
    with torch.inference_mode():
        feats_va = feats_va.to(device)
        all_pred = []
        for i in range(0, feats_va.shape[0], batch_size):
            f = feats_va[i : i + batch_size]
            pred = head(f)
            pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
            all_pred.append(pred.cpu())
        log_pred = torch.cat(all_pred, dim=0)
        log_pred = log_pred.clamp(np.log(min_depth), np.log(max_depth))
        meter_pred = torch.exp(log_pred)
    mask = (d_va > min_depth) & (d_va < max_depth)
    rmse = torch.sqrt(((meter_pred[mask] - d_va[mask]) ** 2).mean()).item()

    return {
        "rmse": float(rmse),
        "n_train": int(feats_tr.shape[0]),
        "n_val": int(feats_va.shape[0]),
        "feature_dim": int(C),
    }
