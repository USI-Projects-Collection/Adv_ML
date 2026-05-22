"""
Zero-shot ImageNet classification with OpenCLIP (Tabella 2b).

For each ImageNet class we encode the prompt "a photo of a {class}" with the
text encoder, average over a small ensemble of templates, then classify each
image by cosine similarity between its visual CLS embedding and the 1000
class text embeddings.

We use the standard 80-template ImageNet ensemble from CLIP (Radford 2021).
Restricting to the 50 classes used in our subset is a fair eval: we take
argmax only across those 50 class embeddings.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# Compact prompt ensemble (subset of CLIP paper's 80 templates).
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a photo of many {}.",
    "a photo of the {}.",
    "a low resolution photo of a {}.",
    "a cropped photo of a {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
    "a photo of a small {}.",
    "a photo of a large {}.",
]


@torch.inference_mode()
def encode_text_classifier(
    text_encoder,
    tokenizer,
    class_names: Sequence[str],
    *,
    device: str = "cpu",
) -> torch.Tensor:
    """Returns (num_classes, embed_dim) L2-normalised text embeddings."""
    text_encoder = text_encoder.to(device).eval()
    embeds = []
    for cname in class_names:
        prompts = [t.format(cname) for t in PROMPT_TEMPLATES]
        tokens = tokenizer(prompts).to(device)
        feats = text_encoder.encode_text(tokens)
        feats = F.normalize(feats, dim=-1)
        embeds.append(feats.mean(dim=0))
    embeds = torch.stack(embeds, dim=0)
    embeds = F.normalize(embeds, dim=-1)
    return embeds


@torch.inference_mode()
def zero_shot_eval(
    clip_model,
    visual_forward,
    text_classifier: torch.Tensor,
    dataset,
    class_idx_to_position: dict[int, int],
    *,
    device: str = "cpu",
    batch_size: int = 8,
) -> float:
    """
    Returns Top-1 accuracy.

    visual_forward(x) -> (B, embed_dim) image features (already projected to
    the joint space). Caller must have set up any test-time-register
    parameters before passing it here.
    """
    clip_model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = total = 0
    for x, y in tqdm(loader, desc="zero-shot", leave=False):
        x = x.to(device)
        feats = visual_forward(x)
        feats = F.normalize(feats, dim=-1)
        logits = feats @ text_classifier.T
        preds = logits.argmax(dim=1).cpu().numpy()
        labels_pos = np.asarray(
            [class_idx_to_position[int(yi)] for yi in y], dtype=np.int64
        )
        correct += int((preds == labels_pos).sum())
        total += len(preds)
    return correct / total


def run_zero_shot_openclip(
    *,
    with_registers: bool,
    num_registers: int = 4,
    val_dataset,
    class_names: Sequence[str],
    classes_in_subset: Sequence[int],
    device: str = "cpu",
    batch_size: int = 8,
) -> dict:
    """
    Convenience wrapper: load OpenCLIP (vanilla or test-time-reg), build text
    classifier, run zero-shot, return {"top1": ..., "n_classes": ...}.
    """
    if with_registers:
        from transformers import AutoModel

        m = AutoModel.from_pretrained(
            "amildravid4292/clip-vitb16-test-time-registers",
            trust_remote_code=True,
        ).eval().to(device)
        # Use the OpenCLIP tokenizer for ViT-B-16 — same vocab as the vanilla
        # OpenCLIP B/16 (the HF wrapper exposes the same text tower).
        tokenizer = open_clip.get_tokenizer("ViT-B-16")

        def visual_forward(x):
            return m.model.encode_image(x, num_register_tokens=num_registers)

        text_classifier = encode_text_classifier(m.model, tokenizer, class_names, device=device)
        clip_for_eval = m
    else:
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k"
        )
        clip_model = clip_model.eval().to(device)
        tokenizer = open_clip.get_tokenizer("ViT-B-16")

        def visual_forward(x):
            return clip_model.encode_image(x)

        text_classifier = encode_text_classifier(clip_model, tokenizer, class_names, device=device)
        clip_for_eval = clip_model

    pos = {int(c): i for i, c in enumerate(classes_in_subset)}
    top1 = zero_shot_eval(
        clip_for_eval, visual_forward, text_classifier, val_dataset, pos,
        device=device, batch_size=batch_size,
    )
    return {"top1": float(top1), "n_classes": len(class_names)}
