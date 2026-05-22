"""
Test-time registers — independent variant.

Variante originale (non descritta nel paper Jiang) del metodo test-time
registers: invece di copiare lo stesso valore max(|act|) su tutti gli N TT
register tokens (convenzione Jiang, vedi `activate_on_registers` nel loro
`shared/hook_fn.py`), prendiamo le **top-N posizioni patch più attive** per
ogni register neuron e ne copiamo il valore signed sul TT register
corrispondente: TT_REG_0 riceve il valore della top-1 outlier patch,
TT_REG_1 della top-2, ecc.

**Importante: come in Jiang, azzeriamo TUTTI i patch token per i register
neurons** (non solo le top-N posizioni). La variante riguarda solo *cosa
scrivi nei TT register*, non quante patch azzeri. Se azzerassi solo le top-N,
gli outlier residui resterebbero sui patch token e dominerebbero la
self-attention, peggiorando le mappe invece di ripulirle. (Vedi prima
versione buggata di questo file: causava crollo di tutte le metriche per
N piccoli — bug corretto.)

Motivazione: nel codice Jiang originale per N>1 i TT registers diventano N
copie ridondanti dello stesso broadcast node; il CLS distribuisce l'attention
su N token quasi identici diluendo il segnale globale.

Ipotesi: assegnando un outlier *distinto* a ogni TT register otteniamo N
broadcast node indipendenti, e questo potrebbe ridurre la diluizione
dell'attention per N grandi.

API: identica a `test_time_registers.py` —
  `load_dinov2_with_tt_registers_independent(N, neurons, ...) -> TTRegDINOv2Independent`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import timm
import torch
import torch.nn as nn

from models.base import RegisteredViT, ViTOutput
from ablation.test_time_registers import (
    cached_register_neurons,  # re-export
    find_register_neurons,    # re-export
    load_neuron_search_images,
)


_TIMM_NAME = "vit_large_patch14_dinov2.lvd142m"


class TTRegDINOv2Independent(RegisteredViT):
    """
    DINOv2-L/14 baseline with N **independent** test-time register tokens at
    the end of the sequence: [CLS, PATCH_0...PATCH_P, TT_REG_0...TT_REG_{N-1}].

    Differenza dal modello Jiang originale: ogni TT_REG_r riceve il valore
    della r-esima patch più attiva (top-(r+1)) per ogni register neuron, e le
    top-N posizioni vengono azzerate (non solo la top-1).
    """

    def __init__(self, *, num_registers: int, register_neurons, img_size: int, **kw):
        backbone = timm.create_model(_TIMM_NAME, pretrained=True, img_size=img_size).eval()
        embed_dim = backbone.embed_dim
        patch_size = backbone.patch_embed.patch_size[0]
        grid = img_size // patch_size

        super().__init__(
            backbone=backbone,
            embed_dim=embed_dim,
            num_registers=num_registers,
            patch_grid=(grid, grid),
            img_size=img_size,
            patch_size=patch_size,
            name=f"DINOv2-L/14 +tt-reg{num_registers} (Jiang, indep)",
        )
        self.tt_registers = nn.Parameter(
            torch.zeros(1, num_registers, embed_dim), requires_grad=False
        )
        neurons_by_layer: dict[int, list[int]] = {}
        for layer, neuron, _score in register_neurons:
            neurons_by_layer.setdefault(layer, []).append(neuron)
        self._neurons_by_layer = neurons_by_layer
        self._hook_handles = []
        self._install_hooks()

    def _install_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
        n_reg = self.num_registers
        if n_reg == 0:
            return
        for layer, neuron_list in self._neurons_by_layer.items():
            neuron_idx = torch.tensor(neuron_list, dtype=torch.long)
            blk = self.backbone.blocks[layer]

            def hook(_m, _inp, output, neuron_idx=neuron_idx, n_reg=n_reg):
                # output: (B, N_total, 4096) post-GELU
                num_prefix = self.backbone.num_prefix_tokens  # 1 (CLS only)
                total = output.shape[1]
                P = total - num_prefix - n_reg

                # Estrai il blocco delle patch (no CLS, no TT registers).
                patch_slice = output[:, num_prefix : num_prefix + P, :]   # (B, P, 4096)
                ndev = neuron_idx.to(output.device)
                sel = patch_slice[..., ndev]                              # (B, P, K)
                abs_sel = sel.abs()
                # Trova le top-N posizioni più "outlier" per ogni neurone.
                topk_vals, topk_idx = abs_sel.topk(min(n_reg, P), dim=1)
                # Recupera i valori signed in quelle posizioni.
                signed_vals = sel.gather(1, topk_idx)                      # (B, n_reg, K)

                # Scrivi gli N valori distinti sui N TT register slot
                # (TT_REG_r riceve la r-esima patch più attiva).
                actual_n = signed_vals.shape[1]
                for r in range(actual_n):
                    output[:, -n_reg + r, ndev] = signed_vals[:, r, :]

                # FIX (rispetto al primo run): azzera *tutti* i patch token
                # per quei neuroni (come Jiang originale), non solo le top-N
                # posizioni. Altrimenti gli outlier residui restano attivi
                # sui patch token e dominano la self-attention.
                output[:, num_prefix : num_prefix + P, ndev] = 0
                return output

            self._hook_handles.append(blk.mlp.act.register_forward_hook(hook))

    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.patch_embed(x)
        x = b._pos_embed(x)
        x = b.patch_drop(x)
        x = b.norm_pre(x)
        if self.num_registers > 0:
            tt = self.tt_registers.expand(x.shape[0], -1, -1).to(x.dtype).to(x.device)
            x = torch.cat([x, tt], dim=1)
        for blk in b.blocks:
            x = blk(x)
        x = b.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> ViTOutput:
        tokens = self._forward_tokens(x)
        cls = tokens[:, 0]
        if self.num_registers > 0:
            patches = tokens[:, 1 : 1 + self.num_patches]
            reg = tokens[:, 1 + self.num_patches :]
        else:
            patches = tokens[:, 1:]
            reg = None
        if patches.shape[1] != self.num_patches:
            raise RuntimeError(
                f"[{self.name}] expected {self.num_patches} patch tokens, got {patches.shape[1]}"
            )
        return ViTOutput(cls=cls, patches=patches, registers=reg)


def load_dinov2_with_tt_registers_independent(
    num_registers: int,
    register_neurons: Sequence[tuple[int, int, float]],
    *,
    img_size: int = 518,
) -> TTRegDINOv2Independent:
    """Variante del modello TT-reg con top-N outlier distribuiti su N register distinti."""
    return TTRegDINOv2Independent(
        num_registers=num_registers,
        register_neurons=register_neurons,
        img_size=img_size,
    )
