"""
Test-time registers — independent variant.

Variante originale (non descritta nel paper Jiang) del metodo test-time
registers: invece di copiare lo stesso valore max(|act|) su tutti gli N TT
register tokens (convenzione Jiang, vedi `activate_on_registers` nel loro
`shared/hook_fn.py`), prendiamo le posizioni patch più attive per ogni
register neuron, **filtriamo per quelle che superano la soglia outlier**, e
copiamo il valore signed sui TT register corrispondenti.

Regola di scrittura:
  - Per ogni register neuron, si ordinano le patch per |attivazione|.
  - Si scrivono sui TT register **solo** le posizioni con
    |attivazione| > `OUTLIER_THRESHOLD` (default 150, lo stesso valore che
    Jiang usa come soglia per la norm dei patch outlier).
  - Se ci sono meno outlier veri di N register, i register rimanenti
    restano a zero (non vengono forzati con valori non-outlier).
  - Se ci sono più di N outlier veri, scriviamo solo i top-N (capping).

**Importante: come in Jiang, azzeriamo TUTTI i patch token per i register
neurons** (non solo le posizioni filtrate). La variante riguarda solo *cosa
scrivi nei TT register*, non quante patch azzeri. Se azzerassi solo le
posizioni filtrate, gli outlier residui resterebbero sui patch token e
dominerebbero la self-attention.

Motivazione: nel codice Jiang originale per N>1 i TT registers diventano N
copie ridondanti dello stesso broadcast node; il CLS distribuisce
l'attention su N token quasi identici diluendo il segnale globale.

Ipotesi: assegnando un outlier *distinto* a ogni TT register otteniamo N
broadcast node indipendenti, e questo potrebbe ridurre la diluizione
dell'attention per N grandi. Il filtro sulla soglia evita di scrivere
nei register valori che non rappresentano veri outlier (cioè immagini con
meno di N patch davvero anomale non subiscono "scraping the barrel").

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

# Soglia outlier (allineata al valore `register_norm_threshold = 150` di Jiang).
# Applicata alla magnitudo dell'attivazione MLP sui register neurons: una patch
# è considerata "vero outlier" su un dato neurone se |act| > OUTLIER_THRESHOLD.
OUTLIER_THRESHOLD: float = 150.0


class TTRegDINOv2Independent(RegisteredViT):
    """
    DINOv2-L/14 baseline with N **independent** test-time register tokens at
    the end of the sequence: [CLS, PATCH_0...PATCH_P, TT_REG_0...TT_REG_{N-1}].

    Differenza dal modello Jiang originale: per ogni register neuron, i TT
    register vengono scritti **solo** quando la magnitudo dell'attivazione
    sulla patch supera `OUTLIER_THRESHOLD` (default 150). I register che non
    ricevono un vero outlier restano a zero.
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
            name=f"DINOv2-L/14 +tt-reg{num_registers} (indep, thr={OUTLIER_THRESHOLD:g})",
        )
        self.tt_registers = nn.Parameter(
            torch.zeros(1, num_registers, embed_dim), requires_grad=False
        )
        neurons_by_layer: dict[int, list[int]] = {}
        for layer, neuron, _score in register_neurons:
            neurons_by_layer.setdefault(layer, []).append(neuron)
        self._neurons_by_layer = neurons_by_layer
        self._hook_handles = []
        # Cache delle patch-norm al layer corrente (popolato dal pre-hook,
        # consumato dal mlp.act hook nello stesso blocco).
        self._current_patch_norms: torch.Tensor | None = None
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

            # Pre-forward hook sul blocco: cattura ||x||_2 per ogni patch
            # all'ingresso del blocco. Sarà usato dal mlp.act hook
            # immediatamente successivo come criterio outlier.
            def pre_hook(_m, inputs, n_reg=n_reg):
                x = inputs[0]
                num_prefix = self.backbone.num_prefix_tokens
                P = x.shape[1] - num_prefix - n_reg
                self._current_patch_norms = x[:, num_prefix : num_prefix + P, :].norm(dim=-1)

            self._hook_handles.append(blk.register_forward_pre_hook(pre_hook))

            def hook(_m, _inp, output, neuron_idx=neuron_idx, n_reg=n_reg, layer=layer):
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
                topk_vals, topk_idx = abs_sel.topk(min(n_reg, P), dim=1)  # (B, n_reg, K)
                # Recupera i valori signed in quelle posizioni.
                signed_vals = sel.gather(1, topk_idx)                     # (B, n_reg, K)

                # Filtro outlier: gating sulla *patch norm* (norma L2 del
                # residual stream all'ingresso del blocco), allineato alla
                # definizione di outlier del paper Jiang (threshold = 150).
                patch_norms = self._current_patch_norms                   # (B, P)
                outlier_per_patch = patch_norms > OUTLIER_THRESHOLD       # (B, P)
                # Espandi al numero di neuroni e gather sulle top-N posizioni.
                K = ndev.shape[0]
                outlier_3d = outlier_per_patch.unsqueeze(-1).expand(-1, -1, K)  # (B, P, K)
                slot_is_outlier = outlier_3d.gather(1, topk_idx)          # (B, n_reg, K)
                zeros = torch.zeros_like(signed_vals)
                gated_vals = torch.where(slot_is_outlier, signed_vals, zeros)

                # Diagnostica.
                filled = int(slot_is_outlier.sum().item())
                total_slots = n_reg * K * slot_is_outlier.shape[0]
                pct = (100.0 * filled / total_slots) if total_slots else 0.0
                pn_max = float(patch_norms.max().item())
                pn_mean = float(patch_norms.mean().item())
                n_outlier_patches = int(outlier_per_patch.sum().item())
                P = patch_norms.shape[-1]
                print(
                    f"  [indep N={n_reg} layer={layer}] {K} neurons, "
                    f"patch-norm max={pn_max:.1f}, mean={pn_mean:.1f}, "
                    f"outlier patches > {int(OUTLIER_THRESHOLD)}: "
                    f"{n_outlier_patches}/{P}, "
                    f"slots filled = {filled}/{total_slots} ({pct:.0f}%)"
                )

                actual_n = gated_vals.shape[1]
                for r in range(actual_n):
                    output[:, -n_reg + r, ndev] = gated_vals[:, r, :]

                # Azzera *tutti* i patch token per quei neuroni (come Jiang
                # originale): l'intervento di pulizia sui patch è
                # indipendente dal filtro sulla writing policy.
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
    """Variante TT-reg con outlier distinti gated dalla soglia OUTLIER_THRESHOLD."""
    return TTRegDINOv2Independent(
        num_registers=num_registers,
        register_neurons=register_neurons,
        img_size=img_size,
    )
