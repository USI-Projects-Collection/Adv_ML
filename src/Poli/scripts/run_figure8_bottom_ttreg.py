"""
Figure 8 (bottom) — DINOv2 baseline + N test-time registers (Jiang et al. 2025).

For each N ∈ {0, 1, 2, 4, 8, 16} we add N test-time register tokens to the
**baseline** DINOv2-L/14 (no-reg) and run the same three probes used for
Table 2a. N=0 is the unmodified baseline.

This is a more honest version of Figure 8 than the original
`run_figure8_bottom.py` which started from DINOv2-reg4 and truncated/cycled
the 4 trained registers.

Produces:
  results/figure8_bottom_ttreg.json
  results/figure8_bottom_ttreg.png
"""
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ablation.test_time_registers import (
    cached_register_neurons,
    load_dinov2_with_tt_registers,
)
from data_loaders.ade20k import ADE20kSubset
from data_loaders.imagenet import ImageNetSubset
from data_loaders.nyud import NYUdSubset
from eval.linear_depth import make_transform as make_depth_tfm
from eval.linear_depth import run_linear_depth
from eval.linear_probe import make_transform as make_clf_tfm
from eval.linear_probe import run_linear_probe
from eval.linear_seg import make_transform as make_seg_tfm
from eval.linear_seg import run_linear_seg


REGS = [0, 1, 2, 4, 8, 16]
IMG_SIZE = 518
NEURONS_CACHE = ROOT / "results" / "dinov2_tt_register_neurons.pt"
OUT_FILE = ROOT / "results" / "figure8_bottom_ttreg.json"
OUT_PNG = ROOT / "results" / "figure8_bottom_ttreg.png"


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}", flush=True)
    OUT_FILE.parent.mkdir(exist_ok=True)

    neurons = cached_register_neurons(NEURONS_CACHE, top_k=50, device=device)
    print(f"[neurons] {len(neurons)} register neurons loaded", flush=True)

    results = json.loads(OUT_FILE.read_text()) if OUT_FILE.exists() else {}

    for n in REGS:
        key = str(n)
        if key in results and "error" not in results[key]:
            print(f"[skip ] N={n}: {results[key]}", flush=True)
            continue

        print(f"[run  ] N={n} ImageNet probe", flush=True)
        t0 = time.time()
        try:
            m = load_dinov2_with_tt_registers(n, neurons, img_size=IMG_SIZE)
            tfm = make_clf_tfm(IMG_SIZE)
            tr = ImageNetSubset("train", transform=tfm)
            va = ImageNetSubset("val", transform=tfm)
            r_in = run_linear_probe(m, tr, va, device=device, batch_size=4)
            del m
            if device == "mps":
                torch.mps.empty_cache()

            print(f"[run  ] N={n} ADE20k seg", flush=True)
            m = load_dinov2_with_tt_registers(n, neurons, img_size=IMG_SIZE)
            tfm = make_seg_tfm(IMG_SIZE)
            tr = ADE20kSubset("train", image_transform=tfm, mask_size=IMG_SIZE)
            va = ADE20kSubset("val", image_transform=tfm, mask_size=IMG_SIZE)
            r_seg = run_linear_seg(m, tr, va, device=device, batch_size=4, epochs=30)
            del m
            if device == "mps":
                torch.mps.empty_cache()

            print(f"[run  ] N={n} NYUd depth", flush=True)
            m = load_dinov2_with_tt_registers(n, neurons, img_size=IMG_SIZE)
            tfm = make_depth_tfm(IMG_SIZE)
            tr = NYUdSubset("train", image_transform=tfm, depth_size=IMG_SIZE)
            va = NYUdSubset("val", image_transform=tfm, depth_size=IMG_SIZE)
            r_dep = run_linear_depth(m, tr, va, device=device, batch_size=4, epochs=30)
            del m
            if device == "mps":
                torch.mps.empty_cache()
        except Exception as e:
            print(f"[error] N={n}: {type(e).__name__}: {e}", flush=True)
            results[key] = {"error": f"{type(e).__name__}: {e}"}
            OUT_FILE.write_text(json.dumps(results, indent=2))
            continue

        results[key] = {
            "imagenet_top1": r_in["top1"],
            "ade20k_miou": r_seg["miou"],
            "nyud_rmse": r_dep["rmse"],
            "time_sec": time.time() - t0,
        }
        OUT_FILE.write_text(json.dumps(results, indent=2))
        print(
            f"[done ] N={n}: top1={r_in['top1']:.4f}, "
            f"miou={r_seg['miou']:.4f}, rmse={r_dep['rmse']:.4f}, "
            f"t={results[key]['time_sec']:.0f}s",
            flush=True,
        )

    # Plot
    xs = [int(k) for k in sorted(results.keys(), key=int) if "error" not in results[k]]
    top1 = [results[str(n)]["imagenet_top1"] * 100 for n in xs]
    miou = [results[str(n)]["ade20k_miou"] * 100 for n in xs]
    rmse = [results[str(n)]["nyud_rmse"] for n in xs]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    axes[0].plot(xs, top1, marker="o")
    axes[0].set_title("ImageNet")
    axes[0].set_xlabel("number of [tt-reg] tokens")
    axes[0].set_ylabel("top-1 acc")
    axes[0].set_xticks(xs)
    axes[0].grid(alpha=0.3)

    axes[1].plot(xs, miou, marker="o")
    axes[1].set_title("ADE20k segmentation")
    axes[1].set_xlabel("number of [tt-reg] tokens")
    axes[1].set_ylabel("mIoU")
    axes[1].set_xticks(xs)
    axes[1].grid(alpha=0.3)

    axes[2].plot(xs, rmse, marker="o")
    axes[2].set_title("NYUd depth")
    axes[2].set_xlabel("number of [tt-reg] tokens")
    axes[2].set_ylabel("rmse")
    axes[2].set_xticks(xs)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"\n[final] saved {OUT_PNG}")


if __name__ == "__main__":
    main()
