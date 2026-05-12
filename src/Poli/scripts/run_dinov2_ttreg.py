"""
Table 2a — extra row 'DINOv2 +tt-reg (Jiang)'.

Runs the same three probes used for Table 2a (linear probe ImageNet, linear
segmentation ADE20k, linear depth NYUd) on DINOv2-L/14 baseline modified at
test-time with the Jiang et al. (2025) method:
  - find register neurons on 100 ImageNet train images
  - add N=4 test-time register tokens at the end of the sequence
  - install forward hooks on `block.mlp.act` for layer 17 (top register-neuron
    layer for DINOv2-L) that copy max(|activation|) onto the TT registers and
    zero patch activations.

Saves results to separate JSON files so the original Table 2a rows stay intact:
  results/table_2a_dinov2_ttreg.json
  results/table_2a_seg_dinov2_ttreg.json
  results/table_2a_depth_dinov2_ttreg.json
"""
import json
import sys
import time
from pathlib import Path

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


IMG_SIZE = 518
LABEL = "DINOv2 ViT-L/14 +tt-reg4 (Jiang)"
NUM_TT_REG = 4
NEURONS_CACHE = ROOT / "results" / "dinov2_tt_register_neurons.pt"

OUT_PROBE = ROOT / "results" / "table_2a_dinov2_ttreg.json"
OUT_SEG = ROOT / "results" / "table_2a_seg_dinov2_ttreg.json"
OUT_DEPTH = ROOT / "results" / "table_2a_depth_dinov2_ttreg.json"


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}", flush=True)
    OUT_PROBE.parent.mkdir(exist_ok=True)

    neurons = cached_register_neurons(NEURONS_CACHE, top_k=50, device=device)
    print(f"[neurons] using top-{len(neurons)} register neurons", flush=True)

    def build_model():
        return load_dinov2_with_tt_registers(
            num_registers=NUM_TT_REG,
            register_neurons=neurons,
            img_size=IMG_SIZE,
        )

    # 1) ImageNet probe
    if OUT_PROBE.exists():
        results_probe = json.loads(OUT_PROBE.read_text())
    else:
        results_probe = {}
    if LABEL not in results_probe or "error" in results_probe.get(LABEL, {}):
        print(f"[run  ] {LABEL} | ImageNet Top-1", flush=True)
        t0 = time.time()
        try:
            m = build_model()
            tfm = make_clf_tfm(IMG_SIZE)
            tr = ImageNetSubset("train", transform=tfm)
            va = ImageNetSubset("val", transform=tfm)
            res = run_linear_probe(m, tr, va, device=device, batch_size=4)
            res["time_sec"] = time.time() - t0
            results_probe[LABEL] = res
            OUT_PROBE.write_text(json.dumps(results_probe, indent=2))
            print(f"[done ] top1={res['top1']:.4f} t={res['time_sec']:.0f}s", flush=True)
            del m
            if device == "mps":
                torch.mps.empty_cache()
        except Exception as e:
            print(f"[error] probe: {type(e).__name__}: {e}", flush=True)
            results_probe[LABEL] = {"error": f"{type(e).__name__}: {e}"}
            OUT_PROBE.write_text(json.dumps(results_probe, indent=2))
    else:
        print(f"[skip ] probe already done: top1={results_probe[LABEL]['top1']:.4f}", flush=True)

    # 2) ADE20k segmentation
    if OUT_SEG.exists():
        results_seg = json.loads(OUT_SEG.read_text())
    else:
        results_seg = {}
    if LABEL not in results_seg or "error" in results_seg.get(LABEL, {}):
        print(f"[run  ] {LABEL} | ADE20k seg", flush=True)
        t0 = time.time()
        try:
            m = build_model()
            tfm = make_seg_tfm(IMG_SIZE)
            tr = ADE20kSubset("train", image_transform=tfm, mask_size=IMG_SIZE)
            va = ADE20kSubset("val", image_transform=tfm, mask_size=IMG_SIZE)
            res = run_linear_seg(m, tr, va, device=device, batch_size=4, epochs=30)
            res["time_sec"] = time.time() - t0
            results_seg[LABEL] = res
            OUT_SEG.write_text(json.dumps(results_seg, indent=2))
            print(f"[done ] miou={res['miou']:.4f} t={res['time_sec']:.0f}s", flush=True)
            del m
            if device == "mps":
                torch.mps.empty_cache()
        except Exception as e:
            print(f"[error] seg: {type(e).__name__}: {e}", flush=True)
            results_seg[LABEL] = {"error": f"{type(e).__name__}: {e}"}
            OUT_SEG.write_text(json.dumps(results_seg, indent=2))
    else:
        print(f"[skip ] seg already done: miou={results_seg[LABEL]['miou']:.4f}", flush=True)

    # 3) NYUd depth
    if OUT_DEPTH.exists():
        results_dep = json.loads(OUT_DEPTH.read_text())
    else:
        results_dep = {}
    if LABEL not in results_dep or "error" in results_dep.get(LABEL, {}):
        print(f"[run  ] {LABEL} | NYUd depth", flush=True)
        t0 = time.time()
        try:
            m = build_model()
            tfm = make_depth_tfm(IMG_SIZE)
            tr = NYUdSubset("train", image_transform=tfm, depth_size=IMG_SIZE)
            va = NYUdSubset("val", image_transform=tfm, depth_size=IMG_SIZE)
            res = run_linear_depth(m, tr, va, device=device, batch_size=4, epochs=30)
            res["time_sec"] = time.time() - t0
            results_dep[LABEL] = res
            OUT_DEPTH.write_text(json.dumps(results_dep, indent=2))
            print(f"[done ] rmse={res['rmse']:.4f} t={res['time_sec']:.0f}s", flush=True)
            del m
            if device == "mps":
                torch.mps.empty_cache()
        except Exception as e:
            print(f"[error] depth: {type(e).__name__}: {e}", flush=True)
            results_dep[LABEL] = {"error": f"{type(e).__name__}: {e}"}
            OUT_DEPTH.write_text(json.dumps(results_dep, indent=2))
    else:
        print(f"[skip ] depth already done: rmse={results_dep[LABEL]['rmse']:.4f}", flush=True)

    print("\n[final] all done")


if __name__ == "__main__":
    main()
