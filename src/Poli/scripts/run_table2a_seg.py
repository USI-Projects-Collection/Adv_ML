"""Run linear segmentation probe (ADE20k) for all 6 model variants."""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loaders.ade20k import ADE20kSubset
from eval.linear_seg import make_transform, run_linear_seg
from models import load_deit3, load_dinov2, load_openclip


CONFIGS = [
    ("DINOv2 ViT-L/14",       lambda: load_dinov2(False, img_size=518), 518),
    ("DINOv2 ViT-L/14 +reg4", lambda: load_dinov2(True,  img_size=518), 518),
    ("OpenCLIP ViT-B/16",      lambda: load_openclip(False),             224),
    ("OpenCLIP +tt-reg4",      lambda: load_openclip(True, num_registers=4), 224),
    ("DeiT-III ViT-B/16",      lambda: load_deit3(False),                224),
    ("DeiT-III +reg4 (inj.)",  lambda: load_deit3(True, num_registers=4), 224),
]

OUT_FILE = ROOT / "results" / "table_2a_seg.json"


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}", flush=True)

    OUT_FILE.parent.mkdir(exist_ok=True)
    results = json.loads(OUT_FILE.read_text()) if OUT_FILE.exists() else {}

    for label, factory, img_size in CONFIGS:
        if label in results and "error" not in results[label]:
            print(f"[skip ] {label} -> miou={results[label]['miou']:.4f}", flush=True)
            continue
        print(f"[run  ] {label}", flush=True)
        t0 = time.time()
        try:
            model = factory()
            tfm = make_transform(img_size)
            tr = ADE20kSubset("train", image_transform=tfm, mask_size=img_size)
            va = ADE20kSubset("val",   image_transform=tfm, mask_size=img_size)
            res = run_linear_seg(model, tr, va, device=device, batch_size=4, epochs=30)
        except Exception as e:
            print(f"[error] {label}: {type(e).__name__}: {e}", flush=True)
            results[label] = {"error": f"{type(e).__name__}: {e}"}
            OUT_FILE.write_text(json.dumps(results, indent=2))
            continue
        res["time_sec"] = time.time() - t0
        results[label] = res
        OUT_FILE.write_text(json.dumps(results, indent=2))
        print(f"[done ] {label}: miou={res['miou']:.4f}, dim={res['feature_dim']}, t={res['time_sec']:.0f}s", flush=True)
        del model

    print(f"\n[final] {OUT_FILE}")
    for k, r in results.items():
        if "error" in r:
            print(f"  {k}: ERROR {r['error']}")
        else:
            print(f"  {k}: miou={r['miou']:.4f}")


if __name__ == "__main__":
    main()
