"""Run linear probe for all 6 model variants and dump results to JSON."""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loaders.imagenet import ImageNetSubset
from eval.linear_probe import make_transform, run_linear_probe
from models import load_deit3, load_dinov2, load_openclip


CONFIGS = [
    ("DINOv2 ViT-L/14",       lambda: load_dinov2(False, img_size=518), 518),
    ("DINOv2 ViT-L/14 +reg4", lambda: load_dinov2(True,  img_size=518), 518),
    ("OpenCLIP ViT-B/16",      lambda: load_openclip(False),             224),
    ("OpenCLIP +tt-reg4",      lambda: load_openclip(True, num_registers=4), 224),
    ("DeiT-III ViT-B/16",      lambda: load_deit3(False),                224),
    ("DeiT-III +reg4 (inj.)",  lambda: load_deit3(True, num_registers=4), 224),
]

OUT_FILE = ROOT / "results" / "table_2a.json"


def main():
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"[device] {device}", flush=True)

    OUT_FILE.parent.mkdir(exist_ok=True)
    results = json.loads(OUT_FILE.read_text()) if OUT_FILE.exists() else {}

    for label, factory, img_size in CONFIGS:
        if label in results:
            print(f"[skip ] {label} -> top1={results[label]['top1']:.4f}", flush=True)
            continue
        print(f"[run  ] {label}", flush=True)
        t0 = time.time()
        model = factory()
        tfm = make_transform(img_size)
        train_ds = ImageNetSubset("train", transform=tfm)
        val_ds = ImageNetSubset("val", transform=tfm)
        try:
            res = run_linear_probe(model, train_ds, val_ds, device=device, batch_size=4)
        except Exception as e:
            print(f"[error] {label}: {type(e).__name__}: {e}", flush=True)
            results[label] = {"error": f"{type(e).__name__}: {e}"}
            OUT_FILE.write_text(json.dumps(results, indent=2))
            del model
            continue
        dt = time.time() - t0
        res["time_sec"] = dt
        results[label] = res
        OUT_FILE.write_text(json.dumps(results, indent=2))
        print(f"[done ] {label}: top1={res['top1']:.4f}, dim={res['feature_dim']}, time={dt:.0f}s", flush=True)
        del model

    print(f"\n[final] results in {OUT_FILE}")
    for label, r in results.items():
        if "error" in r:
            print(f"  {label}: ERROR {r['error']}")
        else:
            print(f"  {label}: top1={r['top1']:.4f}")


if __name__ == "__main__":
    main()
