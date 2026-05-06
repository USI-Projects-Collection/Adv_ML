"""Run zero-shot ImageNet classification (Tabella 2b) for OpenCLIP w/ vs w/o registers."""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torchvision.transforms as T

from data_loaders.imagenet import ImageNetSubset
from data_loaders.imagenet_classes import imagenet_classes
from eval.zero_shot_clip import run_zero_shot_openclip


OUT_FILE = ROOT / "results" / "table_2b.json"


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[device] {device}", flush=True)
    OUT_FILE.parent.mkdir(exist_ok=True)
    results = {}

    # Load class metadata for our 50-class subset.
    meta = json.loads((ROOT / "data" / "imagenet" / "meta.json").read_text())
    class_indices = meta["classes"]
    class_names = [imagenet_classes[i] for i in class_indices]

    # Standard CLIP preprocessing: 224×224, OpenAI mean/std.
    tfm = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    val_ds = ImageNetSubset("val", transform=tfm)

    for label, kwargs in [
        ("OpenCLIP ViT-B/16",      {"with_registers": False}),
        ("OpenCLIP +tt-reg4",      {"with_registers": True, "num_registers": 4}),
    ]:
        print(f"[run  ] {label}", flush=True)
        t0 = time.time()
        try:
            res = run_zero_shot_openclip(
                val_dataset=val_ds,
                class_names=class_names,
                classes_in_subset=class_indices,
                device=device,
                batch_size=8,
                **kwargs,
            )
        except Exception as e:
            print(f"[error] {label}: {type(e).__name__}: {e}", flush=True)
            results[label] = {"error": f"{type(e).__name__}: {e}"}
            OUT_FILE.write_text(json.dumps(results, indent=2))
            continue
        res["time_sec"] = time.time() - t0
        results[label] = res
        OUT_FILE.write_text(json.dumps(results, indent=2))
        print(f"[done ] {label}: top1={res['top1']:.4f}, t={res['time_sec']:.0f}s", flush=True)

    print(f"\n[final] {OUT_FILE}")
    for k, r in results.items():
        if "error" in r:
            print(f"  {k}: ERROR {r['error']}")
        else:
            print(f"  {k}: top1={r['top1']:.4f}")


if __name__ == "__main__":
    main()
