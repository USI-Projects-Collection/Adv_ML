# Vision Transformers Need Registers Reproduction

This folder contains the implementation for reproducing selected results from
`paper.pdf`, limited to the claims listed in `src/Kun/codex/PLAN.md`.

## Setup

Install the local dependencies:

```bash
python3 -m pip install -r src/Kun/requirements.txt
```

The scripts use PyTorch with `cuda` when requested and available, otherwise CPU.

## Data

All downloaded datasets and model caches should stay under `src/Kun/data` where
possible. Step 1 first tries Caltech101 through `torchvision.datasets.Caltech101`.
Caltech101 is a lightweight image-classification dataset used here only to
select deterministic example images that are different from the paper's Figure 2
examples. If the Caltech101 mirror is unavailable, the script automatically
falls back to the CIFAR10 test split and records that fallback in the metadata.

If downloading datasets is not possible, pass `--image-dir` with a directory
containing at least 4 local RGB images:

```bash
python3 src/Kun/code/make_figure2.py --image-dir path/to/images --mode cpu --device auto --max-images 4
```

## Step 1: Figure 2

Generate a qualitative grid with the original images in the first column and
standalone attention maps in the model columns:

```bash
python3 src/Kun/code/make_figure2.py --mode cpu --device auto --max-images 4
```

Outputs:

- `src/Kun/results/figure2_attention_maps.png`
- `src/Kun/results/figure2_attention_maps.json`

The default layout keeps attention maps separated from the input images, matching
the paper's Figure 2 style. It uses the same Viridis-style color temperature
by default: dark purple/blue for low attention, green for midrange values, and
yellow for high attention. For debugging only, add `--overlay` to blend maps
over the original images.

The CPU mode uses one public checkpoint per model family:

- DeiT-III-B from `timm`
- OpenCLIP ViT-B/16 from `open_clip_torch`
- DINO ViT-B/16 from Hugging Face model `facebook/dino-vitb16`
- DINOv2 ViT-B/14 from Hugging Face model `facebook/dinov2-base`

For a closer paper-style run with the larger variants:

```bash
python3 src/Kun/code/make_figure2.py --mode exact --device cuda --max-images 4
```

Exact mode adds DeiT-III-L, OpenCLIP ViT-L/14, and DINOv2 giant. It is intended
for a GPU machine because DINOv2 giant is not practical for local CPU execution.

## Step 2: Figure 3

Generate DINO vs DINOv2 patch-token L2 norm maps for one image not used in the
Figure 2 grid, plus DINO and DINOv2 norm histograms over a small sampled image
set:

```bash
python3 src/Kun/code/make_figure3.py --mode cpu --device auto --max-hist-images 16
```

Outputs:

- `src/Kun/results/figure3_norms.png`
- `src/Kun/results/figure3_norms.json`

CPU mode uses `facebook/dino-vitb16` and `facebook/dinov2-large`. DINOv2-Large
is slower on CPU than DINOv2-Base, but it is closer to the paper's observation
that high-norm outliers appear in larger DINOv2 models. The script reports the
paper's high-norm cutoff of `150` and also records a 99th-percentile fallback
cutoff. Add `--no-download` only when both datasets and Hugging Face model files
are already cached locally. The paper uses one reference image for the norm maps
and describes the histogram as being over a small image set. To make the
histogram use only the reference image, add `--hist-source reference`.

The norm maps use the last encoder hidden state before the final ViT LayerNorm.
Using Hugging Face `last_hidden_state` would apply the final LayerNorm and hide
the norm outliers that Figure 3 is meant to expose.

For a closer paper-style run:

```bash
python3 src/Kun/code/make_figure3.py --mode exact --device cuda --max-hist-images 64
```

Exact mode uses DINOv2 giant and is intended for a GPU machine.

## Step 3: Figure 5a

Generate the cosine-similarity distribution between DINOv2 input patch
embeddings and their valid 4-neighborhood, split by whether the corresponding
output patch token is normal or high-norm/outlier:

```bash
python3 src/Kun/code/make_figure5.py --part 5a --mode cpu --device auto --max-images 16
```

Outputs:

- `src/Kun/results/figure5a_neighbor_cosine.png`
- `src/Kun/results/figure5a_neighbor_cosine.json`

CPU mode uses `facebook/dinov2-large`. Outliers are defined from output-token
norms before the final ViT LayerNorm, matching the norm extraction used for
Figure 3. By default, `--cutoff-mode auto` uses the paper cutoff `150` when it
finds outliers; otherwise it falls back to the 99th-percentile cutoff so the CPU
proxy still produces normal/outlier comparison curves. Use `--cutoff-mode paper`
to force the paper threshold.

## Step 4: Figure 5b

Train lightweight linear probes on frozen DINOv2 patch embeddings to measure
how much local information is retained by normal vs high-norm/outlier tokens:

```bash
python3 src/Kun/code/make_figure5.py --part 5b --mode cpu --device auto --max-images 16
```

Outputs:

- `src/Kun/results/figure5b_local_probes.png`
- `src/Kun/results/figure5b_local_probes.json`

The position probe predicts patch-grid index and reports top-1 accuracy plus
average Euclidean distance in patch coordinates. The reconstruction probe
predicts flattened preprocessed patch pixels and reports mean per-patch L2
error. Increase `--max-images` for more stable numbers; reduce
`--probe-epochs` for a faster smoke test.

## Step 5: Table 1

Train image-classification linear probes from three frozen DINOv2
representations per image: `[CLS]`, one random normal patch token, and one
random outlier patch token:

```bash
python3 src/Kun/code/make_table1.py --mode cpu --device auto --datasets cifar10
```

Outputs:

- `src/Kun/results/table1_linear_probe.png`
- `src/Kun/results/table1_linear_probe.csv`
- `src/Kun/results/table1_linear_probe.json`

CPU mode uses `facebook/dinov2-large` and capped dataset subsets
(`--max-train-images 200`, `--max-test-images 100` by default). CIFAR10 is the
reliable lightweight default. Caltech101 is attempted when requested, but the
script skips it with a warning if the current torchvision mirror is unavailable.
Use larger caps for more stable scores.
