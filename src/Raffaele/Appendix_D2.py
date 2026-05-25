import os
import random
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import FGVCAircraft
from transformers import AutoModel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from skimage.filters import threshold_otsu
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
AIRCRAFT_ROOT = "./src/Raffaele/data/aircraft"
MODEL_SIZE    = "giant"              # "base" (fast) or "giant" (paper-accurate)
IMG_SIZE      = 518                 # DINOv2 native resolution
PATCH_SIZE    = 14
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SEED          = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# MODEL NAMES
# =============================================================================
if MODEL_SIZE == "giant":
    MODEL_NO_REG   = "facebook/dinov2-giant"
    MODEL_WITH_REG = "facebook/dinov2-with-registers-giant"
else:
    MODEL_NO_REG   = "facebook/dinov2-base"
    MODEL_WITH_REG = "facebook/dinov2-with-registers-base"

print(f"Device     : {DEVICE}")
print(f"Model size : {MODEL_SIZE}")
print(f"No-reg     : {MODEL_NO_REG}")
print(f"With-reg   : {MODEL_WITH_REG}")


# =============================================================================
# 1. DATASET
# =============================================================================
def get_aircraft(split: str):
    """
    Returns the FGVC-Aircraft split. Downloads on first run.
    split: "trainval" or "test"
    """
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return FGVCAircraft(
        root=AIRCRAFT_ROOT,
        split=split,
        annotation_level="variant", 
        transform=transform,
        download=True,
    )


# =============================================================================
# 2. FEATURE EXTRACTION
# =============================================================================
def load_model(model_name: str):
    """Load a HuggingFace DINOv2 model and set it to eval mode."""
    print(f"\nLoading {model_name} ...")
    model = AutoModel.from_pretrained(model_name)
    model.eval().to(DEVICE)
    return model


def extract_features(model, dataset, num_regs: int):
    """
    Run the model on the dataset and extract features.

    @params
        - model    : DINOv2 model loaded from HuggingFace
        - dataset  : PyTorch dataset (FGVC-Aircraft split)
        - num_regs : number of register tokens in the model (0 or 4)

    @returns
        - cls_feat   : [N, D]  — [CLS] token embedding
        - patch_feats: [N, n_patches, D]  — all patch token embeddings
        - patch_norms: [N, n_patches]  — L2 norm of each patch embedding
        - reg_feats  : [N, num_regs, D]  — register token embeddings (empty if 0)
        - labels     : [N]  — class label
    """
    # Register hook on last encoder layer to capture pre-norm activations
    prenorm_buffer = {}

    def _hook(module, input, output):
        prenorm_buffer["hs"] = output.detach().cpu()

    last_layer = model.encoder.layer[-1]
    hook_handle = last_layer.register_forward_hook(_hook)

    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                        num_workers=2, pin_memory=True)

    all_cls   = []
    all_patch = []
    all_norms = []
    all_regs  = []
    all_labels= []

    try:
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="Extracting features"):
                imgs = imgs.to(DEVICE)
                out  = model(pixel_values=imgs)
                hs   = out.last_hidden_state 

                cls_tok  = hs[:, 0, :]
                reg_toks = hs[:, 1 : 1 + num_regs, :]
                patches  = hs[:, 1 + num_regs :, :]

                # Norms from pre-norm hook — bimodal signal intact
                hs_pre  = prenorm_buffer["hs"]
                patches_pre = hs_pre[:, 1 + num_regs :, :]
                norms = patches_pre.norm(dim=-1)         

                all_cls.append(cls_tok.cpu())
                all_patch.append(patches.cpu())
                all_norms.append(norms)
                all_regs.append(reg_toks.cpu())
                all_labels.append(labels)
    finally:
        hook_handle.remove()

    return (
        torch.cat(all_cls,    dim=0),   # (N, D)
        torch.cat(all_patch,  dim=0),   # (N, P, D)
        torch.cat(all_norms,  dim=0),   # (N, P)
        torch.cat(all_regs,   dim=0) if num_regs > 0 else None,
        torch.cat(all_labels, dim=0),   # (N,)
    )


# =============================================================================
# 3. ADAPTIVE OUTLIER THRESHOLD
# =============================================================================
def compute_outlier_threshold(patch_norms: torch.Tensor) -> float:
    """
    Find the outlier threshold automatically from the norm distribution.
      1. Otsu on the norm histogram clipped at the 99th percentile.
         Clipping prevents the sparse high-norm tail from dominating the
         histogram, so Otsu finds the valley between the two modes.

      2. Median + 10 x MAD (median absolute deviation).
         MAD is robust to outliers by construction: it measures the spread of
         the bulk of the distribution, so median+10*MAD sits well above the
         normal-patch cluster and below the outlier spike.
         (mean+3std fails here because the outliers inflate both mean and std.)

      3. 95th percentile hard fallback.
    """
    norms = patch_norms.flatten().numpy()

    print(f"  [threshold] norm stats: "
          f"mean={norms.mean():.1f}, median={np.median(norms):.1f}, "
          f"std={norms.std():.1f}, p95={np.percentile(norms, 95):.1f}, "
          f"p99={np.percentile(norms, 99):.1f}, max={norms.max():.1f}")

    def _check(t, name):
        frac = float((norms > t).mean())
        ok = 0.001 < frac < 0.15
        print(f"  [threshold] {name} -> {t:.1f}  ({frac*100:.1f}% outliers)"
              f"  {'OK' if ok else 'skip'}")
        return t if ok else None

    # Strategy 1: Otsu on clipped histogram 
    for clip_pct in [99, 97, 95]:
        try:
            clip_val = float(np.percentile(norms, clip_pct))
            t_otsu = float(threshold_otsu(np.clip(norms, 0, clip_val)))
            result = _check(t_otsu, f"Otsu(clip@p{clip_pct})")
            if result is not None:
                return result
        except Exception as e:
            print(f"  [threshold] Otsu(clip@p{clip_pct}) failed: {e}")

    # Strategy 2: Median + 10 x MAD
    median = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median)))
    result = _check(median + 10.0 * mad, "median+10*MAD")
    if result is not None:
        return result

    # Strategy 3: 95th percentile hard fallback
    t_p95 = float(np.percentile(norms, 95))
    frac = float((norms > t_p95).mean())
    print(f"  [threshold] p95 fallback -> {t_p95:.1f}  ({frac*100:.1f}% outliers)")
    return t_p95


# =============================================================================
# 4. OUTLIER / NORMAL PATCH SELECTION
# =============================================================================
def select_patch_tokens(patch_feats, patch_norms, mode: str, threshold: float):
    """
    For each image, randomly pick ONE patch of the requested type.
    mode: "outlier" — pick one patch with norm > threshold
          "normal"  — pick one patch with norm <= threshold

    @params
        - patch_feats : (N, P, D)  — all patch embeddings
        - patch_norms : (N, P)     — L2 norm of each patch
        - mode        : "outlier" or "normal"
        - threshold   : norm threshold to separate outliers from normal patches
          
    @returns
        - feats  : (N', D)  — one patch per image (images with no eligible patch are dropped)
        - labels : (N',)    — corresponding labels
    """
    N, P, D = patch_feats.shape
    out_feats = []
    valid_idx = []

    for i in range(N):
        norms_i = patch_norms[i]  
        if mode == "outlier":
            eligible = (norms_i > threshold).nonzero(as_tuple=False).squeeze(-1)
        else:
            eligible = (norms_i <= threshold).nonzero(as_tuple=False).squeeze(-1)

        if eligible.numel() == 0:
            continue

        chosen = eligible[random.randint(0, len(eligible) - 1)]
        out_feats.append(patch_feats[i, chosen, :])
        valid_idx.append(i)

    if len(out_feats) == 0:
        raise RuntimeError(
            f"select_patch_tokens(mode='{mode}', threshold={threshold:.1f}): "
            f"no eligible patches found in any of the {N} images. "
            f"Check the threshold"
        )
    return torch.stack(out_feats), torch.tensor(valid_idx)


# =============================================================================
# 5. LINEAR PROBING
# =============================================================================
def linear_probe_classification(train_feats, train_labels,
                                 test_feats,  test_labels,
                                 tag: str):
    """
    Fit a logistic regression on (train_feats, train_labels) and report
    top-1 accuracy on (test_feats, test_labels).
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_feats.numpy())
    X_te = scaler.transform(test_feats.numpy())
    y_tr = train_labels.numpy()
    y_te = test_labels.numpy()

    print(f"  [{tag}] fitting logistic regression "
          f"(train={len(X_tr)}, test={len(X_te)}) ...")
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te) * 100.0
    print(f"  [{tag}] top-1 accuracy = {acc:.1f}%")
    return acc


# =============================================================================
# 6. LOCAL INFORMATION PROBING
# =============================================================================

def linear_probe_position(train_patch, train_norms,
                           test_patch,  test_norms,
                           threshold: float,
                           n_patches: int,
                           tag: str,
                           max_train: int = 50_000,
                           max_test:  int = 20_000):
    """
    Train a linear classifier to predict each patch's grid position.
    We predict (row, col) as a single integer label = row * G + col.

    @returns
        - acc: top-1 accuracy on the test set
    """
    G_fine  = IMG_SIZE // PATCH_SIZE          # 37
    G_coarse = 7                              # coarsen to 7x7 = 49 classes
    step = G_fine // G_coarse                 # ~5 patches per super-cell

    # Build coarse position labels
    fine_idx   = torch.arange(n_patches)      
    row_fine   = fine_idx // G_fine
    col_fine   = fine_idx %  G_fine
    row_coarse = (row_fine  // step).clamp(max=G_coarse - 1)
    col_coarse = (col_fine  // step).clamp(max=G_coarse - 1)
    pos_labels = (row_coarse * G_coarse + col_coarse).long()  

    pos_labels_tr = pos_labels.unsqueeze(0).expand(len(train_patch), -1)  
    pos_labels_te = pos_labels.unsqueeze(0).expand(len(test_patch),  -1)  

    # Keep only normal patches
    mask_tr = (train_norms <= threshold)
    mask_te = (test_norms  <= threshold)

    X_tr = train_patch[mask_tr].numpy()
    y_tr = pos_labels_tr[mask_tr].numpy()
    X_te = test_patch[mask_te].numpy()
    y_te = pos_labels_te[mask_te].numpy()

    # Subsample
    if len(X_tr) > max_train:
        idx = np.random.choice(len(X_tr), max_train, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
    if len(X_te) > max_test:
        idx = np.random.choice(len(X_te), max_test, replace=False)
        X_te, y_te = X_te[idx], y_te[idx]

    print(f"  [{tag}] position probe: {len(X_tr)} train, {len(X_te)} test, "
          f"{G_coarse*G_coarse} classes")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te) * 100.0
    print(f"  [{tag}] position top-1 = {acc:.1f}%")
    return acc


def linear_probe_reconstruction(train_patch, train_norms,
                                  test_patch,  test_norms,
                                  threshold: float,
                                  train_split: str, test_split: str,
                                  tag: str,
                                  max_samples: int = 50_000):
    """
    Train a Ridge regression to predict raw patch pixel values from patch
    embeddings, and report mean L2 error. 

    @returns
        - l2: mean L2 error per patch (see code comments for details)
    """
    patch_pixels = PATCH_SIZE * PATCH_SIZE * 3   # 588 for 14x14 RGB

    raw_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),   # [0, 1], no normalisation
    ])

    def get_raw_patches(split, patch_feats, patch_norms_all, max_n):
        raw_ds = FGVCAircraft(
            root=AIRCRAFT_ROOT,
            split=split,
            annotation_level="variant",
            transform=raw_transform,
            download=False,
        )
        G = IMG_SIZE // PATCH_SIZE   # grid size (37 for 518/14)
        embs, pixs = [], []
        total = 0

        for i in range(len(raw_ds)):
            if total >= max_n:
                break
            img, _ = raw_ds[i]                          
            norms_i = patch_norms_all[i]                
            normal_idx = (norms_i <= threshold).nonzero(as_tuple=False).squeeze(-1)
            if normal_idx.numel() == 0:
                continue

            for p_idx in normal_idx:
                if total >= max_n:
                    break
                p = int(p_idx)
                row, col = p // G, p % G
                # Extract the 14x14 pixel patch at this grid position
                r0, c0 = row * PATCH_SIZE, col * PATCH_SIZE
                patch_pix = img[:, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]   
                pixs.append(patch_pix.flatten())
                embs.append(patch_feats[i, p, :])
                total += 1

        return torch.stack(embs).numpy(), torch.stack(pixs).numpy()

    print(f"  [{tag}] collecting raw train patches (up to {max_samples}) ...")
    X_tr_emb, X_tr_pix = get_raw_patches(train_split, train_patch, train_norms, max_samples)
    print(f"  [{tag}] collecting raw test patches (up to {max_samples//5}) ...")
    X_te_emb, X_te_pix = get_raw_patches(test_split,  test_patch,  test_norms,  max_samples // 5)

    print(f"  [{tag}] reconstruction probe: {len(X_tr_emb)} train, {len(X_te_emb)} test patches")

    scaler = StandardScaler()
    X_tr_emb = scaler.fit_transform(X_tr_emb)
    X_te_emb = scaler.transform(X_te_emb)

    reg = Ridge(alpha=1.0)
    reg.fit(X_tr_emb, X_tr_pix)
    pred = reg.predict(X_te_emb)                        

    # L2 error per patch
    l2 = float(np.mean(np.sqrt(np.sum((X_te_pix - pred) ** 2, axis=1))))
    print(f"  [{tag}] L2 recon error = {l2:.1f}")
    return l2


# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    # Load datasets
    print("\n=== Loading Aircraft dataset ===")
    train_ds = get_aircraft("trainval")
    test_ds  = get_aircraft("test")
    n_patches = (IMG_SIZE // PATCH_SIZE) ** 2
    print(f"  train: {len(train_ds)}, test: {len(test_ds)}, "
          f"patches per image: {n_patches}")

    # TABLE 4 SETUP
    results_t4 = {}

    for num_regs, model_name in [(0, MODEL_NO_REG), (4, MODEL_WITH_REG)]:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}  (num_regs={num_regs})")
        print(f"{'='*60}")

        model = load_model(model_name)

        print("\nExtracting train features ...")
        tr_cls, tr_patch, tr_norms, tr_regs, tr_labels = extract_features(
            model, train_ds, num_regs)

        print("Extracting test features ...")
        te_cls, te_patch, te_norms, te_regs, te_labels = extract_features(
            model, test_ds, num_regs)

        # Compute adaptive threshold from the training norms 
        threshold = compute_outlier_threshold(tr_norms)

        row = {}

        # [CLS] token
        acc = linear_probe_classification(tr_cls, tr_labels, te_cls, te_labels, "[CLS]")
        row["[CLS]"] = acc

        # Normal patch
        tr_feats_n, tr_idx_n = select_patch_tokens(
            tr_patch, tr_norms, "normal", threshold)
        te_feats_n, te_idx_n = select_patch_tokens(
            te_patch, te_norms, "normal", threshold)
        acc = linear_probe_classification(
            tr_feats_n, tr_labels[tr_idx_n],
            te_feats_n, te_labels[te_idx_n], "normal patch")
        row["normal patch"] = acc

        # Outlier patch (only for 0-reg model)
        if num_regs == 0:
            tr_feats_o, tr_idx_o = select_patch_tokens(
                tr_patch, tr_norms, "outlier", threshold)
            te_feats_o, te_idx_o = select_patch_tokens(
                te_patch, te_norms, "outlier", threshold)
            if len(tr_feats_o) > 10 and len(te_feats_o) > 10:
                acc = linear_probe_classification(
                    tr_feats_o, tr_labels[tr_idx_o],
                    te_feats_o, te_labels[te_idx_o], "outlier patch")
                row["outlier patch"] = acc
            else:
                print("  [outlier patch] too few outliers found — "
                      "inspect norm distribution or adjust Otsu sanity bounds")
                row["outlier patch"] = float("nan")
            row["register"] = float("nan")   # N/A for 0-reg model
        else:
            row["outlier patch"] = float("nan")   # N/A (no outliers in reg model)
            # Register token — use first register (reg_0)
            tr_reg0 = tr_regs[:, 0, :]
            te_reg0 = te_regs[:, 0, :]
            acc = linear_probe_classification(
                tr_reg0, tr_labels, te_reg0, te_labels, "register (reg_0)")
            row["register"] = acc

        results_t4[num_regs] = row

        # TABLE 5
        print(f"\n--- Table 5 probing (num_regs={num_regs}) ---")
        pos_acc = linear_probe_position(
            tr_patch, tr_norms, te_patch, te_norms,
            threshold=threshold,
            n_patches=n_patches, tag=f"pos({num_regs} reg)")
        l2_err = linear_probe_reconstruction(
            tr_patch, tr_norms, te_patch, te_norms,
            threshold=threshold,
            train_split="trainval", test_split="test",
            tag=f"recon({num_regs} reg)")

        if num_regs == 0:
            t5_no_reg = (pos_acc, l2_err)
        else:
            t5_with_reg = (pos_acc, l2_err)

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # PRINT TABLES 
    print("\n" + "=" * 65)
    print("TABLE 4 — Global information: Linear probing on Aircraft")
    print("=" * 65)
    print(f"{'#registers':<12} {'[CLS]':>8} {'normal':>8} {'outlier':>9} {'register':>10}")
    print("-" * 65)

    paper_t4 = {
        0: {"[CLS]": 84.6, "normal patch": 15.5, "outlier patch": 73.3, "register": float("nan")},
        1: {"[CLS]": 85.2, "normal patch": 14.5, "outlier patch": float("nan"), "register": 71.1},
    }

    for num_regs in [0, 4]:
        r = results_t4[num_regs]
        p = paper_t4.get(1 if num_regs > 0 else 0, {})

        def fmt(val, ref):
            v = f"{val:.1f}" if not np.isnan(val) else "N/A"
            rv = f"({ref:.1f})" if not np.isnan(ref) else "(N/A)"
            return f"{v:>5} {rv}"

        print(f"{num_regs:<12} "
              f"{fmt(r['[CLS]'],         p.get('[CLS]', float('nan'))):>16}  "
              f"{fmt(r['normal patch'],  p.get('normal patch', float('nan'))):>16}  "
              f"{fmt(r['outlier patch'], p.get('outlier patch', float('nan'))):>16}  "
              f"{fmt(r['register'],      p.get('register', float('nan'))):>16}")

    print("\n" + "=" * 65)
    print("TABLE 5 — Local information: position prediction & reconstruction")
    print("=" * 65)
    print(f"{'#registers':<12} {'patches':<22} {'pos top-1':>10} {'L2 error':>10}")
    print("-" * 65)

    paper_t5 = [(0, "non-outliers", 66.3, 15.9), (4, "non-outliers (all)", 65.8, 16.0)]
    our_t5   = [(0, "non-outliers", t5_no_reg[0], t5_no_reg[1]),
                (4, "non-outliers", t5_with_reg[0], t5_with_reg[1])]

    for i, (num_regs, patches, pos, l2) in enumerate(our_t5):
        p_pos = paper_t5[i][2]
        p_l2  = paper_t5[i][3]
        print(f"{num_regs:<12} {patches:<22} "
              f"{pos:>6.1f} ({p_pos:.1f})  "
              f"{l2:>6.1f} ({p_l2:.1f})")


if __name__ == "__main__":
    main()