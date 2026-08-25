"""
Tabular‑to‑image projection (the **deterministic feature-to-image projection**
stage of the attention-guided tabular-to-image framework).

This script is the central implementation of the attention‑guided
tabular‑to‑image transformation described in Section 4 of the paper.
It converts the preprocessed tabular data into fixed, CNN‑compatible
image representations using a deterministic spatial layout derived
entirely from the frozen TabNet step assignments.

Workflow:
  1. Load the preprocessed numerical arrays (`X_train.npy`, `X_test.npy`,
     feature names) and the saved TabNet step assignment CSV.
  2. Apply an importance cutoff (default 0.005) to discard features with
     negligible attention; this step prevents noise pixels from diluting
     the image signal.
  3. Instantiate a layout strategy (`step_row`, `packed`, `step_sparse`,
     `attention_map`, etc.) via the unified layout interface. The layout
     defines the image dimensions and the mapping from each feature to a
     pixel coordinate.
  4. For every sample, place the feature value (or the attention‑importance
     product) at the assigned (row, col) location, producing a single‑channel
     grayscale image of shape `(C=1, H, W)`.
  5. Save the resulting image arrays (`X_train_img.npy`, `X_val_img.npy`,
     `X_test_img.npy`) and a JSON metadata file that records the layout
     geometry, step groups, and feature ordering.

Key properties:
  - **Fully deterministic:** The same step assignments and layout choice
    always produce the same image coordinates. No randomness or learning
    is involved in this stage.
  - **Decoupled:** The image builder does not depend on the CNN or any
    downstream learner. It operates solely on the artefacts produced by
    the TabNet training.
  - **Reproducible:** All parameters (layout name, importance cutoff) are
    captured in the metadata file, enabling exact regeneration of the
    experimental images.

**Role in the Map–Optimize–Learn framework:**
  This script executes the **Map** step after the **Optimize** step
  (TabNet training) has completed. It bridges the interpretable tabular
  model and the CNN learner, materialising the layout geometry that
  embodies the supervised, task‑specific feature attention. The image
  arrays it writes are directly consumed by `train_cnn.py` and
  `evaluate_cnn.py`.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parents[1]

DATASET = os.environ.get("DATASET", "BreastCancer")
MOL_LAYOUT = os.environ.get("MOL_LAYOUT", "step_row").strip()
SEED = int(os.environ.get("SEED", 42))

# Permutation control (E2): which layout to wrap, and which permutation.
BASE_LAYOUT = os.environ.get("BASE_LAYOUT", "step_row").strip()
PERMUTATION_SEED = int(os.environ.get("PERMUTATION_SEED", 0))

# AGT2I-AM decomposition (E3, Section 6.7.3): which ablation control to run.
AM_VARIANT = os.environ.get("AM_VARIANT", "full").strip()


def layout_tag() -> str:
    """Directory suffix. Must match train_cnn.py and evaluate_cnn.py exactly,
    otherwise the CNN stage will not find the images."""
    if MOL_LAYOUT == "shuffled":
        return f"shuffled-{BASE_LAYOUT}-p{PERMUTATION_SEED}_seed{SEED}"
    if MOL_LAYOUT == "attention_map" and AM_VARIANT != "full":
        return f"attention_map-{AM_VARIANT}_seed{SEED}"
    return f"{MOL_LAYOUT}_seed{SEED}"


PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", str(BASE / "data" / "processed" / DATASET)))

# ---- ISOLATION: respect OUTPUT_DIR for parallel safety ----
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(PROCESSED_DIR)))
OUTPUT_DIR = OUTPUT_DIR / layout_tag()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TABNET_OUT = (
    BASE
    / "tabnet_fs"
    / "outputs"
    / f"output_{DATASET}"
)

print(f"Dataset: {DATASET}")
print(f"Layout: {MOL_LAYOUT}")

TABNET_FS_DIR = BASE / "tabnet_fs"
sys.path.insert(0, str(BASE))

from tabnet_fs.layouts.unified_layouts import (
    create_layout_from_config,
    get_available_layouts,
    validate_layout_name,
)

if not validate_layout_name(MOL_LAYOUT):
    raise ValueError(
        f"Invalid layout '{MOL_LAYOUT}'. "
        f"Available: {get_available_layouts()}"
    )

required_files = [
    "X_train.npy",
    "X_test.npy",
    "y_train.npy",
    "y_test.npy",
    "feature_names.npy"
]

for file in required_files:
    path = PROCESSED_DIR / file
    if not path.exists():
        raise FileNotFoundError(path)

X_train_full = np.load(PROCESSED_DIR / "X_train.npy")   # original 105 samples (Iris)
X_test = np.load(PROCESSED_DIR / "X_test.npy")

y_train_full = np.load(PROCESSED_DIR / "y_train.npy")
y_test = np.load(PROCESSED_DIR / "y_test.npy")

feature_names = np.load(
    PROCESSED_DIR / "feature_names.npy",
    allow_pickle=True
).tolist()

# =============================================================================
# Reuse the exact split on which TabNet was trained and the attention
# statistics computed.  Reconstructing it here would work only by
# coincidence (same function, same seed, same data); reading the persisted
# indices makes the guarantee of Section 4.5.2 explicit.
# =============================================================================
VAL_SPLIT = 0.2   # fraction of original training data used for validation

idx_dir = os.environ.get("TABNET_IDX_DIR")
idx_dir = Path(idx_dir) if idx_dir else TABNET_OUT
idx_train_path = idx_dir / "cnn_train_idx.npy"
idx_val_path   = idx_dir / "cnn_val_idx.npy"

if idx_train_path.exists() and idx_val_path.exists():
    idx_fit = np.load(idx_train_path)
    idx_val = np.load(idx_val_path)
    if idx_fit.max() >= len(X_train_full) or idx_val.max() >= len(X_train_full):
        raise ValueError(
            f"Persisted split indices exceed the training set size "
            f"({len(X_train_full)}). Re-run train_tabnet.py for this fold."
        )
    X_train, X_val = X_train_full[idx_fit], X_train_full[idx_val]
    y_train, y_val = y_train_full[idx_fit], y_train_full[idx_val]
    print(f"Reused TabNet split from {idx_train_path.parent}")
else:
    # Indices not yet written (e.g. during HPO where train_tabnet.py and
    # tabnet_image_builder.py run in the same pipeline call).
    # Reconstruct with the same function, seed and data — result is identical.
    from sklearn.model_selection import train_test_split as _tts
    idx_all = np.arange(len(X_train_full))
    idx_fit, idx_val = _tts(
        idx_all, test_size=VAL_SPLIT,
        stratify=y_train_full, random_state=SEED
    )
    X_train, X_val = X_train_full[idx_fit], X_train_full[idx_val]
    y_train, y_val = y_train_full[idx_fit], y_train_full[idx_val]
    print(f"Split indices not found – reconstructed (seed={SEED}, "
          f"test_size={VAL_SPLIT}). This is expected during HPO.")

print(f"Original train shape: {X_train_full.shape}")
print(f"After split: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

# Allow the caller to override the path to the step assignment CSV
step_csv_path = os.environ.get("TABNET_STEP_CSV_PATH")
if step_csv_path:
    step_csv_path = Path(step_csv_path)
else:
    step_csv_path = TABNET_OUT / "tabnet_step_assignment.csv"

if not step_csv_path.exists():
    raise FileNotFoundError(step_csv_path)

step_df = pd.read_csv(step_csv_path)

print(f"Loaded step assignment ({len(step_df)} features)")

# E4 (threshold sensitivity, Section 4.3.3): configurable via env var,
# defaulting to the value used everywhere else in the thesis.
IMPORTANCE_CUTOFF = float(os.environ.get("IMPORTANCE_CUTOFF", "0.005"))

if MOL_LAYOUT == "attention_map":
    print("Attention map layout: keeping all features (no importance cutoff)")
    print("Reordering step_df to match feature_names order...")
    step_df['feature'] = step_df['feature'].astype(str)
    feature_names_str = [str(f) for f in feature_names]
    step_df = step_df.set_index('feature').loc[feature_names_str].reset_index()
    print(f"Step_df order now matches feature_names: {step_df['feature'].tolist()}")
else:
    step_df = step_df[step_df["global_importance"] >= IMPORTANCE_CUTOFF].copy()
    print(f"After importance cutoff: {len(step_df)} features remaining")

if step_df.empty:
    raise ValueError("No features survived importance filtering")

feature_to_idx: Dict[str, int] = {
    str(feature): idx
    for idx, feature in enumerate(feature_names)
}

_layout_kwargs = {}
if MOL_LAYOUT == "shuffled":
    _layout_kwargs = {"base_layout": BASE_LAYOUT,
                      "permutation_seed": PERMUTATION_SEED}
    print(f"Permutation control: wrapping '{BASE_LAYOUT}', "
          f"permutation seed {PERMUTATION_SEED}")
elif MOL_LAYOUT == "attention_map":
    _layout_kwargs = {"variant": AM_VARIANT}
    if AM_VARIANT != "full":
        print(f"AM decomposition control: variant='{AM_VARIANT}'")

layout = create_layout_from_config(MOL_LAYOUT, step_df, **_layout_kwargs)

# Packed layouts only implement map_feature_by_name; map_feature returns
# (0, 0).  When ShuffledLayout wraps one of them, routing must follow the
# BASE layout, not the wrapper's own name.
EFFECTIVE_NAME = getattr(layout, "base_name", layout.name)

channels, height, width = layout.compute_image_shape()

print(f"Image shape: {channels}x{height}x{width}")

def build_layout_images(
    X: np.ndarray
) -> np.ndarray:
    n_samples = X.shape[0]
    images = np.zeros(
        (n_samples, channels, height, width),
        dtype=np.float32
    )

    placed = 0
    missing = 0

    if layout.name == "attention_map":
        weight_matrix = layout.get_weight_matrix()
        skip_norm = getattr(layout, "skip_normalization", False)

        if skip_norm:
            # AM-noNorm (E3): no percentile clip on the input, no [0,1]
            # rescale on the output. Pixel values stay on the standardised
            # scale, matching every other layout.
            raw = np.empty((n_samples, height, width), dtype=np.float32)
            for i in range(n_samples):
                raw[i] = weight_matrix * X[i].astype(np.float32)
            images[:, 0] = raw
            return images

        if build_layout_images.is_training:
            build_layout_images.q01 = np.percentile(X, 1, axis=0)
            build_layout_images.q99 = np.percentile(X, 99, axis=0)

        q01 = build_layout_images.q01
        q99 = build_layout_images.q99

        X_robust = np.clip(X, q01, q99)

        raw = np.empty((n_samples, height, width), dtype=np.float32)

        for i in range(n_samples):
            sample_vec = X_robust[i].astype(np.float32)
            raw[i] = weight_matrix * sample_vec

        if build_layout_images.is_training:
            build_layout_images.img_q01 = np.percentile(raw, 1)
            build_layout_images.img_q99 = np.percentile(raw, 99)

        img_q01 = build_layout_images.img_q01
        img_q99 = build_layout_images.img_q99

        raw = np.clip(raw, img_q01, img_q99)

        images[:, 0] = (
            raw - img_q01
        ) / (img_q99 - img_q01 + 1e-8)

        return images

    # Under the permutation control the coordinate still belongs to
    # feature_name, but the VALUE written there comes from another feature.
    def _source_of(feature_name):
        if hasattr(layout, "resolve_content"):
            return layout.resolve_content(feature_name)
        return feature_name

    if EFFECTIVE_NAME in ("packed", "packed_T"):
        for feature_name in list(step_df["feature"]):
            source_key = str(_source_of(feature_name))
            if source_key not in feature_to_idx:
                missing += 1
                continue
            feature_idx = feature_to_idx[source_key]
            row, col = layout.map_feature_by_name(feature_name)
            if row >= height or col >= width:
                continue
            images[:, 0, row, col] = X[:, feature_idx]
            placed += 1

    else:
        for step, features in layout.step_groups.items():
            for local_rank, feature_name in enumerate(features):
                source_key = str(_source_of(feature_name))
                if source_key not in feature_to_idx:
                    missing += 1
                    continue
                feature_idx = feature_to_idx[source_key]
                row, col = layout.map_feature(step, local_rank)
                if row >= height or col >= width:
                    continue
                images[:, 0, row, col] = X[:, feature_idx]
                placed += 1

    print(f"Placed features: {placed}")
    if missing > 0:
        print(f"Missing features: {missing}")

    return images

# ------------------------------------------------------------
# Compute robust scaling statistics on the FULL training set
# (only relevant for attention_map; harmless for other layouts)
# ------------------------------------------------------------
if layout.name == "attention_map":
    print("\nComputing attention_map robust statistics on full training set...")
    build_layout_images.is_training = True
    _ = build_layout_images(X_train_full)   # triggers statistics computation
    print("Robust statistics captured.")

# ------------------------------------------------------------
# Build actual train/val/test images WITH is_training=False
# (so they use the captured statistics)
# ------------------------------------------------------------
print("\nBuilding images...")
build_layout_images.is_training = False

X_train_img = build_layout_images(X_train)
X_val_img   = build_layout_images(X_val)
X_test_img  = build_layout_images(X_test)

print(f"Train image shape: {X_train_img.shape}")
print(f"Val image shape:   {X_val_img.shape}")
print(f"Test image shape:  {X_test_img.shape}")

# ---- Save all image arrays and labels ----
np.save(OUTPUT_DIR / "X_train_img.npy", X_train_img)
np.save(OUTPUT_DIR / "X_val_img.npy", X_val_img)
np.save(OUTPUT_DIR / "X_test_img.npy", X_test_img)

np.save(OUTPUT_DIR / "y_train.npy", y_train)   # reduced training labels
np.save(OUTPUT_DIR / "y_val.npy", y_val)
np.save(OUTPUT_DIR / "y_test.npy", y_test)     # re‑saved for consistency

# ---- Layout metadata ----
_probe = X_train_img[:min(200, len(X_train_img)), 0]
occupied = int((_probe != 0).any(axis=0).sum())

layout_metadata = {
    "dataset": DATASET,
    "layout_name": MOL_LAYOUT,
    "image_shape": {
        "channels": channels,
        "height": height,
        "width": width
    },
    "step_groups": layout.step_groups,
    "feature_order": list(step_df["feature"]),
    "importance_cutoff": IMPORTANCE_CUTOFF,
    "generation_timestamp": pd.Timestamp.now().isoformat(),    
    "validation_split": VAL_SPLIT,
    "random_seed": SEED,
    "base_layout": BASE_LAYOUT if MOL_LAYOUT == "shuffled" else None,
    "permutation_seed": PERMUTATION_SEED if MOL_LAYOUT == "shuffled" else None,
    "am_variant": AM_VARIANT if MOL_LAYOUT == "attention_map" else None,
    # --- geometry, for Table 6.1 ---
    "n_features_total":    int(len(feature_names)),
    "n_features_retained": int(len(step_df)),
    "retention_rate":      round(len(step_df) / len(feature_names), 4),
    "total_pixels":        int(height * width),
    "occupied_pixels":     occupied,
    "sparsity":            round(1 - occupied / (height * width), 4),
    "degenerate_1d":       bool(height == 1 or width == 1),
    "cnn_capacity_bucket": ("small"  if height * width <= 16
                            else "medium" if height * width <= 100
                            else "large"),
}

metadata_path = OUTPUT_DIR / f"tabnet_layout_{layout_tag()}.json"

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(layout_metadata, f, indent=2, default=str)

step_df.to_csv(
    OUTPUT_DIR / "tabnet_spatial_assignment.csv",
    index=False
)

print("\n" + "=" * 60)
print("TABNET IMAGE BUILD COMPLETE")
print("=" * 60)

print(f"Dataset: {DATASET}")
print(f"Layout: {MOL_LAYOUT}")
print(f"Image shape: {height}x{width}")
print(f"Train samples: {len(X_train_img)}")
print(f"Val samples:   {len(X_val_img)}")
print(f"Test samples:  {len(X_test_img)}")

print(f"\nSaved:")
print(f"  X_train_img.npy")
print(f"  X_val_img.npy")
print(f"  X_test_img.npy")
print(f"  y_train.npy (reduced)")
print(f"  y_val.npy")
print(f"  y_test.npy")
print(f"  {metadata_path.name}")

print("=" * 60)