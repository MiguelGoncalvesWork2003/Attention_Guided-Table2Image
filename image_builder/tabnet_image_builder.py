#tabnet_image_builder.py
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
  5. Save the resulting image arrays (`X_train_img.npy`, `X_test_img.npy`)
     and a JSON metadata file that records the layout geometry, step groups,
     and feature ordering.

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

BASE = Path(__file__).resolve().parents[1]

DATASET = os.environ.get("DATASET", "BreastCancer")
MOL_LAYOUT = os.environ.get("MOL_LAYOUT", "step_row").strip()

PROCESSED_DIR = BASE / "data" / "processed" / DATASET

# ---- ISOLATION: respect OUTPUT_DIR for parallel safety ----
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(PROCESSED_DIR)))
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

sys.path.insert(0, str(TABNET_FS_DIR))

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

X_train = np.load(PROCESSED_DIR / "X_train.npy")
X_test = np.load(PROCESSED_DIR / "X_test.npy")

y_train = np.load(PROCESSED_DIR / "y_train.npy")
y_test = np.load(PROCESSED_DIR / "y_test.npy")

feature_names = np.load(
    PROCESSED_DIR / "feature_names.npy",
    allow_pickle=True
).tolist()

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

step_csv_path = TABNET_OUT / "tabnet_step_assignment.csv"

if not step_csv_path.exists():
    raise FileNotFoundError(step_csv_path)

step_df = pd.read_csv(step_csv_path)

print(f"Loaded step assignment ({len(step_df)} features)")

IMPORTANCE_CUTOFF = 0.005

if MOL_LAYOUT == "attention_map":
    print("Attention map layout: keeping all features (no importance cutoff)")
    print("Reordering step_df to match feature_names order...")
    # Ensure both are strings – necessary when dataset column names are integers
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

layout = create_layout_from_config(
    MOL_LAYOUT,
    step_df
)

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

        # --------------------------------------------------
        # Robust feature clipping parameters (training only)
        # --------------------------------------------------
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

        # --------------------------------------------------
        # Robust image scaling (training statistics only)
        # --------------------------------------------------
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

    if layout.name in ("packed", "packed_T"):
        ordered_features = list(step_df["feature"])
        for feature_name in ordered_features:
            feature_key = str(feature_name)
            if feature_key not in feature_to_idx:
                missing += 1
                continue
            feature_idx = feature_to_idx[feature_key]
            row, col = layout.map_feature_by_name(feature_name)
            if row >= height or col >= width:
                continue
            images[:, 0, row, col] = X[:, feature_idx]
            placed += 1

    else:

        for step, features in layout.step_groups.items():

            for local_rank, feature_name in enumerate(features):

                feature_key = str(feature_name)

                if feature_key not in feature_to_idx:
                    missing += 1
                    continue

                feature_idx = feature_to_idx[feature_key]

                row, col = layout.map_feature(
                    step,
                    local_rank
                )

                if row >= height or col >= width:
                    continue

                images[:, 0, row, col] = X[:, feature_idx]

                placed += 1

    print(f"Placed features: {placed}")

    if missing > 0:
        print(f"Missing features: {missing}")

    return images

print("\nBuilding images...")

build_layout_images.is_training = True
X_train_img = build_layout_images(X_train)

build_layout_images.is_training = False
X_test_img = build_layout_images(X_test)

print(f"Train image shape: {X_train_img.shape}")
print(f"Test image shape: {X_test_img.shape}")

np.save(OUTPUT_DIR / "X_train_img.npy", X_train_img)
np.save(OUTPUT_DIR / "X_test_img.npy", X_test_img)

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
    "generation_timestamp": pd.Timestamp.now().isoformat()
}

metadata_path = (
    OUTPUT_DIR
    / f"tabnet_layout_{MOL_LAYOUT}.json"
)

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
print(f"Test samples: {len(X_test_img)}")

print(f"\nSaved:")
print(f"  X_train_img.npy")
print(f"  X_test_img.npy")
print(f"  {metadata_path.name}")

print("=" * 60)