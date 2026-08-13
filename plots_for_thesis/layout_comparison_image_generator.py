#!/usr/bin/env python
"""
Visualise image representations of tabular data for multiple methods.
Methods: IGTD (real), IGTD‑inspired (MDS), DeepInsight, Naive Reshape, AG‑T2I (5 layouts).

AG‑T2I images are loaded from the raw pipeline output arrays and normalised
identically to mol_visualizations.py (global min‑max across train+test).
Non‑AG‑T2I images use global min‑max from the full tabular training set for
consistent contrast. Attention map now shows the correct feature ordering.
Seed defaults to 42 to match the main pipeline.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")

# ---- Automatic project root discovery ----
def find_project_root():
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "preprocessing" / "run_preprocessing.py").exists():
            return current
        current = current.parent
    cwd = Path.cwd()
    if (cwd / "preprocessing" / "run_preprocessing.py").exists():
        return cwd
    raise FileNotFoundError(
        "Could not find project root (expected 'preprocessing/run_preprocessing.py')."
    )

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tabnet_fs"))

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from running_all_models.models_factory import (
    IGTD_Mapper,
    DeepInsightMapper,
    RealIGTDMapper,
)
from tabnet_fs.layouts.unified_layouts import create_layout_from_config

VIZ_OUT = PROJECT_ROOT / "figures"
VIZ_OUT.mkdir(exist_ok=True, parents=True)

AG_T2I_LAYOUTS = ["step_row", "packed", "packed_T", "step_sparse", "attention_map"]
DEFAULT_SEED = 42          # matches the dashboard's default
FOLD = 0

# ---- Index map builders ----
def build_naive_index_map(n_features):
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_features)
    side = int(np.ceil(np.sqrt(n_features)))
    img = np.full((side, side), -1, dtype=int)
    for pos, feat in enumerate(perm):
        r = pos // side
        c = pos % side
        img[r, c] = feat
    return img

def build_real_igtd_index_map(mapper):
    side = mapper.side
    img = np.full((side, side), -1, dtype=int)
    for pos, feat in enumerate(mapper.index):
        r = pos // side
        c = pos % side
        img[r, c] = feat
    return img

def build_deepinsight_index_map(mapper, n_features):
    img = np.full((mapper.side, mapper.side), -1, dtype=int)
    for feat in range(n_features):
        probe = np.zeros((1, n_features), dtype=np.float32)
        probe[0, feat] = 1.0
        probe_img = mapper.transform(probe)[0, 0]
        loc = np.where(probe_img == 1.0)
        if len(loc[0]) > 0:
            r, c = loc[0][0], loc[1][0]
            img[r, c] = feat
    return img

def load_layout_metadata(dataset_name, layout, seed):
    path = PROJECT_ROOT / "data" / "processed" / dataset_name / f"{layout}_seed{seed}" / f"tabnet_layout_{layout}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def build_agt2i_index_map_from_metadata(layout_name, metadata, feature_names):
    """
    Build an index map from layout metadata.
    Returns a 2D int array (H,W) with feature indices, or None.
    """
    if metadata is None or "step_groups" not in metadata:
        return None

    name_to_idx = {str(name): idx for idx, name in enumerate(feature_names)}
    step_groups = metadata["step_groups"]
    H, W = metadata["image_shape"]["height"], metadata["image_shape"]["width"]
    img = np.full((H, W), -1, dtype=int)

    if layout_name in ("step_row", "step_sparse"):
        for step_str, feats in step_groups.items():
            step = int(step_str)
            for local_rank, feat_name in enumerate(feats):
                feat_idx = name_to_idx.get(str(feat_name), -1)
                row = step
                col = local_rank
                if row < H and col < W:
                    img[row, col] = feat_idx
        return img

    if layout_name in ("packed", "packed_T"):
        rows = []
        for step_str, feats in step_groups.items():
            for feat_name in feats:
                rows.append({"feature": feat_name, "dominant_step": int(step_str),
                             "global_importance": 1.0})
        dummy_df = pd.DataFrame(rows)
        layout = create_layout_from_config(layout_name, dummy_df)
        for _, row in dummy_df.iterrows():
            feat_name = row["feature"]
            r, c = layout.map_feature_by_name(feat_name)
            if 0 <= r < H and 0 <= c < W:
                feat_idx = name_to_idx.get(str(feat_name), -1)
                img[r, c] = feat_idx
        return img

    # ---- NEW: attention_map ----
    if layout_name == "attention_map":
        # Column order: features sorted by dominant step, then global importance descending.
        # The step_groups dict already reflects this order from the layout.
        feature_sequence = []
        for step_str in sorted(step_groups.keys(), key=int):
            feature_sequence.extend(step_groups[step_str])

        # Each column corresponds to a feature; every row shows the same feature index.
        for col, feat_name in enumerate(feature_sequence):
            if col >= W:
                break
            feat_idx = name_to_idx.get(feat_name, -1)
            if feat_idx >= 0:
                img[:, col] = feat_idx
        return img

    # Other layouts (should not happen)
    return None

# ---- Load AG‑T2I image from raw arrays (like mol_visualizations) ----
def load_agt2i_image(dataset_name, layout, sample_idx, seed):
    """
    Load the raw test image from the pipeline output folder and normalise
    using the same global min/max as mol_visualizations.py (train+test).
    Returns uint8 grayscale array (0‑255), or None if files missing.
    """
    processed_dir = PROJECT_ROOT / "data" / "processed" / dataset_name
    input_dir = processed_dir / f"{layout}_seed{seed}"

    train_path = input_dir / "X_train_img.npy"
    test_path  = input_dir / "X_test_img.npy"
    if not train_path.exists() or not test_path.exists():
        return None

    X_train_img = np.load(train_path)
    X_test_img  = np.load(test_path)

    # flatten to 2D for min/max
    train_flat = X_train_img[:, 0] if X_train_img.ndim == 4 else X_train_img
    test_flat  = X_test_img[:, 0] if X_test_img.ndim == 4 else X_test_img
    global_min = min(train_flat.min(), test_flat.min())
    global_max = max(train_flat.max(), test_flat.max())

    # extract sample
    img = X_test_img[sample_idx]
    if img.ndim == 3:
        img = img[0]   # (H, W)

    # replicate exact process_for_visualization
    img = np.nan_to_num(img, nan=global_min, posinf=global_max, neginf=global_min)
    denominator = global_max - global_min
    if denominator <= 0:
        normed = np.zeros_like(img)
    else:
        normed = (img - global_min) / denominator
    normed = np.clip(normed, 0.0, 1.0)

    return (normed * 255).astype(np.uint8)

# ---- Non‑AG‑T2I image rendering (global contrast from tabular data) ----
def render_non_agt2i_image(raw_img, global_min, global_max):
    """
    Normalise raw_img using the global min/max from the full tabular training set.
    """
    if global_max - global_min < 1e-8:
        normed = np.zeros_like(raw_img)
    else:
        normed = (raw_img - global_min) / (global_max - global_min)
    return (np.clip(normed, 0.0, 1.0) * 255).astype(np.uint8)

# ---- Main ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--target", type=str, default="Class")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Seed used in pipeline (default: {DEFAULT_SEED})")
    args = parser.parse_args()

    dataset_name = args.dataset
    target_col = args.target
    sample_idx = args.sample
    SEED = args.seed

    raw_path = PROJECT_ROOT / "data" / "raw" / f"{dataset_name}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")
    df = pd.read_csv(raw_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_features = X.shape[1]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y_enc))
    if FOLD >= len(folds):
        raise ValueError(f"Fold {FOLD} out of range")
    train_idx, test_idx = folds[FOLD]

    global_processed = PROJECT_ROOT / "data" / "processed" / dataset_name
    X_train_full = np.load(global_processed / "X_train.npy")
    X_test = np.load(global_processed / "X_test.npy")
    feature_names = np.load(global_processed / "feature_names.npy", allow_pickle=True).tolist()
    feature_names = [str(n) for n in feature_names]

    if sample_idx >= len(X_test):
        raise ValueError(f"Sample index {sample_idx} exceeds test set size {len(X_test)}")

    # Global normalisation constants for non‑AG‑T2I
    GLOBAL_MIN_TAB = X_train_full.min()
    GLOBAL_MAX_TAB = X_train_full.max()

    print("Fitting non‑AG‑T2I mappers...")
    mapper_igtd_inspired = IGTD_Mapper(n_features)
    mapper_igtd_inspired.fit(X_train_full)
    mapper_deepinsight = DeepInsightMapper(n_features)
    mapper_deepinsight.fit(X_train_full)
    mapper_real_igtd = RealIGTDMapper(n_features)
    mapper_real_igtd.fit(X_train_full)

    sample_data = X_test[sample_idx:sample_idx+1]

    methods = {
        "IGTD (real)":       "real_igtd",
        "IGTD-inspired":     "igtd",
        "DeepInsight":       "deepinsight",
        "Naive Reshape":     "naive",
    }
    for layout in AG_T2I_LAYOUTS:
        methods[f"AG-T2I-{layout}"] = layout

    display_images = {}
    index_maps = {}

    # ---- AG‑T2I from raw arrays ----
    for layout in AG_T2I_LAYOUTS:
        method_name = f"AG-T2I-{layout}"
        img = load_agt2i_image(dataset_name, layout, sample_idx, SEED)
        if img is not None:
            display_images[method_name] = img
            meta = load_layout_metadata(dataset_name, layout, SEED)
            idx_map = build_agt2i_index_map_from_metadata(layout, meta, feature_names)
            index_maps[method_name] = idx_map
            print(f"  {method_name}: loaded AG‑T2I image")
        else:
            print(f"  {method_name}: raw data not found – SKIPPING")
            display_images[method_name] = None
            index_maps[method_name] = None

    # ---- Non‑AG‑T2I ----
    for method_name, mode in methods.items():
        if method_name.startswith("AG-T2I-"):
            continue
        try:
            if mode == "igtd":
                mapper = mapper_igtd_inspired
            elif mode == "deepinsight":
                mapper = mapper_deepinsight
            elif mode == "real_igtd":
                mapper = mapper_real_igtd
            elif mode == "naive":
                mapper = None
            else:
                continue

            if mode == "naive":
                rng = np.random.default_rng(SEED)
                perm = rng.permutation(n_features)
                side = int(np.ceil(np.sqrt(n_features)))
                raw = np.zeros(side*side, dtype=sample_data.dtype)
                raw[:n_features] = sample_data[0, perm]
                raw_img = raw.reshape(side, side)
            else:
                raw_img = mapper.transform(sample_data)[0, 0]

            rendered = render_non_agt2i_image(raw_img, GLOBAL_MIN_TAB, GLOBAL_MAX_TAB)
            display_images[method_name] = rendered

            if mode == "igtd":
                idx_map = mapper_igtd_inspired.positions.copy()
            elif mode == "deepinsight":
                idx_map = build_deepinsight_index_map(mapper_deepinsight, n_features)
            elif mode == "real_igtd":
                idx_map = build_real_igtd_index_map(mapper_real_igtd)
            elif mode == "naive":
                idx_map = build_naive_index_map(n_features)
            else:
                idx_map = None
            index_maps[method_name] = idx_map
            print(f"  {method_name}: OK")
        except Exception as e:
            print(f"  {method_name}: FAILED – {e}")
            display_images[method_name] = None
            index_maps[method_name] = None

    # ---- Plot ----
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    axes = axes.flatten()
    method_names = list(methods.keys())

    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.12,
                        wspace=0.35, hspace=0.35)

    for i, ax in enumerate(axes):
        name = method_names[i]
        img = display_images.get(name)
        idx_map = index_maps.get(name)

        if img is not None:
            h, w = img.shape[:2]
            ax.imshow(img, cmap='gray', vmin=0, vmax=255, aspect='equal')
            # add a simple black border around the image
            ax.add_patch(plt.Rectangle((-0.5, -0.5), w, h,
                                       linewidth=1.5, edgecolor='black',
                                       facecolor='none'))
            ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
            ax.grid(which='minor', color='lightgray', linewidth=0.5, linestyle='-')
            ax.set_xticks([])
            ax.set_yticks([])

            # Draw feature numbers only if we have an index map
            if idx_map is not None and idx_map.shape == (h, w):
                for r in range(h):
                    for c in range(w):
                        feat_id = idx_map[r, c]
                        if feat_id >= 0:
                            intensity = img[r, c]
                            text_color = 'black' if intensity > 128 else 'white'
                            fs = min(14, max(6, int(1.8 * min(h, w))))
                            ax.text(c, r, str(feat_id), ha='center', va='center',
                                    fontsize=fs, color=text_color, weight='bold')
            ax.set_title(name, fontsize=10)
        else:
            ax.text(0.5, 0.5, 'FAILED', ha='center', va='center')
            ax.set_title(name, fontsize=10)

    plt.suptitle(f"{dataset_name} – test sample {sample_idx} (of {len(X_test)})", fontsize=14, y=0.98)
    out_plot = VIZ_OUT / f"{dataset_name}_sample{sample_idx}_grid_mol_style.png"
    plt.savefig(out_plot, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {out_plot}")
    plt.show()

if __name__ == "__main__":
    main()