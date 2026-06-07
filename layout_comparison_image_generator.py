# layout_comparison_figure.py
"""
Generate a single compact figure comparing seven tabular‑to‑image layouts
for the same test sample, using the actual preprocessed data and TabNet
step assignments.  An optional attention‑mask heatmap can be prepended.

Each layout image is independently scaled to [0, 1] so that its full
dynamic range is always visible.  Panel spacing is minimised to keep
the figure compact.

Usage:
    DATASET=Cancer python layout_comparison_figure.py
"""

import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------- path setup ----------
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "tabnet_fs"))   # to import unified_layouts

DATASET = os.environ.get("DATASET", "Cancer")
PROCESSED_DIR = BASE / "pasta" / "data" / "processed" / DATASET
MASKS_DIR = BASE / "pasta" / "tabnet_fs" / "outputs" / f"output_{DATASET}"
OUTPUT_PATH = BASE / "layout_comparison_compact.pdf"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

TITLES = {
    "step_row": "AG-T2I-StepRow",
    "packed": "AG-T2I-PackedRow",
    "packed_T": "AG-T2I-PackedCol",
    "step_sparse": "AG-T2I-StepSparse",
    "attention_map": "AG-T2I-AttentionMap",
    "naive": "Naive Reshape",
    "igtd": "IGTD-inspired",
}
LAYOUTS = [
    "step_row",
    "packed",
    "packed_T",
    "step_sparse",
    "attention_map",
    "naive",
    "igtd"
]
SAMPLE_INDEX = 0
INCLUDE_MASK = True   # set True to prepend the attention heatmap

# ---------- load the actual attention masks ----------
def get_attention_heatmap():
    """Load step‑wise soft masks from the same CSV used for image building."""
    step_csv = MASKS_DIR / "tabnet_step_assignment.csv"
    if not step_csv.exists():
        raise FileNotFoundError(f"Missing step assignment file: {step_csv}")
    df = pd.read_csv(step_csv)

    if "step_distribution" in df.columns:
        dists = df["step_distribution"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        mask_df = pd.DataFrame(dists.tolist(), index=df["feature"])
        mask_df.columns = [f"step_{i}" for i in range(mask_df.shape[1])]
    else:
        mask_cols = [c for c in df.columns if c.startswith("step_")]
        mask_df = df[["feature"] + mask_cols].set_index("feature")
        mask_df = mask_df.apply(pd.to_numeric, errors="coerce")

    if "global_importance" in df.columns:
        importance = df.set_index("feature")["global_importance"]
        mask_df = mask_df.loc[importance.sort_values(ascending=False).index]

    return mask_df.astype(np.float64)

# ---------- load preprocessed data ----------
print("Loading preprocessed data...")
X_train = np.load(PROCESSED_DIR / "X_train.npy")
X_test  = np.load(PROCESSED_DIR / "X_test.npy")
y_train = np.load(PROCESSED_DIR / "y_train.npy")
y_test  = np.load(PROCESSED_DIR / "y_test.npy")
feature_names = np.load(PROCESSED_DIR / "feature_names.npy", allow_pickle=True).tolist()

y_train = y_train - y_train.min()
y_test  = y_test - y_test.min()

sample = X_test[SAMPLE_INDEX]
n_features = len(feature_names)

# ---------- AG‑T2I layouts using the actual layout module ----------
def generate_ag_images():
    from tabnet_fs.layouts.unified_layouts import create_layout_from_config

    step_csv = MASKS_DIR / "tabnet_step_assignment.csv"
    step_df = pd.read_csv(step_csv)

    IMPORTANCE_CUTOFF = 0.005
    step_df = step_df[step_df["global_importance"] >= IMPORTANCE_CUTOFF].copy()

    images = {}
    for layout_name in ["step_row", "packed", "packed_T", "step_sparse", "attention_map"]:
        layout = create_layout_from_config(layout_name, step_df)
        _, H, W = layout.compute_image_shape()

        if layout_name == "attention_map":
            weight_matrix = layout.get_weight_matrix()
            img = weight_matrix.copy()
            for j, feat_name in enumerate(step_df["feature"]):
                if feat_name in feature_names:
                    feat_idx = feature_names.index(feat_name)
                    img[:, j] *= sample[feat_idx]
            images[layout_name] = img.astype(np.float32)
            continue

        img = np.zeros((H, W), dtype=np.float32)
        if layout_name in ("packed", "packed_T"):
            for feat_name in step_df["feature"]:
                if feat_name in feature_names:
                    feat_idx = feature_names.index(feat_name)
                    r, c = layout.map_feature_by_name(feat_name)
                    if r < H and c < W:
                        img[r, c] = sample[feat_idx]
        else:
            for step, feats in layout.step_groups.items():
                for local_rank, feat_name in enumerate(feats):
                    if feat_name in feature_names:
                        feat_idx = feature_names.index(feat_name)
                        r, c = layout.map_feature(step, local_rank)
                        if r < H and c < W:
                            img[r, c] = sample[feat_idx]
        images[layout_name] = img
    return images

# ---------- Naive Reshape ----------
def generate_naive_image():
    side = int(np.ceil(np.sqrt(n_features)))
    grid = np.zeros((side, side), dtype=np.float32)
    for i, val in enumerate(sample):
        r, c = i // side, i % side
        grid[r, c] = val
    return grid

# ---------- IGTD‑inspired ----------
def generate_igtd_image():
    from sklearn.manifold import MDS
    import warnings

    corr = np.corrcoef(X_train, rowvar=False)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        mds = MDS(n_components=2, dissimilarity='precomputed',
                  metric=True, n_init=1, max_iter=300, random_state=42)
        coords = mds.fit_transform(dist)

    side = int(np.ceil(np.sqrt(n_features)))
    H, W = side, side
    order = np.lexsort((coords[:, 0], coords[:, 1]))
    img = np.zeros((H, W), dtype=np.float32)
    for idx, feat_idx in enumerate(order):
        r, c = idx // W, idx % W
        img[r, c] = sample[feat_idx]
    return img

# ---------- per‑image normalization ----------
def normalize_img(img):
    vmin, vmax = img.min(), img.max()
    denom = vmax - vmin if vmax > vmin else 1.0
    return (img - vmin) / denom

# ---------- main generation ----------
print("Generating AG‑T2I images...")
ag_images = generate_ag_images()

print("Generating Naive Reshape image...")
naive_img = generate_naive_image()

print("Generating IGTD‑inspired image...")
igtd_img = generate_igtd_image()

images_ordered = {}
for lay in LAYOUTS:
    if lay in ("step_row", "packed", "packed_T", "step_sparse", "attention_map"):
        images_ordered[lay] = ag_images[lay]
    elif lay == "naive":
        images_ordered[lay] = naive_img
    elif lay == "igtd":
        images_ordered[lay] = igtd_img

# ---------- plotting ----------
def make_figure(include_mask, suffix):
    n_panels = len(LAYOUTS) + (1 if include_mask else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.8 * n_panels, 2.8))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    if include_mask:
        ax_mask = axes[panel_idx]
        df_mask = get_attention_heatmap()
        mask_values = df_mask.values.astype(np.float64)
        ax_mask.imshow(mask_values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax_mask.set_xticks(range(df_mask.shape[1]))
        ax_mask.set_xticklabels([f'Step{i}' for i in range(df_mask.shape[1])],
                                rotation=45, ha='right', fontsize=6)
        ax_mask.set_yticks(range(df_mask.shape[0]))
        ax_mask.set_yticklabels(df_mask.index, fontsize=5)
        for i in range(df_mask.shape[0]):
            for j in range(df_mask.shape[1]):
                val = mask_values[i, j]
                ax_mask.text(j, i, f'{val:.2f}', ha='center', va='center',
                             fontsize=4, color='black' if val < 0.7 else 'white')
        ax_mask.set_title("TabNet masks", fontsize=8)
        panel_idx += 1

    for layout in LAYOUTS:
        ax = axes[panel_idx]
        raw_img = images_ordered[layout]
        img = normalize_img(raw_img)          # per‑image 0–1 scaling
        h, w = img.shape

        ax.imshow(img, cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest", aspect="equal")
        ax.add_patch(Rectangle((-0.5, -0.5), w, h,
                               linewidth=1.2, edgecolor='black', facecolor='none'))

        if layout in ("step_row", "step_sparse"):
            metadata_path = PROCESSED_DIR / f"tabnet_layout_{layout}.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                n_steps = metadata["image_shape"]["height"]
                for step in range(1, n_steps):
                    ax.axhline(step - 0.5, color="white", linewidth=1, alpha=0.7)

        ax.set_title(TITLES[layout], fontsize=8, weight="bold")
        ax.axis("off")
        panel_idx += 1

    # very tight spacing, especially between PackedRow and PackedCol
    plt.subplots_adjust(wspace=0.02, hspace=0)
    out_path = OUTPUT_PATH.parent / f"{OUTPUT_PATH.stem}_{suffix}.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure with{'' if include_mask else 'out'} mask to {out_path}")

# Generate both versions
make_figure(include_mask=True, suffix="with_mask")
make_figure(include_mask=False, suffix="layouts_only")

# ---------- standalone heatmap figure ----------
def make_heatmap_only():
    df_mask = get_attention_heatmap()
    mask_values = df_mask.values.T.astype(np.float64)      # transpose: steps × features
    n_rows, n_cols = mask_values.shape
    features = df_mask.index.tolist()
    steps = df_mask.columns.tolist()

    fig, ax = plt.subplots(figsize=(2.8 * n_cols / 3, 2.8 * n_rows / 3))
    im = ax.imshow(mask_values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(steps, fontsize=7)
    for i in range(n_rows):
        for j in range(n_cols):
            val = mask_values[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=5, color='black' if val < 0.7 else 'white')
    ax.set_title("TabNet Step‑wise Attention Masks", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path = OUTPUT_PATH.parent / f"{OUTPUT_PATH.stem}_heatmap_only.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap-only figure to {out_path}")

make_heatmap_only()