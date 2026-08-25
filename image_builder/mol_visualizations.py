# mol_visualizations.py
"""
Visual diagnostics for attention‑guided tabular‑to‑image representations.
[... keep the existing docstring ...]
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

BASE = Path(__file__).resolve().parents[1]

DATASET = os.environ.get("DATASET", "Cancer")
MOL_LAYOUT = os.environ.get("MOL_LAYOUT", "step_row")
SEED = int(os.environ.get("SEED", 42))

PROCESSED_DIR = BASE / "data" / "processed" / DATASET

# ---- ISOLATED INPUT DIRECTORY (same as image builder output) ----
INPUT_DIR = PROCESSED_DIR / f"{MOL_LAYOUT}_seed{SEED}"

# ---- ISOLATED OUTPUT DIRECTORY for visualizations ----
OUTPUT_DIR = (
    BASE
    / "experiments"
    / "mol_visualizations"
    / DATASET
    / f"{MOL_LAYOUT}_seed{SEED}"
)

for subdir in ["grids", "instances", "step_analysis"]:
    (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

print(f"Dataset: {DATASET}")
print(f"Layout: {MOL_LAYOUT}")
print(f"Seed: {SEED}")
print(f"Input directory: {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# ---------- Load data from the isolated input directory ----------
print("\nLoading data...")

required = {
    "X_train_img.npy": INPUT_DIR / "X_train_img.npy",
    "X_test_img.npy":  INPUT_DIR / "X_test_img.npy",
    "y_train.npy":     INPUT_DIR / "y_train.npy",
    "y_test.npy":      INPUT_DIR / "y_test.npy",
}

missing = [name for name, path in required.items() if not path.exists()]
if missing:
    print("=" * 60)
    print("FATAL: Required input files are missing.")
    print("The image builder (tabnet_image_builder.py) probably did not run")
    print("successfully or was skipped. Please re‑run the pipeline.")
    print("Missing files:")
    for m in missing:
        print(f"  - {required[m]}")
    print("=" * 60)
    sys.exit(1)

X_train_img = np.load(required["X_train_img.npy"])
X_test_img  = np.load(required["X_test_img.npy"])
y_train     = np.load(required["y_train.npy"])
y_test      = np.load(required["y_test.npy"])

# Labels are already 0-based from the preprocessing LabelEncoder. Do NOT
# shift by the local minimum: if a class is absent from this split, the
# shift silently relabels every other class, mislabelling the per-class
# grids and average-image figures below.
if y_train.min() < 0 or y_test.min() < 0:
    raise ValueError(
        f"Negative labels found (train min={y_train.min()}, "
        f"test min={y_test.min()}); labels should already be 0-based."
    )

# Layout metadata now also lives in the isolated input directory
layout_path = INPUT_DIR / f"tabnet_layout_{MOL_LAYOUT}.json"

if layout_path.exists():
    with open(layout_path, "r", encoding="utf-8") as f:
        spatial_layout = json.load(f)
    print(f"Loaded layout metadata: {layout_path.name}")
else:
    spatial_layout = None
    print("Warning: Layout metadata not found")

print(f"Train images: {X_train_img.shape}")
print(f"Test images: {X_test_img.shape}")
print(f"Classes: {np.unique(y_train)}")

# ---------- Rest of the script is unchanged ----------
print("\nComputing global normalization...")

X_train_flat = X_train_img[:, 0] if X_train_img.ndim == 4 else X_train_img
X_test_flat  = X_test_img[:, 0] if X_test_img.ndim == 4 else X_test_img

global_min = min(X_train_flat.min(), X_test_flat.min())
global_max = max(X_train_flat.max(), X_test_flat.max())

print(f"Global min: {global_min:.6f}")
print(f"Global max: {global_max:.6f}")


def process_for_visualization(img: np.ndarray) -> np.ndarray:
    """Normalize image into [0,1] for stable visualization."""
    if img.ndim == 3:
        img = img[0]
    img = np.nan_to_num(img, nan=global_min, posinf=global_max, neginf=global_min)
    denominator = global_max - global_min
    if denominator <= 0:
        return np.zeros_like(img)
    img = (img - global_min) / denominator
    return np.clip(img, 0.0, 1.0)


def save_images_grid(
    X_img: np.ndarray,
    y_labels: np.ndarray,
    prefix: str = "train",
    n_grid: int = 9
) -> None:
    classes = np.unique(y_labels)
    for cls in classes:
        indices = np.where(y_labels == cls)[0][:n_grid]
        if len(indices) == 0:
            continue
        n_cols = 3
        n_rows = int(np.ceil(len(indices) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_1d(axes).flatten()
        im = None
        for i, ax in enumerate(axes):
            if i >= len(indices):
                ax.axis("off")
                continue
            img = process_for_visualization(X_img[indices[i]])
            height, width = img.shape
            im = ax.imshow(img, cmap="gray", vmin=0, vmax=1,
                           interpolation="nearest", aspect="equal")
            ax.add_patch(Rectangle((-0.5, -0.5), width, height,
                                   linewidth=1.5, edgecolor="black", facecolor="none"))
            if spatial_layout:
                n_steps = spatial_layout["image_shape"]["height"]
                for step in range(1, n_steps):
                    ax.axhline(step - 0.5, color="white", linewidth=1, alpha=0.7)
            ax.set_title(f"Sample {indices[i]}")
            ax.axis("off")
        if im is not None:
            fig.colorbar(im, ax=axes.tolist(), shrink=0.7)
        fig.suptitle(f"MOL Images - Class {cls} ({prefix})", fontsize=14, weight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "grids" / f"{prefix}_class_{cls}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved class {cls} grid")


def save_single_mol_image(
    img: np.ndarray,
    idx: int,
    label=None,
    prefix: str = "test"
) -> None:
    img = process_for_visualization(img)
    height, width = img.shape
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(img, cmap="gray", vmin=0, vmax=1,
                   interpolation="nearest", aspect="equal")
    ax.add_patch(Rectangle((-0.5, -0.5), width, height,
                           linewidth=2, edgecolor="black", facecolor="none"))
    if spatial_layout:
        n_steps = spatial_layout["image_shape"]["height"]
        for step in range(1, n_steps):
            ax.axhline(step - 0.5, color="white", linewidth=1.2, alpha=0.8)
    title = f"MOL Image {idx}"
    if label is not None:
        title += f" | Class {label}"
    ax.set_title(title, fontsize=12, weight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "instances" / f"{prefix}_instance_{idx}.png",
                dpi=200, bbox_inches="tight")
    plt.close()


def save_step_analysis(X_img: np.ndarray, prefix: str = "train") -> None:
    if spatial_layout is None:
        print("  Skipping step analysis (no layout metadata)")
        return
    step_groups = spatial_layout.get("step_groups", {})
    X_flat = X_img[:, 0] if X_img.ndim == 4 else X_img
    n_steps = X_flat.shape[1]
    avg_activation = np.mean(X_flat, axis=(0, 2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(range(n_steps), avg_activation, color="gray", edgecolor="black")
    axes[0].set_title("Average Activation per Step")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Activation")
    step_counts = [len(step_groups.get(str(step), [])) for step in range(n_steps)]
    axes[1].bar(range(n_steps), step_counts, color="lightgray", edgecolor="black")
    axes[1].set_title("Features per Step")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Feature Count")
    plt.suptitle(f"Step Analysis ({prefix})", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "step_analysis" / f"{prefix}_step_analysis.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved step analysis ({prefix})")


def save_average_images_per_class(
    X_img: np.ndarray,
    y_labels: np.ndarray,
    prefix: str = "train"
) -> None:
    classes = np.unique(y_labels)
    n_cols = min(3, len(classes))
    n_rows = int(np.ceil(len(classes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for i, cls in enumerate(classes):
        ax = axes[i]
        indices = np.where(y_labels == cls)[0]
        class_images = X_img[indices]
        if class_images.ndim == 4:
            class_images = class_images[:, 0]
        avg_img = np.mean(class_images, axis=0)
        avg_img = process_for_visualization(avg_img)
        im = ax.imshow(avg_img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if spatial_layout:
            n_steps = spatial_layout["image_shape"]["height"]
            for step in range(1, n_steps):
                ax.axhline(step - 0.5, color="white", linewidth=1)
        ax.set_title(f"Class {cls} (n={len(indices)})")
        ax.axis("off")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"Average MOL Images per Class ({prefix})", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix}_average_per_class.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved average images ({prefix})")


print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

save_images_grid(X_train_img, y_train, prefix="train")
save_images_grid(X_test_img, y_test, prefix="test")

for i in range(min(5, len(X_test_img))):
    save_single_mol_image(X_test_img[i], idx=i, label=y_test[i], prefix="test")

save_step_analysis(X_train_img, prefix="train")
save_step_analysis(X_test_img, prefix="test")

save_average_images_per_class(X_train_img, y_train, prefix="train")
save_average_images_per_class(X_test_img, y_test, prefix="test")

print("\nVisualization complete.")
print(f"Saved to: {OUTPUT_DIR}")