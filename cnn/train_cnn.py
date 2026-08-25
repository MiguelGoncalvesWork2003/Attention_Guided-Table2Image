# train_cnn.py
"""
CNN training script for attention‑guided tabular‑to‑image representations.

This script implements the **CNN learning stage** of the attention-guided
tabular-to-image framework: it trains a convolutional neural network on the
image representations previously produced by a deterministic, attention‑guided
layout. The training process is fully decoupled from both the TabNet
feature‑attention model and the layout builder.

Key properties:
  - Loads `X_train_img.npy` (and optionally `X_val_img.npy`) from the
    processed data directory, together with the encoded labels.
  - Instantiates a `TabNetCNN` with the exact spatial dimensions of the
    generated images, ensuring the architecture matches the layout geometry
    without any resizing or interpolation.
  - Uses a fixed hyperparameter set (learning rate, optimizer, dropout,
    batch size, epochs) read from environment variables, with a
    `ReduceLROnPlateau` scheduler for stable convergence.
  - Selects the best checkpoint by macro-averaged ROC-AUC on the validation
    split, and stops early once that score has not improved for `CNN_PATIENCE`
    epochs (Section 5.5 of the thesis).
  - After training, computes extended training‑set metrics and saves them
    for later comparison with test metrics.

The script expects the image arrays to be 4D `(N, C, H, W)`, where `C=1`
(single‑channel grayscale images). This format is the direct output of the
layout projection, ensuring no information is lost or distorted.

**Role in the Map–Optimize–Learn pipeline:**
  - After the **Map** stage (preprocessing → image generation) and the
    **Optimize** stage (TabNet training → layout derivation), this script
    performs the final supervised learning step using a CNN.
  - No feedback is ever passed from the CNN back to the earlier stages,
    preserving the controlled experimental protocol described in Section 4.
  - The saved configuration and model checkpoint enable fully reproducible
    evaluation, which is carried out by `evaluate_cnn.py`.
"""

import numpy as np
import torch
import random
import json
import os
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.metrics import roc_auc_score
from cnn_model import TabNetCNN
from cnn_architectures import build_model, count_parameters

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from running_all_models.metrics import compute_extended_metrics

BASE = Path(__file__).resolve().parent.parent

DATASET = os.environ.get("DATASET", "BreastCancer")
LAYOUT = os.environ.get("MOL_LAYOUT", "step_row")
SEED = int(os.environ.get("SEED", 42))

# ---- ISOLATION: use OUTPUT_DIR for all file I/O ----
_root = Path(os.environ.get("OUTPUT_DIR", str(BASE / "data" / "processed" / DATASET)))
BASE_LAYOUT = os.environ.get("BASE_LAYOUT", "step_row").strip()
PERMUTATION_SEED = int(os.environ.get("PERMUTATION_SEED", 0))
AM_VARIANT = os.environ.get("AM_VARIANT", "full").strip()
if LAYOUT == "shuffled":
    _tag = f"shuffled-{BASE_LAYOUT}-p{PERMUTATION_SEED}_seed{SEED}"
elif LAYOUT == "attention_map" and AM_VARIANT != "full":
    _tag = f"attention_map-{AM_VARIANT}_seed{SEED}"
else:
    _tag = f"{LAYOUT}_seed{SEED}"

# E1 (layout transfer, Table 6.10): images are generated once per
# (dataset, layout, seed) and shared read-only across every architecture --
# that reuse is the property being tested, not an implementation detail.
# IMAGE_DIR is that shared directory; TASK_OUTPUT_DIR is per-architecture, so
# results from different architectures never collide. CNN_ARCH defaults to
# "tabnetcnn", so any run that doesn't set it behaves exactly as before.
CNN_ARCH = os.environ.get("CNN_ARCH", "tabnetcnn").strip().lower()
IMAGE_DIR = _root / _tag
if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Image directory {IMAGE_DIR} not found. Run tabnet_image_builder.py "
        f"for DATASET={DATASET} MOL_LAYOUT={LAYOUT} SEED={SEED} first."
    )
TASK_OUTPUT_DIR = IMAGE_DIR / f"arch_{CNN_ARCH}"
TASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = BASE / "cnn" / "cnn_models"   # keep this for backward compat if needed
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LEARNING_RATE = float(os.environ.get("CNN_LEARNING_RATE", 1e-3))
OPTIMIZER_NAME = os.environ.get("CNN_OPTIMIZER", "adam").lower()
DROPOUT = float(os.environ.get("CNN_DROPOUT", 0.3))
EPOCHS = int(os.environ.get("CNN_EPOCHS", 50))
BATCH_SIZE = int(os.environ.get("CNN_BATCH_SIZE", 32))
PATIENCE = int(os.environ.get("CNN_PATIENCE", 20))          # (b) early stopping

PROCESSED_DIR = BASE / "data/processed" / DATASET
OUTPUT_DIR = BASE / "cnn/cnn_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = TASK_OUTPUT_DIR / f"best_model_{DATASET}_{LAYOUT}_seed{SEED}.pth"
CONFIG_PATH = TASK_OUTPUT_DIR / f"cnn_config_{DATASET}_{LAYOUT}_seed{SEED}.json"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # (c) deterministic CUDA, as claimed in Section 5.5
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

# Load images from the shared, per-(dataset,layout,seed) directory -- read
# by every architecture, written by none of them.
X_train = np.load(IMAGE_DIR / "X_train_img.npy")

# Labels: after the image builder split, y_train.npy in PROCESSED_DIR is the reduced training set
y_train = np.load(IMAGE_DIR / "y_train.npy")

if X_train.ndim != 4:
    raise ValueError(f"Expected [B,C,H,W], got {X_train.shape}")

# Labels are already 0-based from the preprocessing LabelEncoder.
# Do NOT shift by the local minimum: if a class is absent from this split,
# the shift silently misaligns labels with the training encoding.
if y_train.min() < 0:
    raise ValueError(f"Negative labels in training set: min={y_train.min()}")
n_classes = int(os.environ.get("N_CLASSES", 0)) or int(y_train.max()) + 1

image_shape = X_train.shape[1:]  # (C, H, W)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=torch.cuda.is_available()
)

# Validation data (if exists)
val_loader = None
X_val_path = IMAGE_DIR / "X_val_img.npy"
y_val_path = IMAGE_DIR / "y_val.npy"
if X_val_path.exists() and y_val_path.exists():
    X_val = np.load(X_val_path)
    y_val = np.load(y_val_path)
    # Same reasoning as for y_train: no local re-basing.
    if y_val.min() < 0:
        raise ValueError(f"Negative labels in validation set: min={y_val.min()}")
    if y_val.max() >= n_classes:
        raise ValueError(
            f"Validation label {y_val.max()} exceeds n_classes={n_classes}; "
            f"set N_CLASSES to the dataset-wide class count."
        )
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        pin_memory=torch.cuda.is_available()
    )
    print(f"Validation data loaded: {len(X_val)} samples")
else:
    print("No validation data found – using only training accuracy for model selection (may overfit)")

model = build_model(
    CNN_ARCH,
    n_classes=n_classes,
    input_channels=image_shape[0],
    image_height=image_shape[1],
    image_width=image_shape[2],
    dropout=DROPOUT,
)
print(f"Architecture: {CNN_ARCH} | trainable parameters: {count_parameters(model):,}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if OPTIMIZER_NAME == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
elif OPTIMIZER_NAME == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
elif OPTIMIZER_NAME == "rmsprop":
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
else:
    print(f"Unknown optimizer '{OPTIMIZER_NAME}', falling back to Adam")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5
)

# (a) selection by macro-AUC, not accuracy
best_val_score = -np.inf
best_val_acc = 0.0          # kept for logging and backward compatibility
best_epoch = 0
stale = 0
stopped_early = False

for epoch in range(EPOCHS):
    model.train()
    train_correct = 0
    train_total = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        preds = out.argmax(dim=1)
        train_correct += (preds == yb).sum().item()
        train_total += yb.size(0)

    train_acc = train_correct / train_total

    # ---- Validation step: macro-AUC drives selection and the scheduler ----
    if val_loader is not None:
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                vp.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
                vt.append(yb.numpy())
        vp, vt = np.concatenate(vp), np.concatenate(vt)

        try:
            val_score = (
                roc_auc_score(vt, vp[:, 1]) if n_classes == 2
                else roc_auc_score(vt, vp, multi_class="ovr", average="macro",
                                   labels=list(range(n_classes)))
            )
        except ValueError:
            # A class missing from the validation split makes AUC undefined.
            val_score = float((vp.argmax(1) == vt).mean())

        val_acc = float((vp.argmax(1) == vt).mean())
        scheduler.step(1.0 - val_score)
    else:
        val_score = val_acc = train_acc

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | train_acc: {train_acc:.4f} "
          f"| val_auc: {val_score:.4f} | val_acc: {val_acc:.4f}")

    # ---- Best checkpoint + early stopping ----
    if val_score > best_val_score:
        best_val_score = val_score
        best_val_acc = val_acc
        best_epoch = epoch + 1
        stale = 0
        torch.save({
            "model_state_dict": model.state_dict(),
            "dataset": DATASET,
            "layout": LAYOUT,
            "seed": SEED,
            "image_shape": image_shape,
            "n_classes": n_classes,
            "architecture": CNN_ARCH,
            "n_parameters": count_parameters(model),
            "best_val_score": best_val_score,
            "best_val_metric": "macro_auc",
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "hyperparameters": {
                "learning_rate": LEARNING_RATE,
                "optimizer": OPTIMIZER_NAME,
                "dropout": DROPOUT,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "patience": PATIENCE
            }
        }, MODEL_PATH)
        print(f"  -> New best model saved (val macro-AUC={best_val_score:.4f})")
    else:
        stale += 1
        if stale >= PATIENCE:
            stopped_early = True
            print(f"Early stopping at epoch {epoch + 1} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

# Guard: if every epoch produced a NaN/degenerate score, nothing was written.
if not MODEL_PATH.exists():
    raise RuntimeError(
        f"No checkpoint was saved for {DATASET}/{LAYOUT}/seed{SEED}: "
        f"validation score never improved on {best_val_score}."
    )

# ------------------------------------------------------------
# Evaluate on training set using the BEST model
# ------------------------------------------------------------
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    eval_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    for xb, yb in eval_loader:
        xb = xb.to(device)
        out = model(xb)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.cpu().numpy())

train_metrics_dict = compute_extended_metrics(all_labels, all_preds)
train_metrics = {f"train_{k}": v for k, v in train_metrics_dict.items()}

train_results_path = TASK_OUTPUT_DIR / f"cnn_training_results_{LAYOUT}_seed{SEED}.json"
with open(train_results_path, "w") as f:
    json.dump(train_metrics, f, indent=2)

print(f"Train metrics saved to {train_results_path}")

# ------------------------------------------------------------
# Save configuration
# ------------------------------------------------------------
config = {
    "dataset": DATASET,
    "layout": LAYOUT,
    "seed": SEED,
    "image_shape": list(image_shape),
    "n_classes": n_classes,
    "architecture": CNN_ARCH,
    "n_parameters": count_parameters(model),
    "best_val_score": best_val_score,
    "best_val_metric": "macro_auc",
    "best_val_acc": best_val_acc,
    "best_epoch": best_epoch,
    "epochs_run": epoch + 1,
    "stopped_early": stopped_early,
    "hyperparameters": {
        "learning_rate": LEARNING_RATE,
        "optimizer": OPTIMIZER_NAME,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE
    }
}

with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)

print(f"\n✅ Training completed. Best validation macro-AUC: {best_val_score:.4f} "
      f"at epoch {best_epoch} (ran {epoch + 1}/{EPOCHS} epochs)")
print(f"Model saved to: {MODEL_PATH}")
print(f"Config saved to: {CONFIG_PATH}")