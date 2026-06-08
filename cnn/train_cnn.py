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
  - Saves the best model checkpoint (based on validation accuracy) together
    with a JSON configuration file that records all hyperparameters and
    the final image shape.
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
from cnn_model import TabNetCNN

# Add project root to path for shared metrics import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from running_all_models.metrics import compute_extended_metrics

BASE = Path(__file__).resolve().parent.parent

DATASET = os.environ.get("DATASET", "BreastCancer")
LAYOUT = os.environ.get("MOL_LAYOUT", "step_row")
SEED = int(os.environ.get("SEED", 42))

LEARNING_RATE = float(os.environ.get("CNN_LEARNING_RATE", 1e-3))
OPTIMIZER_NAME = os.environ.get("CNN_OPTIMIZER", "adam").lower()
DROPOUT = float(os.environ.get("CNN_DROPOUT", 0.3))
EPOCHS = int(os.environ.get("CNN_EPOCHS", 50))
BATCH_SIZE = int(os.environ.get("CNN_BATCH_SIZE", 32))

PROCESSED_DIR = BASE / "data/processed" / DATASET
OUTPUT_DIR = BASE / "cnn/cnn_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / f"best_model_{DATASET}_{LAYOUT}_seed{SEED}.pth"
CONFIG_PATH = OUTPUT_DIR / f"cnn_config_{DATASET}_{LAYOUT}_seed{SEED}.json"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

X_train = np.load(PROCESSED_DIR / "X_train_img.npy")
y_train = np.load(PROCESSED_DIR / "y_train.npy")

if X_train.ndim != 4:
    raise ValueError(f"Expected [B,C,H,W], got {X_train.shape}")

# Normalise labels to 0..C-1
y_train = y_train - y_train.min()
n_classes = len(np.unique(y_train))

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
X_val_path = PROCESSED_DIR / "X_val_img.npy"
y_val_path = PROCESSED_DIR / "y_val.npy"
if X_val_path.exists() and y_val_path.exists():
    X_val = np.load(X_val_path)
    y_val = np.load(y_val_path)
    y_val = y_val - y_val.min()
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

model = TabNetCNN(
    n_classes=n_classes,
    input_channels=image_shape[0],
    image_height=image_shape[1],
    image_width=image_shape[2],
    dropout=DROPOUT
)

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


best_val_acc = 0.0
best_epoch = 0

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

    # Validation step
    if val_loader is not None:
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                preds = out.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)
        val_acc = val_correct / val_total
        scheduler.step(1 - val_acc)  # Reduce LR when validation accuracy stops improving
    else:
        val_acc = train_acc  # fallback

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f}")

    # Save best model based on validation accuracy (or training if no validation)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save({
            "model_state_dict": model.state_dict(),
            "dataset": DATASET,
            "layout": LAYOUT,
            "seed": SEED,
            "image_shape": image_shape,
            "n_classes": n_classes,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "hyperparameters": {
                "learning_rate": LEARNING_RATE,
                "optimizer": OPTIMIZER_NAME,
                "dropout": DROPOUT,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS
            }
        }, MODEL_PATH)
        print(f"  -> New best model saved (val_acc={best_val_acc:.4f})")

# ------------------------------------------------------------
# Evaluate on training set using the BEST model (not the final state)
# ------------------------------------------------------------
# Reload the best checkpoint
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

# Use the shared metrics function
train_metrics_dict = compute_extended_metrics(all_labels, all_preds)
# Prefix keys with "train_"
train_metrics = {f"train_{k}": v for k, v in train_metrics_dict.items()}

train_results_path = PROCESSED_DIR / f"cnn_training_results_{LAYOUT}_seed{SEED}.json"
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
    "best_val_acc": best_val_acc,
    "best_epoch": best_epoch,
    "hyperparameters": {
        "learning_rate": LEARNING_RATE,
        "optimizer": OPTIMIZER_NAME,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS
    }
}

with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)

print(f"\n✅ Training completed. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
print(f"Model saved to: {MODEL_PATH}")
print(f"Config saved to: {CONFIG_PATH}")