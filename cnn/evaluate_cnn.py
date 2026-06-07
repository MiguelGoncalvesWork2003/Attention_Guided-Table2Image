# evaluate_cnn.py
"""
CNN evaluation script — the single authority for computing and saving
classification metrics on the test set.

This script loads a trained `TabNetCNN` model (from a checkpoint produced
by `train_cnn.py`) and evaluates it on the held‑out test images. It
computes a comprehensive set of metrics using the centralised
`running_all_models.metrics` module and saves all results in a structured
format for both programmatic consumption and the Streamlit UI.

Workflow:
  1. Load the CNN configuration JSON (image shape, hyperparameters) and
     the model checkpoint.
  2. Load `X_test_img.npy` and `y_test.npy`; validate tensor dimensions
     and normalise labels to 0‑based indexing.
  3. Perform batched inference with `torch.no_grad()` to obtain predicted
     classes and class probabilities.
  4. Call `compute_extended_metrics` to produce accuracy, balanced
     accuracy, macro/weighted precision/recall/F1, Cohen’s kappa, confusion
     matrix, classification report, and (for binary problems) ROC‑AUC.
  5. Save the full metrics dictionary as JSON, and also export the raw
     predictions, probabilities, confusion matrix, and classification
     report as separate files for downstream analysis.
  6. Print a summary table for quick inspection.

**Role in the Map–Optimize–Learn pipeline:**
  - Constitutes the **evaluation** sub‑stage of **Learn**, delivering the
    performance numbers that populate Table 1 and the ablation studies in
    the paper.
  - The output files are the authoritative source of all reported CNN
    metrics, guaranteeing that the same numbers can be reproduced from the
    saved artefacts without re‑running the entire pipeline.
  - By separating evaluation from training, the script reinforces the
    pipeline’s modularity and reproducibility: the model is never modified
    during evaluation, and all metrics are computed in a standardised,
    library‑based manner.
"""

import numpy as np
import torch
import json
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

# Add project root to import shared metrics
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from running_all_models.metrics import compute_extended_metrics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cnn.cnn_model import TabNetCNN
from evaluation.metrics import save_metrics_to_json   # still used for saving JSON

def main():
    """Main evaluation routine."""
    
    # -------------------------
    # CONFIGURATION
    # -------------------------
    BASE = Path(__file__).resolve().parent.parent
    
    DATASET = os.environ.get("DATASET", "BreastCancer")
    LAYOUT = os.environ.get("MOL_LAYOUT", "step_row")
    SEED = int(os.environ.get("SEED", 42))
    
    MODEL_DIR = BASE / "cnn" / "cnn_models"
    PROCESSED_DIR = BASE / "data" / "processed" / DATASET
    
    # Model paths
    MODEL_PATH = MODEL_DIR / f"best_model_{DATASET}_{LAYOUT}_seed{SEED}.pth"
    CONFIG_PATH = MODEL_DIR / f"cnn_config_{DATASET}_{LAYOUT}_seed{SEED}.json"
    
    # -------------------------
    # VALIDATE INPUTS
    # -------------------------
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"CNN config not found at {CONFIG_PATH}. "
            f"Run train_cnn.py first."
        )
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CNN model not found at {MODEL_PATH}. "
            f"Run train_cnn.py first."
        )
    
    # Load config
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    
    # -------------------------
    # LOAD TEST DATA
    # -------------------------
    X_test_path = PROCESSED_DIR / "X_test_img.npy"
    y_test_path = PROCESSED_DIR / "y_test.npy"
    
    if not X_test_path.exists():
        raise FileNotFoundError(f"Test images not found at {X_test_path}")
    if not y_test_path.exists():
        raise FileNotFoundError(f"Test labels not found at {y_test_path}")
    
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    # Validate dimensions
    if X_test.ndim != 4:
        raise ValueError(
            f"Expected test images shape [B, C, H, W], got {X_test.shape}"
        )
    
    # Normalize labels to 0-based indexing
    y_test = y_test - y_test.min()
    
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Image shape: {X_test.shape[1:]} (C, H, W)")
    print(f"Classes: {len(np.unique(y_test))}")
    
    # -------------------------
    # LOAD MODEL
    # -------------------------
    print(f"\nLoading model from {MODEL_PATH}...")
    
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    
    # Extract model parameters
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        n_classes = checkpoint.get("n_classes", len(np.unique(y_test)))
        model_state = checkpoint["model_state_dict"]
    else:
        # Legacy format - direct state dict
        n_classes = len(np.unique(y_test))
        model_state = checkpoint
    
    # Instantiate model
    model = TabNetCNN(
        n_classes=n_classes,
        input_channels=config["image_shape"][0],
        image_height=config["image_shape"][1],
        image_width=config["image_shape"][2]
    )
    
    model.load_state_dict(model_state)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Classes: {n_classes}")
    
    # -------------------------
    # RUN INFERENCE
    # -------------------------
    print("\nRunning inference on test set...")
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            outputs = model(xb)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    print(f"Inference complete: {len(y_pred)} predictions made")
    
    # -------------------------
    # COMPUTE METRICS
    # -------------------------
    print("\nComputing evaluation metrics...")
    
    # Use the same comprehensive metric function as the baselines
    metrics = compute_extended_metrics(y_test, y_pred, y_prob)
    
    # Add additional metrics used by the UI / down‑stream analysis
    metrics["cohen_kappa"] = cohen_kappa_score(y_test, y_pred)
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["classification_report"] = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    
    # Add metadata
    metrics["dataset"] = DATASET
    metrics["layout"] = LAYOUT
    metrics["seed"] = SEED
    metrics["model_path"] = str(MODEL_PATH)
    metrics["image_shape"] = config["image_shape"]
    
    # -------------------------
    # SAVE RESULTS
    # -------------------------
    print("\nSaving evaluation results...")
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save comprehensive JSON results (for UI)
    results_json_path = PROCESSED_DIR / f"cnn_evaluation_results_{LAYOUT}.json"
    save_metrics_to_json(metrics, str(results_json_path))
    print(f"✓ Results saved to {results_json_path}")
    
    # Save predictions
    pred_path = PROCESSED_DIR / f"y_test_pred_{LAYOUT}.npy"
    np.save(pred_path, y_pred)
    print(f"✓ Predictions saved to {pred_path}")
    
    # Save probabilities
    prob_path = PROCESSED_DIR / f"y_test_prob_{LAYOUT}.npy"
    np.save(prob_path, y_prob)
    print(f"✓ Probabilities saved to {prob_path}")
    
    # Save confusion matrix separately
    cm_path = PROCESSED_DIR / f"confusion_matrix_{LAYOUT}.npy"
    np.save(cm_path, np.array(metrics["confusion_matrix"]))
    print(f"✓ Confusion matrix saved to {cm_path}")
    
    # Save classification report separately
    report_path = PROCESSED_DIR / f"classification_report_{LAYOUT}.json"
    with open(report_path, 'w') as f:
        json.dump(metrics["classification_report"], f, indent=2)
    print(f"✓ Classification report saved to {report_path}")
    
    # Save combined predictions/labels for analysis
    combined = {
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist()
    }
    if y_prob is not None:
        combined["y_prob"] = y_prob.tolist()
    
    combined_path = PROCESSED_DIR / f"predictions_{LAYOUT}.json"
    with open(combined_path, 'w') as f:
        json.dump(combined, f)
    print(f"✓ Combined predictions saved to {combined_path}")
    
    # -------------------------
    # PRINT SUMMARY
    # -------------------------
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"Dataset:    {DATASET}")
    print(f"Layout:     {LAYOUT}")
    print(f"Seed:       {SEED}")
    print(f"Accuracy:   {metrics['accuracy']*100:.2f}%")
    print(f"Balanced:   {metrics['balanced_accuracy']*100:.2f}%")
    print(f"F1 Macro:   {metrics['f1_macro']*100:.2f}%")
    print(f"F1 Weighted:{metrics['f1_weighted']*100:.2f}%")
    print(f"Kappa:      {metrics['cohen_kappa']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()