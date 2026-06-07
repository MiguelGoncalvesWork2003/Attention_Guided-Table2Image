"""
Metrics computation module for the attention-guided tabular-to-image framework.

This module provides pure, reusable functions for evaluating classification
performance. It enforces a clean separation between metric computation and
presentation logic (e.g., Streamlit UI), ensuring that all evaluation code is
centralised, testable, and reproducible.

Key functions:
  - `compute_classification_metrics`: Computes accuracy, balanced accuracy,
    macro precision/recall/F1, Cohen’s kappa, confusion matrix, and a full
    classification report. For binary problems, it optionally computes ROC-AUC
    and ROC curve data.
  - `format_metrics_for_display`: Converts the raw metrics dictionary into
    human-readable percentage/string values suitable for UI or paper tables.
  - `save_metrics_to_json`: Serialises the metrics dictionary to JSON,
    automatically handling numpy array conversion.

**Role in the Map–Optimize–Learn pipeline:**
  - **Learn:** After training the CNN classifier on the image representations,
    this module evaluates its predictions against the ground truth.
  - The metrics are used to populate Table 1 and the ablation tables in the
    paper, providing standardised, comparable performance figures across
    baselines and layout strategies.
  - The separation of computation from UI ensures that the exact same metrics
    can be generated in both interactive (Streamlit) and script‑based workflows,
    contributing to the full reproducibility of the reported results.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from typing import Dict, Any
import json

def compute_classification_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: np.ndarray = None
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for ROC)
        
    Returns:
        Dictionary containing all metrics
    """
    # Basic metrics
    acc = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_precision = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(
        y_true, y_pred, 
        zero_division=0, 
        output_dict=True
    )
    
    # Build results dictionary
    results = {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_score": macro_f1,
        "cohen_kappa": kappa,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_classes": len(np.unique(y_true)),
        "correct_predictions": int(np.sum(y_pred == y_true)),
        "total_samples": len(y_true)
    }
    
    # Add ROC/AUC for binary classification if probabilities provided
    if y_prob is not None and results["n_classes"] == 2:
        try:
            results["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            results["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist()
            }
        except ValueError:
            pass
    
    return results

def format_metrics_for_display(metrics: Dict[str, Any]) -> Dict[str, str]:
    """
    Format metrics dictionary for human-readable display.
    
    Args:
        metrics: Dictionary from compute_classification_metrics
        
    Returns:
        Dictionary with formatted string values
    """
    formatted = {
        "Accuracy": f"{metrics['accuracy']*100:.2f}%",
        "Balanced Accuracy": f"{metrics['balanced_accuracy']*100:.2f}%",
        "Macro Precision": f"{metrics['precision_macro']*100:.2f}%",
        "Macro Recall": f"{metrics['recall_macro']*100:.2f}%",
        "Macro F1-Score": f"{metrics['f1_score']*100:.2f}%",
        "Cohen's Kappa": f"{metrics['cohen_kappa']:.3f}"
    }
    
    if "roc_auc" in metrics:
        formatted["ROC AUC"] = f"{metrics['roc_auc']:.3f}"
    
    return formatted

def format_classification_report_for_display(report: Dict) -> Dict:
    """
    Format classification report for cleaner display.
    
    Args:
        report: Classification report dict
        
    Returns:
        Formatted report with percentage strings
    """
    formatted = {}
    for class_name, class_metrics in report.items():
        if isinstance(class_metrics, dict):
            formatted[class_name] = {}
            for metric_name, value in class_metrics.items():
                if metric_name in ['precision', 'recall', 'f1-score']:
                    formatted[class_name][metric_name] = f"{value*100:.2f}%"
                else:
                    formatted[class_name][metric_name] = value
        else:
            formatted[class_name] = class_metrics
    
    return formatted

def save_metrics_to_json(metrics: Dict[str, Any], filepath: str) -> None:
    """
    Save metrics dictionary to JSON file.
    
    Args:
        metrics: Metrics dictionary
        filepath: Path to save JSON
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = metrics.copy()
    if "confusion_matrix" in serializable_metrics:
        if isinstance(serializable_metrics["confusion_matrix"], np.ndarray):
            serializable_metrics["confusion_matrix"] = serializable_metrics["confusion_matrix"].tolist()
    
    with open(filepath, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)