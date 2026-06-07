"""
Evaluation utilities for benchmark experiments.
Provides extended metrics, ROC data, and misclassified sample capture.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

def compute_extended_metrics(y_true, y_pred, y_proba=None):
    """
    Compute a rich set of classification metrics.
    ROC‑AUC is included only if probabilities are provided.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_proba is not None:
        # multiclass ROC‑AUC: one‑vs‑rest
        try:
            metrics["roc_auc"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["roc_auc"] = np.nan
    return metrics


def get_wrong_cases(y_true, y_pred, indices=None, true_labels=None, pred_labels=None):
    """
    Return a DataFrame with misclassified samples.

    Parameters
    ----------
    y_true : array-like of encoded labels
    y_pred : array-like of encoded labels
    indices : array-like, optional
        Original sample indices (e.g., row numbers in the full dataset).
    true_labels : array-like, optional
        Original (decoded) true class names.
    pred_labels : array-like, optional
        Original (decoded) predicted class names.

    Returns
    -------
    pd.DataFrame
        Columns: [index, true_label, pred_label, true_label_decoded, pred_label_decoded]
    """
    wrong_mask = (y_true != y_pred)
    if not np.any(wrong_mask):
        return pd.DataFrame(columns=["index", "true_label", "pred_label", "true_label_decoded", "pred_label_decoded"])
    df = pd.DataFrame()
    if indices is not None:
        df["index"] = np.array(indices)[wrong_mask]
    else:
        df["index"] = np.where(wrong_mask)[0]
    df["true_label"] = np.array(y_true)[wrong_mask]
    df["pred_label"] = np.array(y_pred)[wrong_mask]
    if true_labels is not None:
        df["true_label_decoded"] = np.array(true_labels)[wrong_mask]
    if pred_labels is not None:
        df["pred_label_decoded"] = np.array(pred_labels)[wrong_mask]
    return df.reset_index(drop=True)