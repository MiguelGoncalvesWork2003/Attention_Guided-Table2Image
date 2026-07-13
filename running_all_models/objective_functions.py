"""objective_functions.py – Optuna objective for one model on one dataset."""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from search_spaces import SEARCH_FUNCTIONS
from models_factory import get_model_from_params


def objective(trial, model_name, X_train, y_train, n_features, n_classes):
    """
    Args:
        trial: Optuna trial object
        model_name: str, key in SEARCH_FUNCTIONS
        X_train: np.ndarray or pd.DataFrame, unscaled (but already imputed if needed)
        y_train: np.ndarray, integer labels
        n_features: int
        n_classes: int

    Returns:
        mean F1 macro across 3 folds
    """
    # Get hyperparameters for this trial
    params = SEARCH_FUNCTIONS[model_name](trial)

    # Instantiate model with these parameters
    model = get_model_from_params(model_name, n_features, n_classes, params)

    # 3‑fold inner CV
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []

    for train_idx, val_idx in inner_cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Fit (some models need special handling, but get_model_from_params returns a
        # classifier with a standard scikit‑learn interface)
        try:
            model.fit(X_tr, y_tr)
        except Exception:
            # If the model fails (e.g., TabNet with wrong dimensions), return a bad score
            return 0.0

        # Predict
        if hasattr(model, "predict_proba"):
            y_pred = model.predict(X_val)
        else:
            y_pred = model.predict(X_val)

        f1 = f1_score(y_val, y_pred, average="macro")
        f1_scores.append(f1)

    return np.mean(f1_scores)