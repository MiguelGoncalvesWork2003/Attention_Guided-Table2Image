"""
Hyperparameter search for all models (baseline + AG‑T2I).
- Tree models & MLP: RandomizedSearchCV with publication‑informed grids.
- TabNet & FT-Transformer: manual random search (sklearn‑compatible wrapper).
- AG‑T2I: per‑layout search via your existing api.py (random method by default).
Results saved to running_all_models/hyperparameter_results/<dataset>_best_params.json
"""

import sys
import json
import time
import copy
import itertools
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from joblib import Parallel, delayed

from api import SimplePipelineAPI
from running_all_models.models_factory import get_models
from running_all_models.utils import set_seed

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
SEARCH_CV = 3          # inner CV for baseline models (tree/MLP)
N_ITER = 20            # random parameter combinations for each baseline model
SEED = 42
VAL_SIZE = 0.2         # validation split for neural models (TabNet, FT‑Transformer)
RESULTS_DIR = PROJECT_ROOT / "running_all_models" / "hyperparameter_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# AG‑T2I search settings
AGT2I_TRIALS_PER_LAYOUT = 20   # number of trials per layout
AGT2I_METHOD = "random"        # "random" or "bayesian"

# ------------------------------------------------------------
# Publication‑informed parameter grids
# ------------------------------------------------------------
PARAM_GRIDS = {
    "XGBoost": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 6, 9, 12],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1, 10],
        "reg_lambda": [0, 0.1, 1, 10],
    },
    "LightGBM": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 6, 9, 12],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1, 10],
        "reg_lambda": [0, 0.1, 1, 10],
    },
    "CatBoost": {
        "n_estimators": [100, 200, 300, 500],
        "depth": [3, 6, 9, 12],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bylevel": [0.6, 0.8, 1.0],
        "l2_leaf_reg": [1, 3, 5, 10],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 6, 9, 12, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
    },
    "MLP": {
        "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
        "max_iter": [200, 300],
    },
}

# ------------------------------------------------------------
# Manual search for TabNet (pytorch_tabnet) & FT‑Transformer
# ------------------------------------------------------------
TABNET_SEARCH_GRID = {
    "n_d": [8, 16, 32, 64],          # feature embedding dimension
    "n_a": [8, 16, 32, 64],          # attention embedding dimension
    "n_steps": [3, 5, 7],
    "gamma": [1.0, 1.3, 1.5, 2.0],
    "lambda_sparse": [1e-5, 1e-4, 1e-3],
    "lr": [0.01, 0.02, 0.05],
}

FT_TRANSFORMER_GRID = {
    "dim": [32, 64, 128],
    "depth": [2, 3, 4],
    "heads": [8],
    "attn_dropout": [0.0, 0.1, 0.2],
    "ff_dropout": [0.0, 0.1, 0.2],
    "lr": [1e-4, 1e-3, 1e-2],
}

def _sample_params(grid):
    """Sample one random combination from a grid."""
    return {k: np.random.choice(v) if isinstance(v, list) else v for k, v in grid.items()}

def _evaluate_model(model, X_train, y_train, X_val, y_val):
    """Train and evaluate accuracy on validation set."""
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    return accuracy_score(y_val, pred)

def manual_random_search(model_class, param_grid, X_train, y_train, n_iter=20, val_size=0.2):
    """Random search for a non‑sklearn model using a simple train/val split."""
    best_score = -1
    best_params = None
    # Use a fixed validation set for fairness
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=SEED, stratify=y_train
    )
    for i in range(n_iter):
        params = _sample_params(param_grid)
        # Clone model class with new parameters
        model = model_class(**params)
        try:
            acc = _evaluate_model(model, X_tr, y_tr, X_val, y_val)
            if acc > best_score:
                best_score = acc
                best_params = copy.deepcopy(params)
        except Exception as e:
            print(f"    Trial {i+1} failed with {params}: {e}")
    return {"best_params": best_params, "best_score": float(best_score) if best_params else 0.0}

# ------------------------------------------------------------
# Preprocessing (same as benchmark)
# ------------------------------------------------------------
def preprocess_data(X_raw):
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    return X_scaled

# ------------------------------------------------------------
# Run search for all baseline models
# ------------------------------------------------------------
def search_baseline_models(dataset_name, target_col):
    raw_path = PROJECT_ROOT / "data" / "raw" / f"{dataset_name}.csv"
    df = pd.read_csv(raw_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    n_features = X.shape[1]
    n_classes = len(le.classes_)

    # Scale for neural models
    X_scaled = preprocess_data(X)

    # Get model instances (dummy, we'll use classes)
    models_dict = get_models(n_features, n_classes)
    best_params_all = {}

    for model_name, model_obj in models_dict.items():
        if model_name.startswith("AG-T2I"):
            continue
        print(f"\n--- {model_name} ---")
        if model_name in ["IGTD-inspired", "Naive Reshape"]:
            print("  No tunable parameters, skipping.")
            continue

        # Choose data: tree models use raw, neural use scaled
        if model_name in ["XGBoost", "LightGBM", "CatBoost", "Random Forest"]:
            X_use = X.values
        else:
            X_use = X_scaled

        # ----- SKLearn compatible models (XGB, LGBM, CatBoost, RF, MLP) -----
        if model_name in PARAM_GRIDS:
            grid = PARAM_GRIDS[model_name]
            # For CatBoost we need to disable verbosity
            if model_name == "CatBoost":
                model_obj.set_params(verbose=0)
            if model_name == "MLP":
                model_obj.set_params(early_stopping=True, validation_fraction=0.1, random_state=SEED)

            search = RandomizedSearchCV(
                estimator=model_obj,
                param_distributions=grid,
                n_iter=N_ITER,
                cv=StratifiedKFold(n_splits=SEARCH_CV, shuffle=True, random_state=SEED),
                scoring="accuracy",
                n_jobs=1,
                random_state=SEED,
                verbose=1,
            )
            t0 = time.time()
            search.fit(X_use, y_enc)
            elapsed = time.time() - t0
            best = {"best_params": search.best_params_, "best_score": float(search.best_score_), "time": elapsed}
            print(f"  Best score: {best['best_score']:.4f}")

        # ----- TabNet (pytorch_tabnet) -----
        elif model_name == "TabNet":
            # Get class from object
            model_cls = model_obj.__class__
            best = manual_random_search(model_cls, TABNET_SEARCH_GRID, X_use, y_enc, n_iter=N_ITER)
            best["time"] = 0.0

        # ----- FT-Transformer (lite) -----
        elif model_name == "FT-Transformer (lite)":
            model_cls = model_obj.__class__
            best = manual_random_search(model_cls, FT_TRANSFORMER_GRID, X_use, y_enc, n_iter=N_ITER)
            best["time"] = 0.0

        else:
            print(f"  No search routine for {model_name}, skipping.")
            continue

        best_params_all[model_name] = best

    return best_params_all

# ------------------------------------------------------------
# AG‑T2I per‑layout search
# ------------------------------------------------------------
def search_agt2i_per_layout(dataset_name, target_col, trials_per_layout=20, method="random"):
    layouts = ["step_row", "packed", "packed_T", "step_sparse", "attention_map"]
    api = SimplePipelineAPI(base_path=PROJECT_ROOT)
    best_per_layout = {}

    for layout in layouts:
        print(f"\n--- AG-T2I-{layout} (method={method}) ---")
        if method == "random":
            df = api.random_search(
                dataset=dataset_name,
                target_column=target_col,
                layouts=[layout],
                n_trials=trials_per_layout,
                seed=SEED,
                quiet=False,
                optimization_metric="accuracy"
            )
        else:  # bayesian
            df = api.bayesian_search(
                dataset=dataset_name,
                target_column=target_col,
                layouts=[layout],
                n_trials=trials_per_layout,
                seed=SEED,
                quiet=False,
                optimization_metric="accuracy"
            )
        if df.empty:
            print("  No successful trials.")
            continue
        best = df.iloc[0]
        best_params = {}
        for col in df.columns:
            if col.startswith("param_"):
                key = col.replace("param_", "")
                best_params[key] = best[col]
        best_per_layout[f"AG-T2I-{layout}"] = {
            "best_params": best_params,
            "best_score": float(best["accuracy"]),
        }
        print(f"  Best accuracy: {best['accuracy']:.4f}")
    return best_per_layout

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def run_full_search(dataset_name, target_col="Class", agt2i_trials=20, agt2i_method="random"):
    print(f"=== Hyperparameter search for {dataset_name} ===")
    set_seed(SEED)

    # 1. Baseline models
    baseline_best = search_baseline_models(dataset_name, target_col)

    # 2. AG‑T2I per layout
    agt2i_best = search_agt2i_per_layout(dataset_name, target_col,
                                         trials_per_layout=agt2i_trials,
                                         method=agt2i_method)

    # Merge and save
    all_best = {**baseline_best, **agt2i_best}
    out_path = RESULTS_DIR / f"{dataset_name}_best_params.json"
    with open(out_path, "w") as f:
        json.dump(all_best, f, indent=2)
    print(f"\n✅ All best parameters saved to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset name (without .csv)")
    parser.add_argument("--target", default="Class")
    parser.add_argument("--agt2i_trials", type=int, default=20,
                        help="Number of trials per AG‑T2I layout")
    parser.add_argument("--agt2i_method", choices=["random", "bayesian"], default="random")
    args = parser.parse_args()

    run_full_search(args.dataset, args.target, args.agt2i_trials, args.agt2i_method)