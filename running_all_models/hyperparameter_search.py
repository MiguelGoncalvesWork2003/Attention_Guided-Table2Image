"""
Hyperparameter search for all models (baseline + AG‑T2I) – parallel version.
- CPU models: parallel across workers (joblib)
- GPU models: sequential to avoid contention on a single GPU
- Results saved to running_all_models/hyperparameter_results/<dataset>_best_params.json
"""

# --- Limit threading inside each worker (must be set before importing sklearn) ---
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import time
import copy
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
from joblib import Parallel, delayed, parallel_backend

from api import SimplePipelineAPI
from running_all_models.models_factory import get_models
from running_all_models.utils import set_seed

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
SEARCH_CV = 3          # inner CV for tree/MLP models
N_ITER = 20            # random parameter combinations for each model
SEED = 42
VAL_SIZE = 0.2         # validation split for neural models (TabNet, FT‑Transformer)
RESULTS_DIR = PROJECT_ROOT / "running_all_models" / "hyperparameter_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# AG‑T2I search settings
AGT2I_TRIALS_PER_LAYOUT = 20   # number of trials per layout
AGT2I_METHOD = "random"        # "random" or "bayesian"

# GPU models that should run sequentially (single GPU)
GPU_MODELS = {
    "TabNet",
    "FT-Transformer (lite)",
}

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
# Manual search grids for TabNet & FT‑Transformer
# ------------------------------------------------------------
TABNET_SEARCH_GRID = {
    "n_d": [8, 16, 32, 64],
    "n_a": [8, 16, 32, 64],
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
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=SEED, stratify=y_train
    )
    for i in range(n_iter):
        params = _sample_params(param_grid)
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
# Worker: search a single baseline model
#   (recreates model & loads data inside the worker)
# ------------------------------------------------------------
def search_single_baseline_model(model_name, dataset_name, target_col, n_features, n_classes):
    """
    Load data, build the model, and run hyperparameter search.
    Returns (model_name, best_result_dict) or (model_name, None) if skipped.
    """
    # Models with no tunable parameters
    if model_name in ["IGTD-inspired", "Naive Reshape"]:
        return model_name, None

    # Load dataset (lightweight – file cache makes it cheap)
    raw_path = PROJECT_ROOT / "data" / "raw" / f"{dataset_name}.csv"
    df = pd.read_csv(raw_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Preprocess only once per worker
    # scaled data for neural models
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # Choose raw vs scaled
    if model_name in ["XGBoost", "LightGBM", "CatBoost", "Random Forest"]:
        X_use = X.values
    else:
        X_use = X_scaled

    # Recreate the model instance
    models_dict = get_models(n_features, n_classes)
    model_obj = models_dict[model_name]

    # ----- sklearn‑compatible models (grid search) -----
    if model_name in PARAM_GRIDS:
        grid = PARAM_GRIDS[model_name]
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
            verbose=0,   # keep output clean when multiple workers run
        )
        t0 = time.time()
        search.fit(X_use, y_enc)
        elapsed = time.time() - t0
        best = {
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "time": elapsed,
        }

    # ----- TabNet (manual search, uses GPU) -----
    elif model_name == "TabNet":
        model_cls = model_obj.__class__
        best = manual_random_search(model_cls, TABNET_SEARCH_GRID, X_use, y_enc, n_iter=N_ITER)
        best["time"] = 0.0

    # ----- FT-Transformer (lite) (manual search, GPU if available) -----
    elif model_name == "FT-Transformer (lite)":
        model_cls = model_obj.__class__
        best = manual_random_search(model_cls, FT_TRANSFORMER_GRID, X_use, y_enc, n_iter=N_ITER)
        best["time"] = 0.0

    else:
        print(f"  No search routine for {model_name}, skipping.")
        return model_name, None

    print(f"  ✅ {model_name:20s} best score: {best['best_score']:.4f}  (time: {best.get('time',0):.1f}s)")
    return model_name, best

# ------------------------------------------------------------
# Worker: search a single AG‑T2I layout
# ------------------------------------------------------------
def search_single_ag_t2i_layout(layout, dataset_name, target_col, trials, method, seed):
    """
    Run random/bayesian search for one AG‑T2I layout.
    Returns (layout_key, best_result_dict) or (layout_key, None).
    """
    api = SimplePipelineAPI(base_path=PROJECT_ROOT)
    try:
        if method == "random":
            df = api.random_search(
                dataset=dataset_name,
                target_column=target_col,
                layouts=[layout],
                n_trials=trials,
                seed=seed,
                quiet=False,
                optimization_metric="accuracy",
            )
        else:  # bayesian
            df = api.bayesian_search(
                dataset=dataset_name,
                target_column=target_col,
                layouts=[layout],
                n_trials=trials,
                seed=seed,
                quiet=False,
                optimization_metric="accuracy",
            )
    except Exception as e:
        print(f"  ❌ AG-T2I-{layout} search failed: {e}")
        return f"AG-T2I-{layout}", None

    if df.empty:
        print(f"  ⚠ AG-T2I-{layout} no successful trials.")
        return f"AG-T2I-{layout}", None

    best = df.iloc[0]
    best_params = {col.replace("param_", ""): best[col] for col in df.columns if col.startswith("param_")}
    result = {
        "best_params": best_params,
        "best_score": float(best["accuracy"]),
    }
    print(f"  ✅ AG-T2I-{layout:20s} accuracy: {best['accuracy']:.4f}")
    return f"AG-T2I-{layout}", result

# ------------------------------------------------------------
# Main parallel search orchestration
# ------------------------------------------------------------
def run_full_search(dataset_name, target_col="Class", agt2i_trials=20, agt2i_method="random",
                    cpu_workers=None):
    print(f"=== Hyperparameter search for {dataset_name} ===")
    set_seed(SEED)

    # Determine n_features / n_classes from a quick peek (lightweight)
    raw_path = PROJECT_ROOT / "data" / "raw" / f"{dataset_name}.csv"
    df = pd.read_csv(raw_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_features = X.shape[1]
    n_classes = len(le.classes_)

    # Get model names only – do not create heavy instances here
    all_models = get_models(n_features, n_classes).keys()
    baseline_names = [m for m in all_models if not m.startswith("AG-T2I")]

    # Separate CPU / GPU models using simple set
    cpu_names = [m for m in baseline_names if m not in GPU_MODELS]
    gpu_names = [m for m in baseline_names if m in GPU_MODELS]

    # ----- 1. CPU models in parallel -----
    print("\n--- CPU models (parallel) ---")
    cpu_tasks = [(name, dataset_name, target_col, n_features, n_classes) for name in cpu_names]

    cpu_results = {}
    if cpu_tasks:
        # Use cpu_workers if given, else all cores minus one to avoid oversubscription
        n_jobs = cpu_workers if cpu_workers else max(1, os.cpu_count() - 1)
        with parallel_backend('loky', n_jobs=n_jobs):
            res_list = Parallel(verbose=10, pre_dispatch="2*n_jobs")(
                delayed(search_single_baseline_model)(*task) for task in cpu_tasks
            )
        for name, best in res_list:
            if best is not None:
                cpu_results[name] = best

    # ----- 2. GPU models (sequential to avoid contention) -----
    print("\n--- GPU models (sequential) ---")
    gpu_results = {}
    for name in gpu_names:
        # These models are few (TabNet, maybe FT-Transformer), so run one by one
        _, best = search_single_baseline_model(name, dataset_name, target_col, n_features, n_classes)
        if best is not None:
            gpu_results[name] = best

    # ----- 3. AG‑T2I layouts (sequential, all use GPU) -----
    layouts = ["step_row", "packed", "packed_T", "step_sparse", "attention_map"]
    agt2i_results = {}
    for layout in layouts:
        key, best = search_single_ag_t2i_layout(
            layout, dataset_name, target_col, agt2i_trials, agt2i_method, SEED
        )
        if best is not None:
            agt2i_results[key] = best

    # Combine and save
    all_best = {**cpu_results, **gpu_results, **agt2i_results}
    out_path = RESULTS_DIR / f"{dataset_name}_best_params.json"
    with open(out_path, "w") as f:
        json.dump(all_best, f, indent=2)

    print(f"\n✅ All best parameters saved to {out_path}")

    # ------------------------------------------------------------
    # Automatically run benchmark_parallel afterwards
    # ------------------------------------------------------------
    import subprocess

    benchmark_script = PROJECT_ROOT / "running_all_models" / "benchmark_parallel.py"
    results_hps_dir = PROJECT_ROOT / "running_all_models" / "results_hps"
    results_hps_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["RESULTS_DIR"] = str(results_hps_dir)

    print("\n🚀 Starting benchmark_parallel...\n")
    workers = cpu_workers or max(1, os.cpu_count() - 2)
    subprocess.run(
        [
            sys.executable,
            str(benchmark_script),
            "--dataset",
            dataset_name,
            "--workers",
            str(workers),
        ],
    )
    print("\n✅ Benchmark completed successfully!")
    print(f"📁 Results saved to: {results_hps_dir}")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset name (without .csv)")
    parser.add_argument("--target", default="Class")
    parser.add_argument("--agt2i_trials", type=int, default=20,
                        help="Number of trials per AG‑T2I layout")
    parser.add_argument("--agt2i_method", choices=["random", "bayesian"], default="random")
    parser.add_argument("--cpu_workers", type=int, default=None,
                        help="Number of parallel workers for CPU models (default: all CPUs minus one)")
    args = parser.parse_args()

    run_full_search(args.dataset, args.target, args.agt2i_trials, args.agt2i_method,
                    cpu_workers=args.cpu_workers)