"""
Benchmark script – parallel models per dataset, full CPU utilisation,
with caching of TabNet training per fold and explicit GPU support.
Now supports --model filter for single model benchmarking.
"""

import os
import sys
import json
import hashlib
import time
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from filelock import FileLock
import warnings
from models_factory import get_tuned_models
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ------------------------------------------------------------
# GPU configuration
# ------------------------------------------------------------
USE_GPU = True   # Set to False to force CPU usage

def get_torch_device():
    """Return 'cuda' if a GPU is available and USE_GPU is True, else 'cpu'."""
    if not USE_GPU:
        return 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except ImportError:
        pass
    return 'cpu'

# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def get_out_dir():
    """Return the results output directory (respecting RESULTS_DIR env var)."""
    out = Path(os.environ.get("RESULTS_DIR", 
                             str(PROJECT_ROOT / "running_all_models" / "results_parallel")))
    out.mkdir(parents=True, exist_ok=True)
    return out

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed, parallel_backend
from execution.runner import run_step

from running_all_models.metrics import compute_extended_metrics, get_wrong_cases
from running_all_models.utils import set_seed, mean_std_ci
from running_all_models.models_factory import get_models

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
N_SPLITS = 5
SEEDS = [0, 1, 2]
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

DATASETS = [
    ("Iris", "Class"),
    #("Diabetes", "Class"),
    #("Cancer", "Class"),
    #("Glass", "Class"),
    #("Card", "Class"),
    #("Thyroid", "Class"),
    #("Heart", "Class"),
    #("Horse", "Class"),
    #("Gene", "Class"),
    #("Soybean", "Class"),
    #("Adult", "Class"),
    #("Bank", "Class"),
    #("Electricity", "Class"),
    #("Magic04", "Class"),
    #("Poker_Hand", "Class"),
    #("Forest_Cover_Type", "Class"),
]

# ------------------------------------------------------------
# Utility: unique fold identifier for caching
# ------------------------------------------------------------
def _fold_id(dataset, seed, train_idx, test_idx, tabnet_params=None):
    """Return a unique identifier that also captures the tuned TabNet parameters."""
    base = f"{dataset}_seed{seed}_" + hashlib.md5(
        np.concatenate([train_idx, test_idx]).tobytes()
    ).hexdigest()[:12]
    if tabnet_params:
        param_hash = hashlib.md5(
            json.dumps(tabnet_params, sort_keys=True).encode()
        ).hexdigest()[:12]
        base = f"{base}_tabnet{param_hash}"
    return base

# ------------------------------------------------------------
# Preprocessing cache for baseline models
# ------------------------------------------------------------
def _cache_path(dataset, seed, fold, kind):
    return CACHE_DIR / dataset / f"seed{seed}" / f"fold{fold}" / f"{kind}"

def _cached_preprocessing(dataset, seed, fold, X_train_raw, X_test_raw):
    cache_train = _cache_path(dataset, seed, fold, "X_train.npy")
    cache_test  = _cache_path(dataset, seed, fold, "X_test.npy")
    lock = FileLock(str(cache_train) + ".lock")

    with lock:
        if cache_train.exists() and cache_test.exists():
            X_train = np.load(cache_train)
            X_test  = np.load(cache_test)
        else:
            imputer = SimpleImputer(strategy='median')
            X_train_imp = pd.DataFrame(
                imputer.fit_transform(X_train_raw),
                columns=X_train_raw.columns
            )
            X_test_imp = pd.DataFrame(
                imputer.transform(X_test_raw),
                columns=X_test_raw.columns
            )
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_imp)
            X_test  = scaler.transform(X_test_imp)
            cache_train.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_train, X_train)
            np.save(cache_test, X_test)
    return X_train, X_test

# ------------------------------------------------------------
# Helper to move model to GPU
# ------------------------------------------------------------
def _move_model_to_device(model):
    device = get_torch_device()
    if device == 'cuda' and hasattr(model, 'to'):
        model.to(device)

# ------------------------------------------------------------
# Global preprocessing (to avoid race conditions on Windows)
# ------------------------------------------------------------
def ensure_global_preprocessing(dataset_name, target_col):
    """Run preprocessing once globally before any parallel tasks."""
    base = PROJECT_ROOT
    global_processed = base / "data" / "processed" / dataset_name
    required_files = [
        "X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy",
        "feature_names.npy"
    ]
    if all((global_processed / f).exists() for f in required_files):
        return

    env = {
        "DATASET": dataset_name,
        "TARGET_COL": target_col,
        "SEED": "42",
        "DROP_THRESHOLD": "0.5",
        "CAT_MISSING": "explicit",
        "NUM_MISSING": "median",
        "SCALING": "standard",
        "ENCODE_CATEGORICALS": "true",
        "PROCESSED_DIR": str(global_processed),
        "MOL_LAYOUT": "step_row",   # irrelevant
        "EXPERIMENT_ID": "global_prep",
    }
    success, _, _ = run_step(
        name="Global Preprocessing",
        script_path=base / "preprocessing" / "run_preprocessing.py",
        env_vars=env,
    )
    if not success:
        raise RuntimeError(f"Global preprocessing failed for {dataset_name}")

# ------------------------------------------------------------
# AG‑T2I helper with caching + tuned hyperparameters
# ------------------------------------------------------------
def run_agt2i_fold(dataset, target, layout, seed, train_idx, test_idx,
                   tabnet_params=None, cnn_params=None):
    """
    Execute the pipeline for one AG‑T2I layout.
    Preprocessing + TabNet training are cached per (fold, tabnet_params).
    Only image building + CNN training are repeated for each layout.
    """
    base = PROJECT_ROOT
    fold_str = _fold_id(dataset, seed, train_idx, test_idx, tabnet_params)

    # ---- Unique cache directory for this fold's TabNet outputs ----
    fold_cache_dir = CACHE_DIR / "tabnet_cache" / fold_str
    fold_cache_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(fold_cache_dir / ".lock"))

    # ---- Global processed directory (read‑only source of original preprocessed data) ----
    global_processed = base / "data" / "processed" / dataset

    env = {
        "DATASET": dataset,
        "TARGET_COL": target,
        "SEED": str(seed),
        "DROP_THRESHOLD": "0.5",
        "CAT_MISSING": "explicit",
        "NUM_MISSING": "median",
        "SCALING": "standard",
        "ENCODE_CATEGORICALS": "true",
        "FORMAT_STEP_DISTRIBUTION": "true",
        "OPTIMIZATION_METRIC": "accuracy",
    }
    device = get_torch_device()
    if device == 'cuda':
        env["CUDA_VISIBLE_DEVICES"] = "0"

    # ---- Ensure global preprocessed data exists (once per dataset) ----
    global_lock = FileLock(str(global_processed / ".preprocess.lock"))
    required_global_files = [
        "X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy",
        "feature_names.npy"
    ]
    with global_lock:
        missing = [f for f in required_global_files if not (global_processed / f).exists()]
        if missing:
            # Run preprocessing on the full dataset (no custom split)
            env_prep = env.copy()
            env_prep["PROCESSED_DIR"] = str(global_processed)
            env_prep["MOL_LAYOUT"] = "step_row"  # layout irrelevant for preprocessing
            env_prep["EXPERIMENT_ID"] = "global_prep"
            # Remove any custom split env vars so the default split is used
            env_prep.pop("TRAIN_IDX_PATH", None)
            env_prep.pop("TEST_IDX_PATH", None)
            env_prep.pop("USE_CUSTOM_SPLIT", None)
            success, _, _ = run_step(
                name="Global Preprocessing",
                script_path=base / "preprocessing" / "run_preprocessing.py",
                env_vars=env_prep,
            )
            if not success:
                raise RuntimeError("Global preprocessing failed")

    # ---- Cached preprocessing + TabNet (once per unique fold/params) ----
    tabnet_flag = fold_cache_dir / "tabnet_trained.flag"
    with lock:
        if not tabnet_flag.exists():
            tmp_dir = fold_cache_dir / "tmp_work"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            # Copy global preprocessed files into tmp_dir
            for fname in required_global_files:
                src = global_processed / fname
                if src.exists():
                    shutil.copy2(src, tmp_dir / fname)
            # Also copy feature_names.npy from artifacts if present
            artifacts_dir = global_processed / "artifacts"
            if (artifacts_dir / "feature_names.npy").exists():
                shutil.copy2(artifacts_dir / "feature_names.npy", tmp_dir / "feature_names.npy")

            # Save train/test indices for the custom split
            idx_dir = tmp_dir / "custom_split"
            idx_dir.mkdir(exist_ok=True)
            train_path = idx_dir / "train_idx.npy"
            test_path  = idx_dir / "test_idx.npy"
            np.save(train_path, train_idx)
            np.save(test_path, test_idx)

            # Update environment to use the temporary directory
            env["PROCESSED_DIR"] = str(tmp_dir)
            env["MOL_LAYOUT"] = "step_row"          # arbitrary, needed for preprocessing
            env["EXPERIMENT_ID"] = f"tabnet_cache_{fold_str}"
            env["TRAIN_IDX_PATH"] = str(train_path)
            env["TEST_IDX_PATH"]  = str(test_path)
            env["USE_CUSTOM_SPLIT"] = "true"

            # Apply tuned TabNet parameters
            if tabnet_params:
                env.update({
                    "TABNET_N_STEPS": str(tabnet_params.get("n_steps", 6)),
                    "TABNET_STEP_DIM": str(tabnet_params.get("step_dim", 8)),
                    "TABNET_ATTN_DIM": str(tabnet_params.get("attn_dim", 8)),
                    "TABNET_GAMMA": str(tabnet_params.get("gamma", 1.5)),
                    "TABNET_LAMBDA_SPARSE": str(tabnet_params.get("lambda_sparse", 1e-4)),
                    "TABNET_MASK_TYPE": tabnet_params.get("mask_type", "sparsemax"),
                    "TABNET_LEARNING_RATE": str(tabnet_params.get("learning_rate", 2e-2)),
                    "TABNET_BATCH_SIZE": str(tabnet_params.get("batch_size", 32)),
                    "TABNET_MAX_EPOCHS": str(tabnet_params.get("max_epochs", 100)),
                })

            # Run preprocessing on the fold's subset (creates the custom split)
            success, _, _ = run_step(
                name="Preprocessing",
                script_path=base / "preprocessing" / "run_preprocessing.py",
                env_vars=env,
            )
            if not success:
                raise RuntimeError("Preprocessing failed for AG‑T2I cache")

            # ------------------------------------------------------------
            # Guard: TabNet batch norm needs >1 sample per batch
            # ------------------------------------------------------------
            X_train_check = np.load(tmp_dir / "X_train.npy")
            if X_train_check.shape[0] < 2:
                tabnet_flag.touch()   # mark as completed to avoid re-run
                # Return NaN metrics (will be ignored in summary)
                return {
                    "layout": layout,
                    "seed": seed,
                    "train": {
                        "accuracy": np.nan,
                        "balanced_accuracy": np.nan,
                        "f1_macro": np.nan,
                        "precision_macro": np.nan,
                        "recall_macro": np.nan,
                    },
                    "test": {
                        "accuracy": np.nan,
                        "balanced_accuracy": np.nan,
                        "f1_macro": np.nan,
                        "precision_macro": np.nan,
                        "recall_macro": np.nan,
                        "f1_weighted": np.nan,
                        "precision_weighted": np.nan,
                        "recall_weighted": np.nan,
                        "auroc": np.nan,
                    },
                }

            # TabNet training – redirect its output to fold_cache_dir
            env["OUTPUT_DIR"] = str(fold_cache_dir)
            try:
                success, _, _ = run_step(
                    name="TabNet Training",
                    script_path=base / "tabnet_fs" / "train_tabnet.py",
                    env_vars=env,
                )
                if not success:
                    raise RuntimeError("TabNet training failed")
            except Exception:
                tabnet_flag.touch()
                return {
                    "layout": layout,
                    "seed": seed,
                    "train": {
                        "accuracy": np.nan, "balanced_accuracy": np.nan,
                        "f1_macro": np.nan, "precision_macro": np.nan,
                        "recall_macro": np.nan,
                    },
                    "test": {
                        "accuracy": np.nan, "balanced_accuracy": np.nan,
                        "f1_macro": np.nan, "precision_macro": np.nan,
                        "recall_macro": np.nan, "f1_weighted": np.nan,
                        "precision_weighted": np.nan, "recall_weighted": np.nan,
                        "auroc": np.nan,
                    },
                }

            # Copy the produced step assignment back to global processed dir
            step_csv_src = Path(env["OUTPUT_DIR"]) / "tabnet_output" / "tabnet_step_assignment.csv"
            if step_csv_src.exists():
                shutil.copy2(step_csv_src, global_processed / "tabnet_step_assignment.csv")

            # Mark this fold as trained
            tabnet_flag.touch()

    # ---- Prepare isolated output directory for image building & CNN ----
    output_dir = global_processed / f"{fold_str}_{layout}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy the fold's preprocessed subset (from the cache) into the isolated directory
    fold_data_dir = fold_cache_dir / "tmp_work"   # where preprocessing saved its output
    for fname in ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy",
                  "feature_names.npy"]:
        src = fold_data_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    # Copy TabNet step assignment from global (now updated by this fold)
    step_csv = global_processed / "tabnet_step_assignment.csv"
    if step_csv.exists():
        shutil.copy2(step_csv, output_dir / "tabnet_step_assignment.csv")

    # Now set environment for the image building + CNN steps
    env["PROCESSED_DIR"] = str(output_dir)
    env["OUTPUT_DIR"] = str(output_dir)
    env["MOL_LAYOUT"] = layout
    env["EXPERIMENT_ID"] = f"{layout}_seed{seed}_{fold_str[:8]}"
    env["TRAIN_IDX_PATH"] = str(output_dir / "custom_split" / "train_idx.npy")
    env["TEST_IDX_PATH"]  = str(output_dir / "custom_split" / "test_idx.npy")
    env["USE_CUSTOM_SPLIT"] = "true"   # not actually used by image builder

    # Apply tuned CNN parameters
    if cnn_params:
        env.update({
            "CNN_LEARNING_RATE": str(cnn_params.get("learning_rate", 1e-3)),
            "CNN_OPTIMIZER": cnn_params.get("optimizer", "adam"),
            "CNN_EPOCHS": str(cnn_params.get("epochs", 50)),
            "CNN_BATCH_SIZE": str(cnn_params.get("batch_size", 32)),
            "CNN_DROPOUT": str(cnn_params.get("dropout", 0.3)),
        })

    # ---- Image building ----
    env["TABNET_STEP_CSV_PATH"] = str(fold_cache_dir / "tabnet_output" / "tabnet_step_assignment.csv")
    success, _, _ = run_step(
        name="Image Building",
        script_path=base / "image_builder" / "tabnet_image_builder.py",
        env_vars=env,
    )
    if not success:
        raise RuntimeError(f"Image building failed for layout {layout}")

    # ---- CNN training ----
    success, _, _ = run_step(
        name="CNN Training",
        script_path=base / "cnn" / "train_cnn.py",
        env_vars=env,
    )
    if not success:
        raise RuntimeError(f"CNN training failed for layout {layout}")

    # ---- CNN evaluation ----
    success, _, _ = run_step(
        name="CNN Evaluation",
        script_path=base / "cnn" / "evaluate_cnn.py",
        env_vars=env,
    )
    if not success:
        raise RuntimeError(f"CNN evaluation failed for layout {layout}")

    # ---- Collect results from the subdirectory created by train_cnn / evaluate_cnn ----
    eval_subdir = output_dir / f"{layout}_seed{seed}"

    # First try JSON files (preferred)
    results_file = eval_subdir / f"cnn_evaluation_results_{layout}.json"
    test_metrics = {}
    if results_file.exists():
        with open(results_file, "r") as f:
            test_metrics = json.load(f)

    train_file = eval_subdir / f"cnn_training_results_{layout}_seed{seed}.json"
    train_metrics = {}
    if train_file.exists():
        with open(train_file, "r") as f:
            train_metrics = json.load(f)

    # Fallback: compute directly from saved prediction files
    if not test_metrics:
        y_test_path = eval_subdir / "y_test.npy"
        y_pred_path = eval_subdir / f"y_test_pred_{layout}.npy"
        y_prob_path = eval_subdir / f"y_test_prob_{layout}.npy"
        if y_test_path.exists() and y_pred_path.exists():
            y_test = np.load(y_test_path)
            y_pred = np.load(y_pred_path)
            y_prob = np.load(y_prob_path) if y_prob_path.exists() else None
            y_test = y_test - y_test.min()
            test_metrics = compute_extended_metrics(y_test, y_pred, y_prob)

    return {
        "layout": layout,
        "seed": seed,
        "train": {
            "accuracy": train_metrics.get("train_accuracy", np.nan),
            "balanced_accuracy": train_metrics.get("train_balanced_accuracy", np.nan),
            "f1_macro": train_metrics.get("train_f1_macro", np.nan),
            "precision_macro": train_metrics.get("train_precision_macro", np.nan),
            "recall_macro": train_metrics.get("train_recall_macro", np.nan),
        },
        "test": {
            "accuracy": test_metrics.get("accuracy", np.nan),
            "balanced_accuracy": test_metrics.get("balanced_accuracy", np.nan),
            "f1_macro": test_metrics.get("f1_macro", np.nan),
            "precision_macro": test_metrics.get("precision_macro", np.nan),
            "recall_macro": test_metrics.get("recall_macro", np.nan),
            "f1_weighted": test_metrics.get("f1_weighted", np.nan),
            "precision_weighted": test_metrics.get("precision_weighted", np.nan),
            "recall_weighted": test_metrics.get("recall_weighted", np.nan),
            "auroc": test_metrics.get("roc_auc", np.nan),
        },
    }

# ------------------------------------------------------------
# Single‑model runner (updated signature)
# ------------------------------------------------------------
def run_model_on_fold(model_name, model_obj, dataset_name, target_col, seed, fold,
                      train_idx, test_idx, le_classes, tabnet_params=None, cnn_params=None):
    results = []
    def add_nan_rows():
        for subset in ["train", "test"]:
            results.append({
                "model": model_name, "seed": seed, "fold": fold, "subset": subset,
                "accuracy": np.nan, "balanced_accuracy": np.nan,
                "precision_macro": np.nan, "recall_macro": np.nan,
                "f1_macro": np.nan, "precision_weighted": np.nan,
                "recall_weighted": np.nan, "f1_weighted": np.nan,
                "roc_auc": np.nan, "time_sec": np.nan
            })

    try:
        raw_path = PROJECT_ROOT / "data" / "raw" / f"{dataset_name}.csv"
        if not raw_path.exists():
            print(f"[ERROR] {model_name} – dataset not found: {raw_path}")
            add_nan_rows()
            return results

        df = pd.read_csv(raw_path)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train_raw = X.iloc[train_idx]
        X_test_raw  = X.iloc[test_idx]
        y_train_fold = y_encoded[train_idx]
        y_test_fold  = y_encoded[test_idx]

        is_agt2i = model_name.startswith("AG-T2I-")

        if not is_agt2i:
            if model_obj is None:
                raise ValueError(f"Model object is None for baseline {model_name}")

            # IGTD does not need scaling (its mapper does min‑max internally)
            model_needs_scaling = model_name in [
                "FT-Transformer (lite)", "IGTD-inspired", "Naive Reshape", "TabNet"
            ]
            if model_needs_scaling:
                X_train, X_test = _cached_preprocessing(
                    dataset_name, seed, fold, X_train_raw, X_test_raw
                )
            else:
                X_train = X_train_raw.values
                X_test  = X_test_raw.values

            _move_model_to_device(model_obj)

            start_t = time.time()
            if model_name == "TabNet":
                if len(y_train_fold) < 2:
                    raise ValueError("Training set too small for TabNet (batch norm requires >1 sample)")
                model_obj.fit(
                    X_train, y_train_fold,
                    eval_set=[(X_test, y_test_fold)],
                    eval_metric=["accuracy"],
                    max_epochs=200, patience=20,
                    batch_size=16, virtual_batch_size=8,
                    drop_last=False
                )
            else:
                model_obj.fit(X_train, y_train_fold)

            if hasattr(model_obj, "predict_proba"):
                y_proba_train = model_obj.predict_proba(X_train)
                y_pred_train = np.argmax(y_proba_train, axis=1)
                y_proba_test = model_obj.predict_proba(X_test)
                y_pred_test = np.argmax(y_proba_test, axis=1)
            else:
                y_proba_train = None
                y_pred_train = model_obj.predict(X_train)
                y_proba_test = None
                y_pred_test = model_obj.predict(X_test)

            elapsed = time.time() - start_t

            train_metrics = compute_extended_metrics(
                y_train_fold, y_pred_train, y_proba_train
            )
            test_metrics = compute_extended_metrics(
                y_test_fold, y_pred_test, y_proba_test
            )
            for subset, met in [("train", train_metrics), ("test", test_metrics)]:
                results.append({
                    "model": model_name, "seed": seed, "fold": fold,
                    "subset": subset, **met, "time_sec": elapsed,
                })

            if y_proba_test is not None:
                roc_df = pd.DataFrame(
                    y_proba_test,
                    columns=[f"prob_class_{i}" for i in range(y_proba_test.shape[1])]
                )
                roc_df["true_label"] = y_test_fold
                roc_out = get_out_dir() / "roc_data" / dataset_name / model_name
                roc_out.mkdir(parents=True, exist_ok=True)
                roc_df.to_csv(roc_out / f"seed{seed}_fold{fold}.csv", index=False)

            le_inner = LabelEncoder()
            le_inner.classes_ = le_classes
            true_decoded = le_inner.inverse_transform(y_test_fold)
            pred_decoded = le_inner.inverse_transform(y_pred_test)
            wrong_df = get_wrong_cases(
                y_test_fold, y_pred_test,
                indices=test_idx,
                true_labels=true_decoded,
                pred_labels=pred_decoded
            )
            if not wrong_df.empty:
                wrong_out = (
                    PROJECT_ROOT / "running_all_models" / "misclassified" /
                    dataset_name / model_name
                )
                wrong_out.mkdir(parents=True, exist_ok=True)
                wrong_df.to_csv(wrong_out / f"seed{seed}_fold{fold}.csv", index=False)

            print(f"  ✅ {model_name:20s} seed {seed} fold {fold} done in {elapsed:.1f}s")

        else:  # AG‑T2I variant
            layout = model_name.replace("AG-T2I-", "")
            start_t = time.time()
            result = run_agt2i_fold(
                dataset_name, target_col, layout, seed, train_idx, test_idx,
                tabnet_params, cnn_params
            )
            elapsed = time.time() - start_t
            for subset in ["train", "test"]:
                sub = result.get(subset, {})
                results.append({
                    "model": model_name, "seed": seed, "fold": fold,
                    "subset": subset,
                    "accuracy": sub.get("accuracy", np.nan),
                    "balanced_accuracy": sub.get("balanced_accuracy", np.nan),
                    "precision_macro": sub.get("precision_macro", np.nan),
                    "recall_macro": sub.get("recall_macro", np.nan),
                    "f1_macro": sub.get("f1_macro", np.nan),
                    "precision_weighted": sub.get("precision_weighted", np.nan),
                    "recall_weighted": sub.get("recall_weighted", np.nan),
                    "f1_weighted": sub.get("f1_weighted", np.nan),
                    "roc_auc": sub.get("auroc", np.nan),
                    "time_sec": elapsed
                })
            print(f"  ✅ {model_name:20s} seed {seed} fold {fold} done in {elapsed:.1f}s")

    except Exception as e:
        print(f"  ❌ {model_name:20s} seed {seed} fold {fold} ERROR: {e}")
        add_nan_rows()

    return results


# ------------------------------------------------------------
# Load AG‑T2I hyperparameters from best_params/<dataset>.json
# ------------------------------------------------------------
def load_agt2i_params(dataset_name):
    params_file = Path(__file__).parent / "best_params" / f"{dataset_name}.json"
    if not params_file.exists():
        return None
    with open(params_file, "r") as f:
        all_params = json.load(f)

    agt2i_params = {}
    for key, val in all_params.items():
        if key.startswith("AG-T2I-"):
            layout = key.replace("AG-T2I-", "")
            tabnet_keys = {
                "n_steps", "step_dim", "attn_dim", "gamma", "lambda_sparse",
                "mask_type", "learning_rate", "batch_size", "max_epochs"
            }
            cnn_keys = {"cnn_lr", "cnn_optimizer", "cnn_dropout", "cnn_epochs"}
            tabnet_params = {}
            cnn_params = {}
            for k, v in val.items():
                if k in tabnet_keys:
                    tabnet_params[k] = v
                elif k in cnn_keys:
                    cnn_params[k.replace("cnn_", "")] = v
            agt2i_params[layout] = (tabnet_params, cnn_params)
    return agt2i_params


# ------------------------------------------------------------
# Benchmark a single dataset (with model_filter support)
# ------------------------------------------------------------
def run_dataset_benchmark(dataset_name, target_col, n_workers=None, only_agt2i=False, model_filter=None):
    raw_path = PROJECT_ROOT / "data" / "raw" / f"{dataset_name}.csv"
    if not raw_path.exists():
        print(f"❌ Dataset not found: {raw_path}")
        return None, None

    # ---- Global preprocessing (once, before any parallel task) ----
    ensure_global_preprocessing(dataset_name, target_col)

    df = pd.read_csv(raw_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_features = X.shape[1]
    n_classes = len(le.classes_)
    le_classes = le.classes_

    # --- Load baseline models (or empty if only_agt2i) ---
    models_dict = {}
    if not only_agt2i:
        models_dict = get_tuned_models(dataset_name, n_features, n_classes)

    # Default AG‑T2I layouts
    agt2i_layouts = [
        "step_row", "packed", "packed_T", "step_sparse", "attention_map"
    ]
    agt2i_params = load_agt2i_params(dataset_name)

    # --- Apply model_filter if provided ---
    if model_filter is not None:
        if isinstance(model_filter, str):
            model_filter = [model_filter]    # normalise to list

        baseline_models = []
        agt2i_selected = []
        for m in model_filter:
            if m.startswith("AG-T2I-"):
                layout = m.replace("AG-T2I-", "")
                if layout not in agt2i_layouts:
                    print(f"Unknown AG‑T2I layout: {layout}")
                    return None, None
                agt2i_selected.append(layout)
            else:
                if m not in models_dict:
                    print(f"Model '{m}' not found in tuned models for {dataset_name}. "
                          f"Available: {list(models_dict.keys())}")
                    return None, None
                baseline_models.append(m)

        # Keep only requested baselines
        if baseline_models:
            models_dict = {m: models_dict[m] for m in baseline_models}
        else:
            models_dict = {}

        # Keep only requested AG‑T2I layouts
        if agt2i_selected:
            agt2i_layouts = agt2i_selected
        else:
            agt2i_layouts = []

        # Determine the only_agt2i flag
        only_agt2i = (len(models_dict) == 0 and len(agt2i_layouts) > 0)

    # --- Build task list ---
    tasks = []
    for seed in SEEDS:
        set_seed(seed)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded)):
            # Baseline models
            for m_name, m_obj in models_dict.items():
                tasks.append((
                    m_name, m_obj, dataset_name, target_col, seed, fold,
                    train_idx, test_idx, le_classes, None, None
                ))
            # AG‑T2I layouts
            for layout in agt2i_layouts:
                tabnet_prm = None
                cnn_prm = None
                if agt2i_params and layout in agt2i_params:
                    tabnet_prm, cnn_prm = agt2i_params[layout]
                tasks.append((
                    f"AG-T2I-{layout}", None, dataset_name, target_col, seed, fold,
                    train_idx, test_idx, le_classes, tabnet_prm, cnn_prm
                ))

    if not tasks:
        print("No tasks to run.")
        return None, None

    mode_str = "AG‑T2I only" if only_agt2i else "baseline + AG‑T2I"
    if model_filter:
        mode_str = f"model={model_filter}"
    print(f"Submitting {len(tasks)} tasks ({mode_str}) in parallel...")
    with parallel_backend('loky', n_jobs=n_workers):
        all_results = Parallel(verbose=10)(
            delayed(run_model_on_fold)(*task) for task in tasks
        )

    flat_results = []
    for res_list in all_results:
        flat_results.extend(res_list)

    results_df = pd.DataFrame(flat_results)
    suffix = "_agt2i" if only_agt2i else ""
    if model_filter:
        if isinstance(model_filter, list):
            safe_name = '_'.join(model_filter)
        else:
            safe_name = model_filter
        suffix = f"_{safe_name}"
    results_df.to_csv(get_out_dir() / f"{dataset_name}_raw{suffix}.csv", index=False)

    summary = []
    for model in results_df["model"].unique():
        sub = results_df[(results_df["model"] == model) & (results_df["subset"] == "test")]
        sub = sub.dropna(subset=["accuracy", "f1_macro"])
        if sub.empty:
            continue
        for metric in ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
                       "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc"]:
            if metric in sub.columns and sub[metric].notna().any():
                mean_val, std_val, ci_val = mean_std_ci(sub[metric].dropna())
                summary.append({
                    "model": model, "metric": metric,
                    "mean": mean_val, "std": std_val, "ci95": ci_val,
                    "time_sec": sub["time_sec"].mean()
                })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(get_out_dir() / f"{dataset_name}_summary{suffix}.csv", index=False)
    with open(get_out_dir() / f"{dataset_name}_summary{suffix}.tex", "w") as f:
        f.write(summary_df.to_latex(index=False, float_format="%.4f"))

    print(f"✅ {dataset_name} benchmark complete. Results in {get_out_dir()}\n")
    return results_df, summary_df


# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--model", nargs='+', default=None,
                    help="One or more model names (space separated).")
    parser.add_argument("--workers", type=int, default=-1,
                        help="Number of parallel workers (default: all CPUs)")
    parser.add_argument("--agt2i", action="store_true",
                        help="Run only AG‑T2I models (skip all baselines)")
    args = parser.parse_args()

    device = get_torch_device()
    print(f"🚀 GPU acceleration: {'ENABLED (CUDA)' if device == 'cuda' else 'DISABLED (CPU)'}")

    if args.dataset:
        datasets_to_run = [
            (ds, tgt) for ds, tgt in DATASETS if ds == args.dataset
        ]
        if not datasets_to_run:
            print(f"Dataset '{args.dataset}' not found. "
                  f"Available: {[ds for ds,_ in DATASETS]}")
            sys.exit(1)
    else:
        datasets_to_run = DATASETS

    print("=" * 60)
    if args.model:
        if args.model:
            mode = f"MODELS: {', '.join(args.model)}"
    elif args.agt2i:
        mode = "AG‑T2I ONLY"
    else:
        mode = "BASELINE + AG‑T2I"
    print(f"STARTING BENCHMARK ({mode})")
    print("=" * 60)

    total_start = time.time()
    for ds_name, ds_target in datasets_to_run:
        print(f"\n▶ Running {ds_name}...")
        run_dataset_benchmark(ds_name, ds_target, n_workers=args.workers,
                              only_agt2i=args.agt2i, model_filter=args.model)

    total_elapsed = time.time() - total_start
    print("\n🏁 All benchmarks finished.")
    print(f"⏱️ Total elapsed time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")