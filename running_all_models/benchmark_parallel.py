"""
Benchmark script – parallel models per dataset, full CPU utilisation,
with caching of TabNet training per fold and explicit GPU support.
(Fixes: parallel‑safe AG‑T2I splits, top‑level worker functions for joblib,
        full parallelism for AG‑T2I tasks via OUTPUT_DIR isolation)
"""

import os
import sys
import json
import hashlib
import time
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

out_dir = Path(
    os.environ.get(
        "RESULTS_DIR",
        PROJECT_ROOT / "running_all_models" / "results_parallel"
    )
)
out_dir.mkdir(parents=True, exist_ok=True)

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
SEEDS = [0, 1, 2, 3, 4]
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

DATASETS = [
    ("Iris", "Class"),
    ("Diabetes", "Class"),
    ("Cancer", "Class"),
    ("Glass", "Class"),
    ("Card", "Class"),
    ("Thyroid", "Class"),
    ("Heart", "Class"),
    ("Horse", "Class"),
    ("Gene", "Class"),
    ("Soybean", "Class"),
    ("Adult", "Class"),
    ("Bank", "Class"),
    ("Electricity", "Class"),
    ("Magic04", "Class"),
    ("Poker_Hand", "Class"),
    ("Forest_Cover_Type", "Class"),
]

# ------------------------------------------------------------
# Utility: unique fold identifier for caching
# ------------------------------------------------------------
def _fold_id(dataset, seed, train_idx, test_idx):
    """Return a short string that uniquely identifies this fold."""
    raw = f"{dataset}_seed{seed}_" + hashlib.md5(
        np.concatenate([train_idx, test_idx]).tobytes()
    ).hexdigest()[:12]
    return raw

# ------------------------------------------------------------
# Preprocessing cache for baseline models
# ------------------------------------------------------------
def _cache_path(dataset, seed, fold, kind):
    return CACHE_DIR / dataset / f"seed{seed}" / f"fold{fold}" / f"{kind}"

def _cached_preprocessing(dataset, seed, fold, X_train_raw, X_test_raw):
    """Return scaled X_train, X_test.  Cached to disk per fold/seed."""
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
# Helper to move model to GPU if it supports it
# ------------------------------------------------------------
def _move_model_to_device(model):
    """Move a PyTorch model to the GPU if it has a .to() method."""
    device = get_torch_device()
    if device == 'cuda' and hasattr(model, 'to'):
        model.to(device)

# ------------------------------------------------------------
# AG‑T2I helper with caching of preprocessing + TabNet per fold
# (Parallel‑safe: per‑fold split files and unique OUTPUT_DIR)
# ------------------------------------------------------------
def run_agt2i_fold(dataset, target, layout, seed, train_idx, test_idx):
    """
    Execute the pipeline for one AG‑T2I layout.
    Preprocessing + TabNet training are cached per fold.
    Only image building + CNN training are repeated for each layout.
    """
    base = PROJECT_ROOT
    fold_str = _fold_id(dataset, seed, train_idx, test_idx)
    cache_dir = CACHE_DIR / "tabnet_cache" / fold_str
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(cache_dir / ".lock"))

    processed_dir = base / "data" / "processed" / dataset
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Per‑fold directory for custom split files – prevents parallel overwrites
    idx_dir = processed_dir / "custom_split" / fold_str
    idx_dir.mkdir(parents=True, exist_ok=True)
    train_path = idx_dir / "train_idx.npy"
    test_path  = idx_dir / "test_idx.npy"

    # Common environment (without layout/output yet)
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

    # -------- Preprocessing + TabNet (cached per fold) --------
    tabnet_flag = cache_dir / "tabnet_trained.flag"
    with lock:
        if not tabnet_flag.exists():
            # Write custom split files ONCE for this fold
            np.save(train_path, train_idx)
            np.save(test_path, test_idx)

            # Use a fixed layout for the cached part (step_row)
            env["MOL_LAYOUT"] = "step_row"
            env["EXPERIMENT_ID"] = f"tabnet_cache_{fold_str}"
            env["TRAIN_IDX_PATH"] = str(train_path)
            env["TEST_IDX_PATH"]  = str(test_path)
            env["USE_CUSTOM_SPLIT"] = "true"
            # No OUTPUT_DIR needed for preprocessing/TabNet – they use global dirs

            success, _, _ = run_step(
                name="Preprocessing",
                script_path=base / "preprocessing" / "run_preprocessing.py",
                env_vars=env,
            )
            if not success:
                raise RuntimeError("Preprocessing failed for AG‑T2I cache")

            success, _, _ = run_step(
                name="TabNet Training",
                script_path=base / "tabnet_fs" / "train_tabnet.py",
                env_vars=env,
            )
            if not success:
                raise RuntimeError("TabNet training failed for AG‑T2I cache")

            tabnet_flag.touch()

    # -------- Now set layout-specific environment and OUTPUT_DIR ----
    env["MOL_LAYOUT"] = layout
    env["EXPERIMENT_ID"] = f"{layout}_seed{seed}_{fold_str[:8]}"
    env["TRAIN_IDX_PATH"] = str(train_path)
    env["TEST_IDX_PATH"]  = str(test_path)
    env["USE_CUSTOM_SPLIT"] = "true"

    # Unique output directory for this (seed, fold, layout) combination
    output_dir = processed_dir / f"{fold_str}_{layout}"
    output_dir.mkdir(parents=True, exist_ok=True)
    env["OUTPUT_DIR"] = str(output_dir)

    # -------- Image building for this specific layout --------
    success, _, _ = run_step(
        name="Image Building",
        script_path=base / "image_builder" / "tabnet_image_builder.py",
        env_vars=env,
    )
    if not success:
        raise RuntimeError(f"Image building failed for layout {layout}")

    # -------- CNN training --------
    success, _, _ = run_step(
        name="CNN Training",
        script_path=base / "cnn" / "train_cnn.py",
        env_vars=env,
    )
    if not success:
        raise RuntimeError(f"CNN training failed for layout {layout}")

    # -------- CNN evaluation --------
    success, _, _ = run_step(
        name="CNN Evaluation",
        script_path=base / "cnn" / "evaluate_cnn.py",
        env_vars=env,
    )
    if not success:
        raise RuntimeError(f"CNN evaluation failed for layout {layout}")

    # -------- Collect results (from the task’s output_dir) --------
    results_file = output_dir / f"cnn_evaluation_results_{layout}.json"
    test_metrics = {}
    if results_file.exists():
        with open(results_file, "r") as f:
            test_metrics = json.load(f)

    train_file = output_dir / f"cnn_training_results_{layout}_seed{seed}.json"
    train_metrics = {}
    if train_file.exists():
        with open(train_file, "r") as f:
            train_metrics = json.load(f)

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
            "f1_macro": test_metrics.get("f1_macro", test_metrics.get("f1_score", np.nan)),
            "precision_macro": test_metrics.get("precision_macro", np.nan),
            "recall_macro": test_metrics.get("recall_macro", np.nan),
            "f1_weighted": test_metrics.get("f1_weighted", np.nan),
            "precision_weighted": test_metrics.get("precision_weighted", np.nan),
            "recall_weighted": test_metrics.get("recall_weighted", np.nan),
            "auroc": test_metrics.get("roc_auc", test_metrics.get("auroc", np.nan)),
        },
    }

# ------------------------------------------------------------
# Single‑model runner – loads data inside worker to avoid pickling
# ------------------------------------------------------------
def run_model_on_fold(model_name, model_obj, dataset_name, target_col, seed, fold,
                      train_idx, test_idx, le_classes):
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

        # ---- Baseline (non‑AG‑T2I) models ----
        if not is_agt2i:
            if model_obj is None:
                raise ValueError(f"Model object is None for baseline {model_name}")

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
                roc_out = out_dir / "roc_data" / dataset_name / model_name
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

        # ---- AG‑T2I variants (now fully parallel) ----
        else:
            layout = model_name.replace("AG-T2I-", "")
            start_t = time.time()
            result = run_agt2i_fold(
                dataset_name, target_col, layout, seed, train_idx, test_idx
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
                    "precision_weighted": np.nan,
                    "recall_weighted": np.nan,
                    "f1_weighted": np.nan,
                    "roc_auc": sub.get("auroc", np.nan),
                    "time_sec": elapsed
                })
            print(f"  ✅ {model_name:20s} seed {seed} fold {fold} done in {elapsed:.1f}s")

    except Exception as e:
        print(f"  ❌ {model_name:20s} seed {seed} fold {fold} ERROR: {e}")
        add_nan_rows()

    return results


# ------------------------------------------------------------
# Benchmark a single dataset – ALL tasks submitted in parallel
# ------------------------------------------------------------
def run_dataset_benchmark(dataset_name, target_col, n_workers=None):
    raw_path = PROJECT_ROOT / "data" / "raw" / f"{dataset_name}.csv"
    if not raw_path.exists():
        print(f"❌ Dataset not found: {raw_path}")
        return None, None

    df = pd.read_csv(raw_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_features = X.shape[1]
    n_classes = len(le.classes_)
    le_classes = le.classes_

    models_dict = get_tuned_models(dataset_name, n_features, n_classes)
    agt2i_layouts = [
        "step_row", "packed", "packed_T", "step_sparse", "attention_map"
    ]

    # Build a single flat list of ALL tasks (baseline + AG-T2I)
    tasks = []
    for seed in SEEDS:
        set_seed(seed)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded)):
            # Baseline models
            for m_name, m_obj in models_dict.items():
                tasks.append((
                    m_name, m_obj, dataset_name, target_col, seed, fold,
                    train_idx, test_idx, le_classes
                ))
            # AG‑T2I models (one task per layout)
            for layout in agt2i_layouts:
                tasks.append((
                    f"AG-T2I-{layout}", None, dataset_name, target_col, seed, fold,
                    train_idx, test_idx, le_classes
                ))

    # Submit everything at once
    print(f"Submitting {len(tasks)} tasks (baseline + AG‑T2I) in parallel...")
    with parallel_backend('loky', n_jobs=n_workers):
        all_results = Parallel(verbose=10)(
            delayed(run_model_on_fold)(*task) for task in tasks
        )

    # Flatten results
    flat_results = []
    for res_list in all_results:
        flat_results.extend(res_list)

    results_df = pd.DataFrame(flat_results)
    results_df.to_csv(out_dir / f"{dataset_name}_raw.csv", index=False)

    # Summary (unchanged)
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
    summary_df.to_csv(out_dir / f"{dataset_name}_summary.csv", index=False)
    with open(out_dir / f"{dataset_name}_summary.tex", "w") as f:
        f.write(summary_df.to_latex(index=False, float_format="%.4f"))

    print(f"✅ {dataset_name} benchmark complete. Results in {out_dir}\n")
    return results_df, summary_df


# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: all CPUs minus one)")
    args = parser.parse_args()

    device = get_torch_device()
    print(f"🚀 GPU acceleration: {'ENABLED (CUDA)' if device == 'cuda' else 'DISABLED (CPU)'}")

    if args.dataset:
        datasets_to_run = [
            (ds, tgt) for ds, tgt in DATASETS if ds == args.dataset
        ]
        if not datasets_to_run:
            print(
                f"Dataset '{args.dataset}' not found. "
                f"Available: {[ds for ds,_ in DATASETS]}"
            )
            sys.exit(1)
    else:
        datasets_to_run = DATASETS

    print("=" * 60)
    print("STARTING BENCHMARK (TABNET CACHING + FULL PARALLELISM + GPU)")
    print("=" * 60)

    total_start = time.time()          # ← start timer

    for ds_name, ds_target in datasets_to_run:
        print(f"\n▶ Running {ds_name}...")
        run_dataset_benchmark(ds_name, ds_target, n_workers=args.workers)

    total_elapsed = time.time() - total_start   # ← stop timer
    print("\n🏁 All benchmarks finished.")
    print(f"⏱️ Total elapsed time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")