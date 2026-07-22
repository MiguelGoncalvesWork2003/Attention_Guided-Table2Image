"""
hyperparameter_search.py – Parallel hyper‑parameter tuning (quiet) +
automatic sequential benchmark with tuned parameters.
All models (baselines + AG‑T2I) are tuned concurrently.
Now uses 3‑fold CV for AG‑T2I as well.
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"
import sys
import json
import time
import warnings
import logging
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import loguniform, uniform, randint
from joblib import Parallel, delayed
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*No early stopping will be performed.*")
logging.getLogger("joblib").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from execution.runner import run_step
from running_all_models.benchmark_parallel import run_agt2i_fold
from running_all_models.models_factory import get_models

# ---------- CONFIGURATION ----------
DATASETS = [
    #("Iris", "Class"),
    #("Diabetes", "Class"),
    #("Cancer", "Class"),
    #("Glass", "Class"),
    #("Card", "Class"),
    ("Thyroid", "Class"),
    ("Heart", "Class"),
    ("Horse", "Class"),
    #("Gene", "Class"),
    #("Soybean", "Class"),
    #("Adult", "Class"),
    #("Bank", "Class"),
    #("Electricity", "Class"),
    #("Magic04", "Class"),
    #("Poker_Hand", "Class"),
    #("Forest_Cover_Type", "Class"),
]

MODELS = [
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "TabNet",
    "FT-Transformer (lite)",
    "IGTD-inspired",
    "Naive Reshape",
]

TRIALS = {
    "XGBoost": 25,
    "LightGBM": 25,
    "CatBoost": 25,
    "TabNet": 25,
    "FT-Transformer (lite)": 25,
    "IGTD-inspired": 25,
    "Naive Reshape": 25,
}

AGT2I_TRIALS = 25
AGT2I_LAYOUTS = ["step_row", "packed", "packed_T", "step_sparse", "attention_map"]

OUT_DIR = Path(__file__).parent / "best_params"
OUT_DIR.mkdir(exist_ok=True, parents=True)

OPTUNA_DB_DIR = PROJECT_ROOT / "experiments" / "hyperparameter_search"
OPTUNA_DB_DIR.mkdir(parents=True, exist_ok=True)


def ensure_global_preprocessing(dataset_name, target_col):
    """Run preprocessing once globally, so parallel tasks don't collide."""
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


def load_and_prepare_dataset(dataset_name: str, target_col: str):
    raw_path = PROJECT_ROOT / "data" / "raw" / f"{dataset_name}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset not found: {raw_path}")
    df = pd.read_csv(raw_path)
    X_df = df.drop(columns=[target_col])
    y = df[target_col]
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_df)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X_imp, y_enc, X_imp.shape[1], len(le.classes_)


def tune_single_model(dataset_name: str, target_col: str, model_name: str, n_trials: int):
    X_full, y_full, n_features, n_classes = load_and_prepare_dataset(dataset_name, target_col)

    if model_name in ["TabNet", "FT-Transformer (lite)", "IGTD-inspired", "Naive Reshape"]:
        scaler = StandardScaler()
        X_full = scaler.fit_transform(X_full)

    model_dict = get_models(n_features, n_classes)
    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}")
    base_model = model_dict[model_name]

    fit_params = {}
    if model_name == "XGBoost":
        param_dist = {
            "n_estimators": randint(100, 300),
            "max_depth": randint(3, 7),
            "learning_rate": loguniform(0.01, 0.2),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
        }
    elif model_name == "LightGBM":
        param_dist = {
            "n_estimators": randint(100, 300),
            "num_leaves": randint(15, 63),
            "learning_rate": loguniform(0.01, 0.2),
            "subsample": uniform(0.6, 0.4),
            "feature_fraction": uniform(0.6, 0.4),
        }
        base_model.set_params(n_jobs=1)
    elif model_name == "CatBoost":
        param_dist = {
            "n_estimators": randint(100, 300),
            "depth": randint(4, 8),
            "learning_rate": loguniform(0.01, 0.2),
            "l2_leaf_reg": uniform(1, 9),
        }
        base_model.set_params(thread_count=1)
    elif model_name == "TabNet":
        param_dist = {
            "n_d": [8, 16],
            "n_a": [8, 16],
            "n_steps": [3, 4, 5],
            "gamma": uniform(1.0, 0.5),
            "lambda_sparse": loguniform(1e-5, 1e-3),
            "optimizer_params": [{"lr": 0.02}, {"lr": 0.01}, {"lr": 0.005}],
        }
        fit_params = {
            "eval_metric": ["accuracy"],
            "max_epochs": 200, "patience": 20,
            "batch_size": 16, "virtual_batch_size": 8,
            "drop_last": False,
        }
    elif model_name == "FT-Transformer (lite)":
        param_dist = {
            "lr": loguniform(1e-4, 3e-3),
            "batch_size": [16, 32],
            "epochs": [50, 80],
        }
    elif model_name in ["IGTD-inspired", "Naive Reshape"]:
        param_dist = {
            "epochs": [50, 80],
            "lr": loguniform(1e-4, 1e-3),
            "dropout": uniform(0.1, 0.4),
        }
    else:
        return dataset_name, model_name, {}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        base_model, param_distributions=param_dist,
        n_iter=n_trials, cv=cv, scoring="f1_macro",
        n_jobs=1, random_state=42, verbose=0,
    )

    start = time.time()
    search.fit(X_full, y_full, **fit_params)
    elapsed = time.time() - start

    print(f"✅ {model_name:25s} on {dataset_name:20s}  →  best CV F1: {search.best_score_:.4f}  ({elapsed:.1f}s)")
    return dataset_name, model_name, search.best_params_


def tune_agt2i_layout(dataset_name: str, target_col: str, layout: str, n_trials: int):
    """Tune a single AG‑T2I layout using 3‑fold cross‑validation (like baselines)."""
    X_full, y_full, _, _ = load_and_prepare_dataset(dataset_name, target_col)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    splits = list(cv.split(X_full, y_full))

    study_name = f"{dataset_name}_{layout}_bayesian_v2"
    db_path = OPTUNA_DB_DIR / f"optuna_study_{layout}.db"
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    def objective(trial):
        tabnet_params = {
            "n_steps": trial.suggest_int("n_steps", 2, 10),
            "step_dim": trial.suggest_int("step_dim", 4, 64, log=True),
            "attn_dim": trial.suggest_int("attn_dim", 4, 64, log=True),
            "gamma": trial.suggest_float("gamma", 1.0, 2.5),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        }
        cnn_params = {
            "learning_rate": trial.suggest_float("cnn_lr", 1e-4, 1e-2, log=True),
            "optimizer": trial.suggest_categorical("cnn_optimizer", ["adam", "sgd", "rmsprop"]),
            "dropout": trial.suggest_float("cnn_dropout", 0.1, 0.6),
            "epochs": trial.suggest_int("cnn_epochs", 30, 100),
        }

        scores = []
        for train_idx, test_idx in splits:
            try:
                result = run_agt2i_fold(
                    dataset_name, target_col, layout,
                    trial.number % 1000,
                    train_idx, test_idx,
                    tabnet_params, cnn_params
                )
                scores.append(result["test"].get("auroc", 0.0))
            except Exception:
                # If the fold fails (e.g. too few samples), return a low score
                scores.append(0.0)
        return float(np.mean(scores))

    start = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - start

    best_roc_auc = study.best_value
    print(f"✅ AG‑T2I‑{layout:16s} on {dataset_name:20s}  →  best CV ROC-AUC: {best_roc_auc:.4f}  ({elapsed:.1f}s)")

    return dataset_name, f"AG-T2I-{layout}", study.best_params


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--workers", type=int, default=-1,
                    help="Number of parallel workers (default: all CPUs)")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--agt2i-trials", type=int, default=25)
    args = parser.parse_args()

    if args.dataset:
        datasets_to_run = [(ds, tgt) for ds, tgt in DATASETS if ds == args.dataset]
        if not datasets_to_run:
            print(f"Dataset '{args.dataset}' not found.")
            sys.exit(1)
    else:
        datasets_to_run = DATASETS

    # Fresh start
    if args.fresh:
        print("Fresh start requested – deleting existing best_params, Optuna studies, and TabNet cache...")
        for ds_name, _ in datasets_to_run:
            json_path = OUT_DIR / f"{ds_name}.json"
            if json_path.exists():
                json_path.unlink()
                print(f"  Removed {json_path}")
        for db_file in OPTUNA_DB_DIR.glob("optuna_study_*.db"):
            db_file.unlink()
            print(f"  Removed {db_file}")
        cache_tabnet = PROJECT_ROOT / "cache" / "tabnet_cache"
        if cache_tabnet.exists():
            shutil.rmtree(cache_tabnet)
            print(f"  Removed {cache_tabnet}")

    # -----------------------------------------------------------------
    # 0. Global preprocessing (once per dataset, before any parallel work)
    # -----------------------------------------------------------------
    print("Ensuring global preprocessed data exists for all datasets...")
    for ds_name, ds_target in datasets_to_run:
        ensure_global_preprocessing(ds_name, ds_target)
        print(f"  ✓ {ds_name}")

    should_tune = args.fresh or not all(
        (OUT_DIR / f"{ds_name}.json").exists() for ds_name, _ in datasets_to_run
    )
    best_per_dataset = {}

    if should_tune:
        # Merge baseline + AG‑T2I tasks into one list
        tasks = []
        for ds_name, ds_target in datasets_to_run:
            for model_name in MODELS:
                n_trials = TRIALS.get(model_name, 25)
                tasks.append(("baseline", ds_name, ds_target, model_name, n_trials))
            for layout in AGT2I_LAYOUTS:
                tasks.append(("agt2i", ds_name, ds_target, layout, args.agt2i_trials))

        n_jobs = args.workers if args.workers is not None else -1
        print(f"Tuning {len(tasks)} models (baseline + AG‑T2I) in parallel with {n_jobs if n_jobs > 0 else 'all'} workers...\n")

        def tune_task(task):
            typ, ds, tgt, name_or_layout, trials = task
            if typ == "baseline":
                return tune_single_model(ds, tgt, name_or_layout, trials)
            else:  # agt2i
                return tune_agt2i_layout(ds, tgt, name_or_layout, trials)

        start_time = time.time()
        all_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(tune_task)(task) for task in tasks
        )
        elapsed = time.time() - start_time
        print(f"\n🏁 All tuning finished in {elapsed/60:.1f} min.\n")

        for ds_name, model_name, params in all_results:
            if ds_name not in best_per_dataset:
                best_per_dataset[ds_name] = {}
            best_per_dataset[ds_name][model_name] = params

        # Save best_params per dataset
        for ds_name, params_dict in best_per_dataset.items():
            json_path = OUT_DIR / f"{ds_name}.json"
            with open(json_path, "w") as f:
                json.dump(params_dict, f, indent=4)
            print(f"Saved best_params/{ds_name}.json")
    else:
        print("✅ Using existing best_params files – tuning skipped.")

    # -----------------------------------------------------------------
    # 2. Benchmark
    # -----------------------------------------------------------------
    if not args.skip_benchmark:
        print("\n" + "=" * 60)
        print("STARTING BENCHMARK WITH TUNED PARAMETERS")
        print("=" * 60)

        hyper_results_dir = PROJECT_ROOT / "running_all_models" / "results_hyperparameter"
        hyper_results_dir.mkdir(parents=True, exist_ok=True)
        os.environ["RESULTS_DIR"] = str(hyper_results_dir)
        print(f"📁 Results will be saved to: {hyper_results_dir}")

        from running_all_models.benchmark_parallel import run_dataset_benchmark

        bench_start = time.time()
        for ds_name, ds_target in datasets_to_run:
            print(f"\n▶ Benchmarking {ds_name}...")
            run_dataset_benchmark(ds_name, ds_target, n_workers=args.workers)
        bench_elapsed = time.time() - bench_start
        print(f"\n🏁 Benchmark finished in {bench_elapsed/60:.1f} min.")
        print(f"Results saved to: {hyper_results_dir}")
    else:
        print("Skipping benchmark (--skip-benchmark was set).")