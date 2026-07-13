"""hyperparameter_search.py – Parallel hyper‑parameter tuning (quiet) +
automatic sequential benchmark with tuned parameters.
Results go to results_hyperparameter/ (isolated from regular runs)."""

import os
import sys
import json
import time
import warnings
import logging
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed

# ---------- Silence everything ----------
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("joblib").setLevel(logging.WARNING)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from objective_functions import objective

# ---------- CONFIGURATION ----------
DATASETS = [
    #("Iris", "Class"),
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

OUT_DIR = Path(__file__).parent / "best_params"
OUT_DIR.mkdir(exist_ok=True, parents=True)
STUDY_DIR = Path(__file__).parent / "studies"
STUDY_DIR.mkdir(exist_ok=True, parents=True)


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
    """Worker: tune one model on one dataset. Returns (dataset_name, model_name, best_params)."""
    X_full, y_full, n_features, n_classes = load_and_prepare_dataset(dataset_name, target_col)
    X_tune, _, y_tune, _ = train_test_split(
        X_full, y_full, test_size=0.2, stratify=y_full, random_state=42
    )

    study_name = f"{dataset_name}_{model_name}"
    db_path = str(STUDY_DIR / f"{study_name}.db")
    storage_url = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
    )

    def wrapped_objective(trial):
        return objective(trial, model_name, X_tune, y_tune, n_features, n_classes)

    study.optimize(wrapped_objective, n_trials=n_trials, show_progress_bar=False)

    print(f"✅ {model_name:25s} on {dataset_name:20s}  →  best F1: {study.best_value:.4f}")

    return dataset_name, model_name, study.best_params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        help="Tune (and optionally benchmark) a single dataset")
    parser.add_argument("--model", type=str, default=None,
                        help="Tune a single model (requires --dataset)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers for tuning (and benchmark if run). "
                             "Default: all CPUs")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="Do not run the final benchmark after tuning")
    args = parser.parse_args()

    # Determine dataset list
    if args.dataset:
        datasets_to_run = [(ds, tgt) for ds, tgt in DATASETS if ds == args.dataset]
        if not datasets_to_run:
            print(f"Dataset '{args.dataset}' not found.")
            sys.exit(1)
    else:
        datasets_to_run = DATASETS

    models_to_run = MODELS if not args.model else [args.model]

    # Build flat list of tasks
    tasks = []
    for ds_name, ds_target in datasets_to_run:
        for model_name in models_to_run:
            n_trials = TRIALS.get(model_name, 25)
            tasks.append((ds_name, ds_target, model_name, n_trials))

    n_jobs = args.workers if args.workers is not None else -1
    print(f"Tuning {len(tasks)} (dataset, model) pairs with {n_jobs if n_jobs > 0 else 'all'} workers...\n")

    # ---------- Parallel tuning ----------
    start_time = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(tune_single_model)(ds, tgt, mdl, trials)
        for ds, tgt, mdl, trials in tasks
    )
    elapsed = time.time() - start_time
    print(f"\n🏁 Tuning finished in {elapsed/60:.1f} min.\n")

    # ---------- Save best_params ----------
    best_per_dataset = {}
    for ds_name, model_name, params in results:
        if ds_name not in best_per_dataset:
            best_per_dataset[ds_name] = {}
        best_per_dataset[ds_name][model_name] = params

    for ds_name, params_dict in best_per_dataset.items():
        json_path = OUT_DIR / f"{ds_name}.json"
        with open(json_path, "w") as f:
            json.dump(params_dict, f, indent=4)
        print(f"Saved best_params/{ds_name}.json")

        # ---------- Automatic benchmark (unless skipped) ----------
        if not args.skip_benchmark:
            print("\n" + "=" * 60)
            print("STARTING BENCHMARK WITH TUNED PARAMETERS")
            print("=" * 60)

            # ---- Direct output to a separate folder ----
            hyper_results_dir = PROJECT_ROOT / "running_all_models" / "results_hyperparameter"
            hyper_results_dir.mkdir(parents=True, exist_ok=True)
            os.environ["RESULTS_DIR"] = str(hyper_results_dir)
            print(f"📁 Results will be saved to: {hyper_results_dir}")

            # Import AFTER setting RESULTS_DIR, so that the benchmark module sees it
            from running_all_models.benchmark_parallel import run_dataset_benchmark

            bench_start = time.time()
            for ds_name, ds_target in datasets_to_run:
                print(f"\n▶ Benchmarking {ds_name}...")
                run_dataset_benchmark(ds_name, ds_target, n_workers=args.workers)
            bench_elapsed = time.time() - bench_start
            print(f"\n🏁 Benchmark finished in {bench_elapsed/60:.1f} min.")
            print(f"Results saved to: {hyper_results_dir}")