"""
hyperparameter_search.py – Parallel hyper‑parameter tuning (quiet) +
automatic sequential benchmark with tuned parameters.
All models (baselines + AG‑T2I) are tuned concurrently.
Now uses 3‑fold CV for AG‑T2I as well.
"""

import os
os.environ["MPLBACKEND"] = "Agg"          # <-- set BEFORE any matplotlib import
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
# The 13 benchmarks of Table 5.1. Do not comment any of these out for a
# --fresh or filter-less run: Electricity, Gene, Soybean and Magic04 are
# reported as "completed" in Chapter 6 (Section 6.1) and any re-run (e.g.
# after a code fix) must still cover them or the thesis numbers silently
# stop being reproducible from this script.
# Poker_Hand and Forest_Cover_Type are the two datasets Section 6.1 lists as
# "still running" — see HPO_SUBSAMPLE_SIZE below before attempting these two.
# Iris is not part of the 13-dataset suite (Table 5.1) and must stay out.
DATASETS = [
    ("Cancer", "Class"),
    ("Card", "Class"),
    ("Diabetes", "Class"),
    ("Electricity", "Class"),
    ("Gene", "Class"),
    ("Glass", "Class"),
    ("Heart", "Class"),
    ("Horse", "Class"),
    ("Magic04", "Class"),
    ("Soybean", "Class"),
    ("Thyroid", "Class"),
    ("Poker_Hand", "Class"),
    ("Forest_Cover_Type", "Class"),
]

# Datasets large enough that the 3-fold inner HPO loop (baselines:
# RandomizedSearchCV; AGT2I: Optuna) needs a stratified subsample rather than
# the full table, per Section 5.5: "the hyper-parameter search is performed
# on a stratified subsample of [SUBSAMPLE SIZE] instances". That bracket is
# still an unfilled placeholder in the thesis text — the number below is a
# functional default only. Decide the final figure, set it here (or via the
# HPO_SUBSAMPLE_SIZE env var), then replace [SUBSAMPLE SIZE] in Section 5.5
# with the same number. The final benchmark (run_dataset_benchmark) always
# trains on the complete outer training fold regardless of this setting.
HPO_SUBSAMPLE_DATASETS = {"Poker_Hand"}#, "Forest_Cover_Type"}
HPO_SUBSAMPLE_SIZE = int(os.environ.get("HPO_SUBSAMPLE_SIZE", "20000"))

MODELS = [
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Random Forest",
    "MLP",
    "TabNet",
    "FT-Transformer (lite)",
    "MDS-layout",
    "IGTD",
    "Naive Reshape",
    "DeepInsight",       
]

TRIALS = {
    "XGBoost": 25,
    "LightGBM": 25,    
    "CatBoost": 25,
    "Random Forest": 25,
    "MLP": 25,
    "TabNet": 15,
    "FT-Transformer (lite)": 15,
    "MDS-layout": 25,
    "IGTD": 10,
    "Naive Reshape": 25,
    "DeepInsight": 25,     
}

AGT2I_TRIALS = 15
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

    # Section 5.5: HPO for the two largest benchmarks runs on a stratified
    # subsample; the final model (benchmark_parallel.py) is unaffected, since
    # it reads the raw CSV independently and never calls this function.
    if dataset_name in HPO_SUBSAMPLE_DATASETS and len(df) > HPO_SUBSAMPLE_SIZE:
        from sklearn.model_selection import train_test_split as _tts
        df, _ = _tts(
            df, train_size=HPO_SUBSAMPLE_SIZE, stratify=df[target_col],
            random_state=42
        )
        print(f"  [HPO subsample] {dataset_name}: using {len(df)}/"
              f"{len(pd.read_csv(raw_path))} stratified rows for tuning "
              f"(HPO_SUBSAMPLE_SIZE={HPO_SUBSAMPLE_SIZE})")

    X_df = df.drop(columns=[target_col])
    y = df[target_col]
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_df)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X_imp, y_enc, X_imp.shape[1], len(le.classes_)


def tune_single_model(dataset_name: str, target_col: str, model_name: str, n_trials: int):
    X_full, y_full, n_features, n_classes = load_and_prepare_dataset(dataset_name, target_col)

    # Exclude models that handle normalization internally (IGTD, DeepInsight)
    NEURAL_MODELS = {
        "TabNet", "FT-Transformer (lite)", "MLP",
        "MDS-layout", "Naive Reshape", "IGTD", "DeepInsight",
    }
    if model_name in NEURAL_MODELS:
        X_full = StandardScaler().fit_transform(X_full)

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
        # NOTE: key must be "iterations", not "n_estimators" — that is the
        # key get_model_from_params() reads back from the saved JSON
        # (models_factory.py). CatBoost's sklearn wrapper accepts
        # "n_estimators" too, as an alias, so RandomizedSearchCV would run
        # without error either way; only the *read-back* key must match, or
        # the tuned value is silently replaced by the untuned default.
        param_dist = {
            "iterations": randint(100, 300),
            "depth": randint(4, 8),
            "learning_rate": loguniform(0.01, 0.2),
            "l2_leaf_reg": uniform(1, 9),
        }
        base_model.set_params(thread_count=1)
    elif model_name == "Random Forest":
        param_dist = {
            "n_estimators": randint(100, 500),
            "max_depth": [None, 5, 10, 20],
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", None],
        }
    elif model_name == "MLP":
        param_dist = {
            "hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128)],
            "alpha": loguniform(1e-5, 1e-2),
            "learning_rate_init": loguniform(1e-4, 1e-2),
        }
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
            # TEMPORARY: reduced from 200. fit_params has no eval_set (it is
            # static across every fold, and RandomizedSearchCV's per-fold
            # validation split isn't available here to inject one), so
            # patience below likely has nothing to monitor and every fit
            # probably runs the full max_epochs regardless -- this measured
            # ~0.72s/epoch on Cancer (699 rows), i.e. ~108 minutes for 15
            # trials x 3 folds at max_epochs=200. Bounding this to 50 caps
            # the worst case at roughly a quarter of that while the
            # underlying missing-eval_set gap gets a proper fix (an internal
            # 80/20 split wrapper, matching T2I_CNN/FTTransformerWrapper).
            "max_epochs": 50, "patience": 10,
            "batch_size": 16, "virtual_batch_size": 8,
            "drop_last": False,
        }
    elif model_name == "FT-Transformer (lite)":
        param_dist = {
            "lr": loguniform(1e-4, 3e-3),
            "batch_size": [16, 32],
            "epochs": [50, 80],
        }
    elif model_name in ["MDS-layout", "Naive Reshape", "IGTD", "DeepInsight"]:
            param_dist = {
                "epochs": [50, 80, 100, 120],  
                "lr": loguniform(1e-4, 2e-3),          
                "dropout": uniform(0.0, 0.5),           
            }
    else:
        return dataset_name, model_name, {}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    from sklearn.metrics import make_scorer, roc_auc_score as _roc
    def _safe_auc(y_true, y_proba, **kw):
        try:
            y_proba = np.asarray(y_proba)
            y_true = np.asarray(y_true)
            if y_proba.ndim == 1:
                # sklearn's own scorer machinery collapses predict_proba to a
                # 1-D "probability of positive class" array whenever the
                # FITTED classifier only saw 2 classes. If the labels being
                # scored here still have more than 2 unique values (a small
                # or high-cardinality dataset's inner-CV fold losing a class
                # is the usual cause), a binary AUC genuinely cannot be
                # computed -- this is a degenerate, unscoreable fold, not an
                # error, so it returns 0.0 directly rather than letting
                # roc_auc_score raise "multi_class must be in ('ovo','ovr')".
                if len(np.unique(y_true)) > 2:
                    return 0.0
                return float(_roc(y_true, y_proba))
            nc = y_proba.shape[1]
            # nc is how many classes the FITTED model saw this fold, which
            # can be fewer than the dataset's true class count on small,
            # imbalanced datasets (e.g. Glass, rarest class ~9/214 rows) if
            # this particular inner-CV training split missed a class
            # entirely. If the validation labels contain a class the model
            # never saw, labels=list(range(nc)) below would silently exclude
            # it and roc_auc_score would raise -- this is a degenerate,
            # unscoreable fold (not a bug), named explicitly rather than
            # falling through to the generic except.
            if y_true.min() < 0 or y_true.max() >= nc:
                return 0.0
            if nc == 2:
                return _roc(y_true, y_proba[:, 1])
            return _roc(y_true, y_proba, multi_class="ovr", average="macro",
                        labels=list(range(nc)))
        except Exception:
            # Genuinely unexpected failure (not the degenerate-fold case
            # above, which is handled explicitly) -- worth seeing once,
            # without flooding the console on every occurrence.
            print(f"[_safe_auc] unexpected scoring failure, returning 0.0")
            return 0.0
    _scorer = make_scorer(_safe_auc, response_method="predict_proba")

    search = RandomizedSearchCV(
        base_model, param_distributions=param_dist,
        n_iter=n_trials, cv=cv, scoring=_scorer,
        n_jobs=1, random_state=42, verbose=0,
        error_score=0.0,
    )

    start = time.time()
    search.fit(X_full, y_full, **fit_params)
    elapsed = time.time() - start

    print(f"✅ {model_name:25s} on {dataset_name:20s}  →  best CV ROC-AUC: {search.best_score_:.4f}  ({elapsed:.1f}s)")
    return dataset_name, model_name, search.best_params_


def tune_agt2i_layout(dataset_name: str, target_col: str, layout: str, n_trials: int):
    """Tune a single AG‑T2I layout using 3‑fold cross‑validation (like baselines)."""
    X_full, y_full, _, n_classes = load_and_prepare_dataset(dataset_name, target_col)
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
                    tabnet_params, cnn_params, n_classes=n_classes
                )
                auroc = result["test"].get("auroc", 0.0)
                scores.append(0.0 if (auroc is None or np.isnan(auroc)) else float(auroc))
            except Exception as exc:
                print(f"    [trial {trial.number} fold warning] {exc}")
                scores.append(0.0)
        valid = [s for s in scores if not np.isnan(s) and s > 0.0]
        if not valid:
            raise optuna.TrialPruned()
        return float(np.mean(valid))

    completed = len(
        study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    )

    remaining = max(0, n_trials - completed)

    if remaining > 0:
        print(f"AG-T2I-{layout}: {completed}/{n_trials} trials completed. Running {remaining} more...")
        start = time.time()
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
        elapsed = time.time() - start
    else:
        print(f"AG-T2I-{layout}: already has {completed} completed trials. Skipping.")
        elapsed = 0

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
    parser.add_argument("--model", nargs='+', default=None,
                    help="One or more model names (space separated).")
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

    # Which model keys does this invocation need present in best_params?
    if args.model:
        requested_keys = list(args.model)
    else:
        requested_keys = MODELS + [f"AG-T2I-{l}" for l in AGT2I_LAYOUTS]

    def already_tuned(ds_name):
        p = OUT_DIR / f"{ds_name}.json"
        if not p.exists():
            return False
        try:
            with open(p) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False
        return all(k in existing for k in requested_keys)

    should_tune = args.fresh or not all(
        already_tuned(ds_name) for ds_name, _ in datasets_to_run
    )
    if not should_tune:
        print(f"✅ best_params already contain {requested_keys} for every dataset – tuning skipped.")
    best_per_dataset = {}

    if should_tune:
        tasks = []
        if args.model:
            for m in args.model:
                if m.startswith("AG-T2I-"):
                    layout = m.replace("AG-T2I-", "")
                    if layout not in AGT2I_LAYOUTS:
                        print(f"Unknown AG‑T2I layout: {layout}")
                        sys.exit(1)
                    for ds_name, ds_target in datasets_to_run:
                        tasks.append(("agt2i", ds_name, ds_target, layout, args.agt2i_trials))
                else:
                    if m not in MODELS:
                        print(f"Unknown model: {m}")
                        sys.exit(1)
                    for ds_name, ds_target in datasets_to_run:
                        n_trials = TRIALS.get(m, 25)
                        tasks.append(("baseline", ds_name, ds_target, m, n_trials))
        else:
            # No filter – add everything
            for ds_name, ds_target in datasets_to_run:
                for model_name in MODELS:
                    n_trials = TRIALS.get(model_name, 25)
                    tasks.append(("baseline", ds_name, ds_target, model_name, n_trials))
                for layout in AGT2I_LAYOUTS:
                    tasks.append(("agt2i", ds_name, ds_target, layout, args.agt2i_trials))

        def tune_task(task):
            typ, ds, tgt, name_or_layout, trials = task
            if typ == "baseline":
                return tune_single_model(ds, tgt, name_or_layout, trials)
            else:  # agt2i
                return tune_agt2i_layout(ds, tgt, name_or_layout, trials)

        # --- Fix: define n_jobs before Parallel ---
        n_jobs = args.workers   # -1 means use all CPUs (default)
        start_time = time.time()
        all_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
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
            existing = {}
            if json_path.exists():
                try:
                    with open(json_path) as f:
                        existing = json.load(f)
                except (json.JSONDecodeError, OSError):
                    existing = {}
            existing.update(params_dict)      # merge; never drop other models
            with open(json_path, "w") as f:
                json.dump(existing, f, indent=4)
            print(f"Saved best_params/{ds_name}.json "
                  f"({len(params_dict)} updated, {len(existing)} total)")
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
            run_dataset_benchmark(ds_name, ds_target, n_workers=args.workers,
                                  model_filter=args.model)
        bench_elapsed = time.time() - bench_start
        print(f"\n🏁 Benchmark finished in {bench_elapsed/60:.1f} min.")
        print(f"Results saved to: {hyper_results_dir}")
    else:
        print("Skipping benchmark (--skip-benchmark was set).")