"""
Benchmark script that compares tree ensembles, tabular deep models, and the
attention‑guided TabNet→CNN pipeline (AG‑T2I) across multiple datasets.

For each dataset, it performs 5‑fold CV for baselines and true fold‑wise
evaluation for AG‑T2I by passing the fold indices to the pipeline API
(with caching disabled to avoid re‑using stale results).

Extended metrics (balanced accuracy, precision, recall, F1, ROC‑AUC) are
computed for both train and test sets.  Misclassified samples and ROC
probability data are saved for further inspection.

PARALLEL VERSION: all models within a fold run concurrently via joblib.
"""

import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed

from api import SimplePipelineAPI
from running_all_models.metrics import compute_extended_metrics, get_wrong_cases
from running_all_models.utils import set_seed, mean_std_ci
from running_all_models.models_factory import get_models

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
N_SPLITS = 5
SEEDS = [0, 1, 2, 3, 4]

DATASETS = [
    ("Adult", "Class"),
    ("Bank", "Class"),
    ("Electricity", "Class"),
    ("Magic04", "Class"),
    ("Poker_Hand", "Class"),
    ("Forest_Cover_Type", "Class")
]

# ------------------------------------------------------------
# AG‑T2I helper – reuse_existing=False to avoid cached results
# ------------------------------------------------------------
def run_agt2i_fold(dataset, target, layout, seed, train_idx, test_idx):
    api = SimplePipelineAPI(base_path=PROJECT_ROOT)
    result = api.run_simple(
        dataset=dataset,
        target_column=target,
        mol_layout=layout,
        seed=seed,
        quiet=True,
        reuse_existing=False,
        train_indices=train_idx,
        test_indices=test_idx
    )
    return result

# ------------------------------------------------------------
# Single‑model runner (for parallel execution)
# ------------------------------------------------------------
def run_model_on_fold(model_name, dataset_name, target_col, seed, fold,
                      train_idx, test_idx, X_train_raw, X_test_raw,
                      y_train_fold, y_test_fold, le_classes, n_features, n_classes):
    """
    Run a single model (baseline or AG‑T2I) on one fold.
    Returns a list of result dictionaries (train and test rows).
    """
    results = []

    # ---- Helper to add NaN rows on error ----
    def add_nan_rows():
        for subset in ["train", "test"]:
            results.append({
                "model": model_name,
                "seed": seed,
                "fold": fold,
                "subset": subset,
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "precision_macro": np.nan,
                "recall_macro": np.nan,
                "f1_macro": np.nan,
                "precision_weighted": np.nan,
                "recall_weighted": np.nan,
                "f1_weighted": np.nan,
                "roc_auc": np.nan,
                "time_sec": np.nan
            })

    # ---- 1. Baseline models ----
    if model_name not in ["AG-T2I-step_row", "AG-T2I-packed", "AG-T2I-packed_T",
                          "AG-T2I-step_sparse", "AG-T2I-attention_map"]:
        try:
            model_dict = get_models(n_features, n_classes)
            model = model_dict[model_name]
            start_t = time.time()

            # Determine scaling
            model_needs_scaling = model_name in ["FT-Transformer (lite)", "IGTD-inspired",
                                                  "Naive Reshape", "TabNet"]
            if model_needs_scaling:
                imputer = SimpleImputer(strategy='median')
                X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns)
                X_test_imp  = pd.DataFrame(imputer.transform(X_test_raw), columns=X_test_raw.columns)
                scaler = StandardScaler()
                X_train_model = scaler.fit_transform(X_train_imp)
                X_test_model  = scaler.transform(X_test_imp)
            else:
                X_train_model = X_train_raw.values
                X_test_model = X_test_raw.values

            # Fit
            if model_name == "TabNet":
                model.fit(X_train_model, y_train_fold,
                          eval_set=[(X_test_model, y_test_fold)],
                          eval_metric=["accuracy"],
                          max_epochs=200, patience=20,
                          batch_size=16, virtual_batch_size=8,
                          drop_last=False)
            else:
                model.fit(X_train_model, y_train_fold)

            # Predict
            if hasattr(model, "predict_proba"):
                y_proba_train = model.predict_proba(X_train_model)
                y_pred_train = np.argmax(y_proba_train, axis=1)
                y_proba_test = model.predict_proba(X_test_model)
                y_pred_test = np.argmax(y_proba_test, axis=1)
            else:
                y_proba_train = None
                y_pred_train = model.predict(X_train_model)
                y_proba_test = None
                y_pred_test = model.predict(X_test_model)

            elapsed = time.time() - start_t

            # Metrics
            train_metrics = compute_extended_metrics(y_train_fold, y_pred_train, y_proba_train)
            test_metrics  = compute_extended_metrics(y_test_fold, y_pred_test, y_proba_test)

            for subset, met in [("train", train_metrics), ("test", test_metrics)]:
                results.append({
                    "model": model_name,
                    "seed": seed,
                    "fold": fold,
                    "subset": subset,
                    **met,
                    "time_sec": elapsed,
                })

            # Save ROC and misclassified (test only)
            if y_proba_test is not None:
                roc_df = pd.DataFrame(y_proba_test,
                                      columns=[f"prob_class_{i}" for i in range(y_proba_test.shape[1])])
                roc_df["true_label"] = y_test_fold
                roc_out = PROJECT_ROOT / "running_all_models" / "roc_data" / dataset_name / model_name
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
                wrong_out = PROJECT_ROOT / "running_all_models" / "misclassified" / dataset_name / model_name
                wrong_out.mkdir(parents=True, exist_ok=True)
                wrong_df.to_csv(wrong_out / f"seed{seed}_fold{fold}.csv", index=False)

        except Exception as e:
            print(f"[ERROR] {model_name} seed {seed} fold {fold}: {e}")
            add_nan_rows()

    # ---- 2. AG‑T2I variants ----
    else:
        layout = model_name.replace("AG-T2I-", "")
        try:
            start_t = time.time()
            result = run_agt2i_fold(dataset_name, target_col, layout, seed, train_idx, test_idx)
            elapsed = time.time() - start_t
            for subset in ["train", "test"]:
                sub = result.get(subset, {})
                results.append({
                    "model": model_name,
                    "seed": seed,
                    "fold": fold,
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
        except Exception as e:
            print(f"[ERROR] {model_name} seed {seed} fold {fold}: {e}")
            for subset in ["train", "test"]:
                results.append({
                    "model": model_name,
                    "seed": seed,
                    "fold": fold,
                    "subset": subset,
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "precision_macro": np.nan,
                    "recall_macro": np.nan,
                    "f1_macro": np.nan,
                    "precision_weighted": np.nan,
                    "recall_weighted": np.nan,
                    "f1_weighted": np.nan,
                    "roc_auc": np.nan,
                    "time_sec": np.nan
                })

    return results

# ------------------------------------------------------------
# Benchmark a single dataset (with model‑level parallelism)
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

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # List of all model names (baselines + AG‑T2I variants)
    model_names = list(get_models(n_features, n_classes).keys()) + [
        "AG-T2I-step_row", "AG-T2I-packed", "AG-T2I-packed_T",
        "AG-T2I-step_sparse", "AG-T2I-attention_map"
    ]

    all_results = []

    for seed in SEEDS:
        set_seed(seed)
        print(f"  Seed {seed}")
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded)):
            print(f"    Fold {fold} – preparing jobs...")

            X_train_raw = X.iloc[train_idx]
            X_test_raw = X.iloc[test_idx]
            y_train_fold = y_encoded[train_idx]
            y_test_fold = y_encoded[test_idx]

            # Build a list of jobs (each job is a tuple of arguments for run_model_on_fold)
            jobs = []
            for m in model_names:
                jobs.append((m, dataset_name, target_col, seed, fold,
                             train_idx, test_idx, X_train_raw, X_test_raw,
                             y_train_fold, y_test_fold, le_classes, n_features, n_classes))

            # Run all models for this fold in parallel
            fold_results = Parallel(n_jobs=n_workers, verbose=0)(
                delayed(run_model_on_fold)(*args) for args in jobs
            )

            # Flatten results (each job returns a list of dicts)
            for res_list in fold_results:
                all_results.extend(res_list)

            # Print a summary of completed models (optional)
            print(f"    Fold {fold} completed.")

    # Save raw results
    results_df = pd.DataFrame(all_results)
    out_dir = PROJECT_ROOT / "running_all_models" / "results"
    out_dir.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(out_dir / f"{dataset_name}_raw.csv", index=False)

    # Summary (only test subsets, average over seeds/folds)
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
                    "model": model,
                    "metric": metric,
                    "mean": mean_val,
                    "std": std_val,
                    "ci95": ci_val,
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
    parser.add_argument("--dataset", type=str, help="Dataset name to run (must match keys in DATASETS)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers for models within a fold. Default: use all CPUs.")
    args = parser.parse_args()

    if args.dataset:
        datasets_to_run = [(ds, tgt) for ds, tgt in DATASETS if ds == args.dataset]
        if not datasets_to_run:
            print(f"Dataset '{args.dataset}' not found. Available: {[ds for ds,_ in DATASETS]}")
            sys.exit(1)
    else:
        datasets_to_run = DATASETS

    print("=" * 60)
    print("STARTING BENCHMARK (PARALLEL MODELS PER FOLD)")
    print("=" * 60)
    for ds_name, ds_target in datasets_to_run:
        print(f"\n▶ Running {ds_name}...")
        run_dataset_benchmark(ds_name, ds_target, n_workers=args.workers)
    print("\n🏁 All benchmarks finished.")