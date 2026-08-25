"""
*** NOT USED FOR ANY RESULT REPORTED IN THE THESIS — SEE WARNING BELOW ***

Benchmark script that compares tree ensembles, tabular deep models, and the
attention‑guided TabNet→CNN pipeline (AG‑T2I) across multiple datasets.

For each dataset, it performs 5‑fold CV for baselines and true fold‑wise
evaluation for AG‑T2I by passing the fold indices to the pipeline API
(with caching disabled to avoid re‑using stale results).

Extended metrics (balanced accuracy, precision, recall, F1, ROC‑AUC) are
computed for both train and test sets.  Misclassified samples and ROC
probability data are saved for further inspection.

--------------------------------------------------------------------------
AUDIT WARNING (do not remove without reading): this script is an earlier,
sequential predecessor to running_all_models/benchmark_parallel.py. It is
NOT listed in Appendix A.9's project directory structure (only
benchmark_parallel.py is), and its SEEDS = [0, 1, 2, 3, 4] (five seeds)
directly contradicts Section 5.5's explicit protocol: "model training and
evaluation are repeated with three fixed seeds (0, 1, 2)". It also drives
everything through api.py's SimplePipelineAPI rather than the direct
subprocess/caching orchestration in benchmark_parallel.py, so its
preprocessing and caching behaviour has not been cross-checked against
Chapter 6's numbers and should be assumed to differ.

Running this file instead of benchmark_parallel.py would silently produce
results computed under a different, undocumented protocol. Recommend either
deleting this file or moving it to an archive/ directory before submission,
so nobody inspecting the repository mistakes it for the script that
produced the reported results.
--------------------------------------------------------------------------
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
    # Return the full structured result (contains 'train' and 'test' dicts)
    return result

# ------------------------------------------------------------
# Benchmark a single dataset
# ------------------------------------------------------------
def run_dataset_benchmark(dataset_name, target_col):
    raw_path = PROJECT_ROOT / "data" / "raw" / f"{dataset_name}.csv"
    if not raw_path.exists():
        print(f"❌ Dataset not found: {raw_path}")
        return None, None

    df = pd.read_csv(raw_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    all_results = []

    for seed in SEEDS:
        set_seed(seed)
        print(f"  Seed {seed}")
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded)):
            X_train_raw = X.iloc[train_idx]
            X_test_raw = X.iloc[test_idx]
            y_train_fold = y_encoded[train_idx]
            y_test_fold = y_encoded[test_idx]

            # --------------------------------------------------------
            # Imputation + scaling for neural models
            # --------------------------------------------------------
            imputer = SimpleImputer(strategy='median')
            X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns)
            X_test_imputed  = pd.DataFrame(imputer.transform(X_test_raw), columns=X_test_raw.columns)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imputed)
            X_test_scaled  = scaler.transform(X_test_imputed)

            # Label encoder for decoding class names
            le_inner = LabelEncoder()
            le_inner.classes_ = le.classes_

            # ----- 1. Baseline models -----
            models_dict = get_models(X.shape[1], len(le.classes_))
            for model_name, model in models_dict.items():
                try:
                    start_t = time.time()

                    # Determine whether this model uses scaled or raw data
                    model_needs_scaling = model_name in [
                        "FT-Transformer (lite)", "MDS-layout", "Naive Reshape", "TabNet"
                    ]

                    X_train_model = X_train_scaled if model_needs_scaling else X_train_raw.values
                    X_test_model  = X_test_scaled if model_needs_scaling else X_test_raw.values

                    if model_name == "TabNet":
                        model.fit(
                            X_train_model, y_train_fold,
                            eval_set=[(X_test_model, y_test_fold)],
                            eval_metric=["accuracy"],
                            max_epochs=200, patience=20,
                            batch_size=16, virtual_batch_size=8,
                            drop_last=False
                        )
                    else:
                        model.fit(X_train_model, y_train_fold)

                    # Predict train & test
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

                    # Compute extended metrics
                    train_metrics = compute_extended_metrics(y_train_fold, y_pred_train, y_proba_train)
                    test_metrics = compute_extended_metrics(y_test_fold, y_pred_test, y_proba_test)

                    # Store results (one row for train, one for test)
                    for subset, met in [("train", train_metrics), ("test", test_metrics)]:
                        all_results.append({
                            "model": model_name,
                            "seed": seed,
                            "fold": fold,
                            "subset": subset,
                            **met,
                            "time_sec": elapsed,
                        })

                    print(f"    {model_name:20s} fold {fold}: "
                          f"train_acc={train_metrics['accuracy']:.3f}, "
                          f"test_acc={test_metrics['accuracy']:.3f}")

                    # Save ROC data for test set
                    if y_proba_test is not None:
                        roc_df = pd.DataFrame(y_proba_test,
                                             columns=[f"prob_class_{i}" for i in range(y_proba_test.shape[1])])
                        roc_df["true_label"] = y_test_fold
                        roc_out = PROJECT_ROOT / "running_all_models" / "roc_data" / dataset_name / model_name
                        roc_out.mkdir(parents=True, exist_ok=True)
                        roc_df.to_csv(roc_out / f"seed{seed}_fold{fold}.csv", index=False)

                    # Save misclassified samples (test set)
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
                    # Add NaN rows
                    for subset in ["train", "test"]:
                        all_results.append({
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

            # ----- 2. AG‑T2I variants (true CV) -----
            for layout in ["step_row", "packed", "packed_T", "step_sparse", "attention_map"]:
                try:
                    start_t = time.time()
                    result = run_agt2i_fold(
                        dataset_name, target_col, layout, seed,
                        train_idx, test_idx
                    )
                    elapsed = time.time() - start_t

                    # result now contains a structured dict with 'train' and 'test'
                    for subset in ["train", "test"]:
                        sub = result.get(subset, {})
                        all_results.append({
                            "model": f"AG-T2I-{layout}",
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
                    print(f"    AG-T2I-{layout:16s} fold {fold}: "
                          f"train_acc={result['train'].get('accuracy', np.nan):.3f}, "
                          f"test_acc={result['test'].get('accuracy', np.nan):.3f}")
                except Exception as e:
                    print(f"[ERROR] AG-T2I-{layout} seed {seed} fold {fold}: {e}")
                    # fallback NaN rows
                    for subset in ["train", "test"]:
                        all_results.append({
                            "model": f"AG-T2I-{layout}",
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
    args = parser.parse_args()

    if args.dataset:
        # Find the matching dataset in the list
        datasets_to_run = [(ds, tgt) for ds, tgt in DATASETS if ds == args.dataset]
        if not datasets_to_run:
            print(f"Dataset '{args.dataset}' not found. Available: {[ds for ds,_ in DATASETS]}")
            sys.exit(1)
    else:
        datasets_to_run = DATASETS   # run all if no argument

    print("=" * 60)
    print("STARTING BENCHMARK")
    print("=" * 60)
    for ds_name, ds_target in datasets_to_run:
        print(f"\n▶ Running {ds_name}...")
        run_dataset_benchmark(ds_name, ds_target)
    print("\n🏁 All benchmarks finished.")