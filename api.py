#api.py
"""
Simplified command‑line and Python API for the full Attention‑Guided
Tabular‑to‑Image pipeline (Map‑Optimize‑Learn).

This script serves as the **single entry point** for executing every stage
of the framework in a controlled, reproducible manner. It directly mirrors
the execution logic of the interactive Streamlit application, but can be
invoked from the terminal or imported as a library, ensuring that the exact
same processing, training, and evaluation code is used in all contexts.

Key capabilities
----------------
* **`run`** – Sequentially executes the complete pipeline for a given
  dataset, layout strategy, and hyperparameter set: preprocessing →
  TabNet training → image building → CNN training → evaluation →
  visualisation generation.
* **`random`** – Performs a random hyperparameter search over TabNet
  architecture and learning parameters, optionally in parallel across
  multiple CPU cores, to identify robust configurations.
* **`bayesian`** – Uses Optuna to perform Bayesian optimisation of
  TabNet and CNN hyperparameters, automatically guiding the search
  towards the best values of a chosen metric (accuracy, F1, etc.).
* **Persistent splits** – Creates and reuses stratified train/val/test
  splits, guaranteeing that different experiments on the same dataset
  are directly comparable and that no data leakage occurs.
* **Full compatibility** – Writes results to the same directory
  structure as the Streamlit dashboard (`data/processed/<dataset>/`
  and `experiments/`), so that numbers can be verified both interactively
  and programmatically.

How it fits in the Map–Optimize–Learn philosophy
-------------------------------------------------
- **Map:** Calls `run_preprocessing.py` to transform the raw data and
  `tabnet_image_builder.py` to project the processed features into
  attention‑guided images.
- **Optimize:** Triggers `train_tabnet.py`, which learns the supervised
  feature attention that defines the spatial layout.
- **Learn:** Launches `train_cnn.py` and `evaluate_cnn.py` to train the
  CNN on the fixed image representations and compute final metrics.
- **Interpretability:** Automatically invokes `mol_visualizations.py` to
  generate the qualitative plots used in the paper (visualisations of the
  AG‑T2I representations).

The script also enables the hyperparameter studies that informed the
experimental protocol (Section 5) by allowing systematic exploration of
layout strategies and learning parameters.

Reproducibility
---------------
All environment variables (dataset name, layout choice, TabNet/CNN
hyperparameters, random seed) are set before each subprocess, and the
global seed is fixed for Python, NumPy, PyTorch, and TensorFlow. Results
are persisted as JSON files, making it possible to re‑evaluate any
configuration without re‑running the entire pipeline.

Usage
-----
    # Single run
    python api.py run breast_cancer --target diagnosis --layout step_row --seed 42

    # Random search
    python api.py random breast_cancer --target diagnosis --trials 50 --jobs 4

    # Bayesian optimisation
    python api.py bayesian breast_cancer --target diagnosis --trials 50
"""

import os
import sys
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import concurrent.futures

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.trial import Trial
from sklearn.model_selection import StratifiedShuffleSplit
from execution.runner import run_step, PipelineStepError

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    os.environ['PYTHONHASHSEED'] = str(seed)

class SplitManager:
    def __init__(self, base_path: Path):
        self.splits_dir = base_path / "data" / "splits"
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    def get_split_path(self, dataset: str, split_id: str = "default") -> Path:
        split_dir = self.splits_dir / dataset / split_id
        split_dir.mkdir(parents=True, exist_ok=True)
        return split_dir

    def create_split(self, dataset: str, target_column: str, raw_data_path: Path,
                 split_id: str = "default", seed: int = 42,
                 test_size: float = 0.2, val_size: float = 0.2) -> Dict:
        df = pd.read_csv(raw_data_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Check if stratification is possible
        from collections import Counter
        class_counts = Counter(y)
        min_class_size = min(class_counts.values())
        use_stratify = min_class_size >= 2

        if use_stratify:
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_val_idx, test_idx = next(sss1.split(X, y))
            relative_val_size = val_size / (1 - test_size)
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
            train_idx, val_idx = next(sss2.split(X.iloc[train_val_idx], y.iloc[train_val_idx]))
            train_idx = train_val_idx[train_idx]
            val_idx = train_val_idx[val_idx]
        else:
            # Fallback to simple random split
            print(f"⚠️ Stratification not possible (smallest class has {min_class_size} sample). Using random split.")
            from sklearn.model_selection import train_test_split
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=None
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_size/(1-test_size),
                random_state=seed, stratify=None
            )
            # Get indices
            train_idx = X_train.index.to_numpy()
            val_idx = X_val.index.to_numpy()
            test_idx = X_test.index.to_numpy()

        self.save_split(dataset, train_idx, val_idx, test_idx, split_id)
        return {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx)
        }
    
    def save_split(self, dataset: str, train_idx: np.ndarray, val_idx: np.ndarray,
                   test_idx: np.ndarray, split_id: str = "default"):
        split_dir = self.get_split_path(dataset, split_id)
        np.save(split_dir / "train_idx.npy", train_idx)
        np.save(split_dir / "val_idx.npy", val_idx)
        np.save(split_dir / "test_idx.npy", test_idx)
        metadata = {
            "split_id": split_id,
            "dataset": dataset,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "created_at": datetime.now().isoformat()
        }
        with open(split_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load_split(self, dataset: str, split_id: str = "default") -> Optional[Dict]:
        split_dir = self.get_split_path(dataset, split_id)
        if not (split_dir / "train_idx.npy").exists():
            return None
        return {
            "train_idx": np.load(split_dir / "train_idx.npy"),
            "val_idx": np.load(split_dir / "val_idx.npy"),
            "test_idx": np.load(split_dir / "test_idx.npy"),
            "metadata": json.load(open(split_dir / "metadata.json"))
        }

    def get_split_env_vars(self, dataset: str, split_id: str = "default") -> Dict[str, str]:
        split_path = self.get_split_path(dataset, split_id)
        return {
            "SPLIT_DIR": str(split_path),
            "USE_PERSISTENT_SPLIT": "true",
            "SPLIT_ID": split_id
        }

class SimplePipelineAPI:
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = Path(base_path) if base_path else Path(__file__).resolve().parent

        # Directories – exactly as in app.py
        self.raw_data_dir = self.base_path / "data" / "raw"
        self.processed_base = self.base_path / "data" / "processed"
        self.preprocess_dir = self.base_path / "preprocessing"
        self.tabnet_dir = self.base_path / "tabnet_fs"
        self.image_dir = self.base_path / "image_builder"
        self.cnn_dir = self.base_path / "cnn"
        self.experiments_dir = self.base_path / "experiments"
        self.results_dir = self.experiments_dir / "results"
        self.mol_viz_base = self.experiments_dir / "mol_visualizations"
        self.hp_search_dir = self.experiments_dir / "hyperparameter_search"

        self.hp_search_dir.mkdir(parents=True, exist_ok=True)

        self.split_manager = SplitManager(self.base_path)

    # ------------------------------------------------------------------
    # Core pipeline run – mimics app.py's execution flow
    # ------------------------------------------------------------------
    def run_simple(
        self,
        dataset: str,
        target_column: str,
        mol_layout: str = "step_row",
        features_to_remove: Optional[List[str]] = None,
        tabnet_params: Optional[Dict[str, Any]] = None,
        layout_params: Optional[Dict[str, Any]] = None,
        cnn_params: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        reuse_existing: bool = True,
        quiet: bool = False,
        experiment_id: Optional[str] = None,
        split_id: str = "default",
        optimization_metric: str = "accuracy",
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        set_global_seed(seed)

        if experiment_id is None:
            param_hash = hashlib.md5(
                json.dumps({"layout": mol_layout, "seed": seed}, sort_keys=True).encode()
            ).hexdigest()[:8]
            experiment_id = f"{mol_layout}_seed{seed}_{param_hash}"

        processed_dir = self.processed_base / dataset
        processed_dir.mkdir(parents=True, exist_ok=True)

        env = {
            "DATASET": dataset,
            "TARGET_COL": target_column,
            "MOL_LAYOUT": mol_layout,
            "SEED": str(seed),
            "EXPERIMENT_ID": experiment_id,
            "OPTIMIZATION_METRIC": optimization_metric,
            "DROP_THRESHOLD": "0.5",
            "CAT_MISSING": "explicit",
            "NUM_MISSING": "median",
            "SCALING": "standard",
            "ENCODE_CATEGORICALS": "true",
            "FORMAT_STEP_DISTRIBUTION": "true",
        }
        env.update(self.split_manager.get_split_env_vars(dataset, split_id))

        if train_indices is not None and test_indices is not None:
            idx_dir = processed_dir / "custom_split"
            idx_dir.mkdir(parents=True, exist_ok=True)
            np.save(idx_dir / "train_idx.npy", train_indices)
            np.save(idx_dir / "test_idx.npy", test_indices)
            env["TRAIN_IDX_PATH"] = str(idx_dir / "train_idx.npy")
            env["TEST_IDX_PATH"] = str(idx_dir / "test_idx.npy")
            env["USE_CUSTOM_SPLIT"] = "true"

        if features_to_remove:
            env["FEATURES_TO_REMOVE"] = ",".join(features_to_remove)

        if tabnet_params:
            env.update({
                "TABNET_N_STEPS": str(tabnet_params.get('n_steps', 6)),
                "TABNET_STEP_DIM": str(tabnet_params.get('step_dim', 8)),
                "TABNET_ATTN_DIM": str(tabnet_params.get('attn_dim', 8)),
                "TABNET_GAMMA": str(tabnet_params.get('gamma', 1.5)),
                "TABNET_LAMBDA_SPARSE": str(tabnet_params.get('lambda_sparse', 1e-4)),
                "TABNET_MASK_TYPE": tabnet_params.get('mask_type', 'sparsemax'),
                "TABNET_LEARNING_RATE": str(tabnet_params.get('learning_rate', 2e-2)),
                "TABNET_BATCH_SIZE": str(tabnet_params.get('batch_size', 32)),
                "TABNET_MAX_EPOCHS": str(tabnet_params.get('max_epochs', 100)),
            })

        if cnn_params:
            env.update({
                "CNN_LEARNING_RATE": str(cnn_params.get('learning_rate', 1e-3)),
                "CNN_OPTIMIZER": cnn_params.get('optimizer', 'adam'),
                "CNN_EPOCHS": str(cnn_params.get('epochs', 50)),
                "CNN_BATCH_SIZE": str(cnn_params.get('batch_size', 32)),
                "CNN_DROPOUT": str(cnn_params.get('dropout', 0.3)),
            })

        if layout_params:
            if (mol_layout == "packed" or mol_layout == "packed_T") and 'target_width' in layout_params:
                env["PACKED_TARGET_WIDTH"] = str(layout_params['target_width'])
            elif mol_layout == "step_sparse" and 'columns_per_step' in layout_params:
                env["SPARSE_COLUMNS_PER_STEP"] = str(layout_params['columns_per_step'])

        processed_dir = self.processed_base / dataset
        processed_dir.mkdir(parents=True, exist_ok=True)
        tabnet_out = self.tabnet_dir / "outputs" / f"output_{dataset}"
        tabnet_out.mkdir(parents=True, exist_ok=True)
        cnn_models_dir = self.cnn_dir / "cnn_models"
        cnn_models_dir.mkdir(parents=True, exist_ok=True)
        mol_viz_dir = self.mol_viz_base / dataset
        mol_viz_dir.mkdir(parents=True, exist_ok=True)

        results_file = processed_dir / f"cnn_evaluation_results_{mol_layout}.json"

        if reuse_existing and results_file.exists():
            if not quiet:
                print(f"Using existing results for {dataset} ({mol_layout})")
            # Even when reusing, we need to return structured result; try to load train metrics too
            test_metrics = {}
            with open(results_file, 'r') as f:
                test_metrics = json.load(f)
            train_metrics = self._load_train_metrics(processed_dir, mol_layout, seed)
            return self._build_structured_result(mol_layout, seed, train_metrics, test_metrics)

        # Step 1: Preprocessing
        if not quiet:
            print("Step 1: Preprocessing...")
        success, output, _ = run_step(
            name="Preprocessing",
            script_path=self.preprocess_dir / "run_preprocessing.py",
            env_vars=env#,
            #timeout=600
        )
        if not success:
            error_msg = output[:500] if output else "Preprocessing failed"
            if not quiet:
                print(f"❌ Preprocessing failed: {error_msg}")
            return self._error_result(dataset, mol_layout, error_msg)

        # Step 2: TabNet training
        if not quiet:
            print("Step 2: TabNet training...")
        success, output, _ = run_step(
            name="TabNet Training",
            script_path=self.tabnet_dir / "train_tabnet.py",
            env_vars=env#,
            #timeout=900
        )
        if not success:
            error_msg = output[:500] if output else "TabNet training failed"
            if not quiet:
                print(f"❌ TabNet training failed: {error_msg}")
            return self._error_result(dataset, mol_layout, error_msg)

        # Step 3: Image building
        if not quiet:
            print("Step 3: Building images...")
        success, output, _ = run_step(
            name="Image Building",
            script_path=self.image_dir / "tabnet_image_builder.py",
            env_vars=env#,
            #timeout=300
        )
        if not success:
            error_msg = output[:500] if output else "Image building failed"
            if not quiet:
                print(f"❌ Image building failed: {error_msg}")
            return self._error_result(dataset, mol_layout, error_msg)

        # Step 4: CNN training
        if not quiet:
            print("Step 4: Training CNN...")
        success, output, _ = run_step(
            name="CNN Training",
            script_path=self.cnn_dir / "train_cnn.py",
            env_vars=env#,
            #timeout=600
        )
        if not success:
            error_msg = output[:500] if output else "CNN training failed"
            if not quiet:
                print(f"❌ CNN training failed: {error_msg}")
            return self._error_result(dataset, mol_layout, error_msg)

        # Step 5: CNN evaluation
        if not quiet:
            print("Step 5: Evaluating CNN...")
        success, output, _ = run_step(
            name="CNN Evaluation",
            script_path=self.cnn_dir / "evaluate_cnn.py",
            env_vars=env#,
            #timeout=300
        )
        if not success:
            error_msg = output[:500] if output else "CNN evaluation failed"
            if not quiet:
                print(f"❌ CNN evaluation failed: {error_msg}")
            return self._error_result(dataset, mol_layout, error_msg)

        # Step 6: MOL visualizations (optional)
        if not quiet:
            print("Step 6: Generating MOL visualizations...")
        try:
            run_step(
                name="MOL Visualization",
                script_path=self.image_dir / "mol_visualizations.py",
                env_vars=env#,
                #timeout=300
            )
        except Exception as e:
            if not quiet:
                print(f"⚠️ MOL visualizations failed (non‑critical): {e}")

        # ------------------------------------------------------------------
        # Read test and train metrics
        # ------------------------------------------------------------------
        test_metrics = {}
        if results_file.exists():
            with open(results_file, 'r') as f:
                test_metrics = json.load(f)

        train_metrics = self._load_train_metrics(processed_dir, mol_layout, seed)

        return self._build_structured_result(mol_layout, seed, train_metrics, test_metrics)
    
    def _load_train_metrics(self, processed_dir, mol_layout, seed):
        train_file = processed_dir / f"cnn_training_results_{mol_layout}_seed{seed}.json"
        if train_file.exists():
            with open(train_file, 'r') as f:
                return json.load(f)
        return {}

    def _build_structured_result(self, mol_layout, seed, train_metrics, test_metrics):
        return {
            "layout": mol_layout,
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
            "error": None
        }
    
    def _read_results(self, results_file: Path, mol_layout: str,
                      tabnet_params, layout_params, cnn_params, seed, optimization_metric):
        metrics = {"accuracy": 0.0, "balanced_accuracy": 0.0, "f1_score": 0.0, "auroc": 0.0}
        try:
            with open(results_file, 'r') as f:
                eval_results = json.load(f)
            metrics.update(eval_results)
        except Exception as e:
            print(f"⚠️ Warning: Could not read {results_file}: {e}", file=sys.stderr)

        return {
            "layout": mol_layout,
            "seed": seed,
            **metrics,
            f"optimized_{optimization_metric}": metrics.get(optimization_metric, metrics['accuracy']),
            "error": None
        }

    def _error_result(self, dataset, mol_layout, error):
        return {
            "dataset": dataset,
            "layout": mol_layout,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "f1_score": 0.0,
            "auroc": 0.0,
            "error": error
        }

    # ------------------------------------------------------------------
    # Random search (parallel, using top‑level function)
    # ------------------------------------------------------------------
    def random_search(
        self,
        dataset: str,
        target_column: str,
        layouts: Optional[List[str]] = None,
        n_trials: int = 50,
        param_distributions: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        quiet: bool = True,
        save_results: bool = True,
        split_id: str = "default",
        optimization_metric: str = "accuracy",
        n_jobs: int = 1
    ) -> pd.DataFrame:
        set_global_seed(seed)
        layouts = layouts or ["step_row", "packed", "packed_T", "step_sparse", 'attention_map']
        self._ensure_persistent_split(dataset, target_column, split_id, seed)

        if param_distributions is None:
            param_distributions = {
                "n_steps": {"type": "choice", "values": [3,4,5,6,7,8]},
                "step_dim": {"type": "choice", "values": [4,8,12,16,24,32]},
                "attn_dim": {"type": "choice", "values": [4,8,12,16,24,32]},
                "gamma": {"type": "uniform", "min": 1.0, "max": 2.0},
                "lambda_sparse": {"type": "loguniform", "min": 1e-5, "max": 1e-3},
                "learning_rate": {"type": "loguniform", "min": 1e-3, "max": 1e-1},
                "batch_size": {"type": "choice", "values": [16,32,64,128]},
            }

        def sample_param(dist):
            t = dist["type"]
            if t == "choice":
                return random.choice(dist["values"])
            elif t == "uniform":
                return random.uniform(dist["min"], dist["max"])
            elif t == "loguniform":
                return np.exp(random.uniform(np.log(dist["min"]), np.log(dist["max"])))
            else:
                return random.randint(dist["min"], dist["max"])

        trials = []
        for idx in range(n_trials):
            params = {name: sample_param(dist) for name, dist in param_distributions.items()}
            layout = random.choice(layouts)
            trials.append((idx, layout, params))

        print(f"\n{'='*60}")
        print(f"RANDOM SEARCH - {dataset}")
        print(f"Trials: {n_trials}, Jobs: {n_jobs}, Metric: {optimization_metric}")
        print(f"{'='*60}\n")

        if n_jobs > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = [
                    executor.submit(
                        _run_trial_standalone,
                        trial, dataset, target_column, seed, split_id,
                        optimization_metric, quiet, str(self.base_path)
                    )
                    for trial in trials
                ]
                results = [f.result() for f in futures]
        else:
            results = [
                _run_trial_standalone(trial, dataset, target_column, seed, split_id,
                                      optimization_metric, quiet, str(self.base_path))
                for trial in trials
            ]

        results = [r for r in results if r is not None]
        df = self._results_to_dataframe(results, optimization_metric)

        if save_results:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = self.hp_search_dir / f"random_search_{dataset}_{ts}.csv"
            df.to_csv(out_file, index=False)
            print(f"\n💾 Results saved to: {out_file}")

        self._display_top_results(df, optimization_metric)
        return df

    def _ensure_persistent_split(self, dataset, target_column, split_id, seed):
        if self.split_manager.load_split(dataset, split_id) is None:
            raw_path = self.raw_data_dir / f"{dataset}.csv"
            if not raw_path.exists():
                raise FileNotFoundError(f"Dataset not found: {raw_path}")
            print(f"Creating persistent split '{split_id}' for {dataset}...")
            self.split_manager.create_split(dataset, target_column, raw_path, split_id, seed)

    def _results_to_dataframe(self, results, metric):
        rows = []
        for r in results:
            row = {
                "layout": r.get("layout", ""),
                "accuracy": r.get("accuracy", 0.0),
                "balanced_accuracy": r.get("balanced_accuracy", 0.0),
                "f1_score": r.get("f1_score", 0.0),
                "auroc": r.get("auroc", 0.0),
                f"optimized_{metric}": r.get(metric, r.get("accuracy", 0.0)),
                "error": r.get("error", ""),
            }
            search_params = r.get("search_params", {})
            for k, v in search_params.items():
                row[f"param_{k}"] = v
            rows.append(row)
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(f"optimized_{metric}", ascending=False)
        return df

    def _display_top_results(self, df, metric, top_k=10):
        if df.empty:
            print("\nNo successful results.")
            return
        opt_col = f"optimized_{metric}"
        print(f"\n{'='*60}")
        print(f"TOP {min(top_k, len(df))} RESULTS (by {metric})")
        print(f"{'='*60}")
        for i in range(min(top_k, len(df))):
            row = df.iloc[i]
            print(f"\n{i+1}. {metric}: {row[opt_col]:.4f} ({row[opt_col]:.2%})")
            print(f"   Layout: {row['layout']}")
            print(f"   Accuracy: {row['accuracy']:.4f}, Balanced Acc: {row['balanced_accuracy']:.4f}")
            param_cols = [c for c in df.columns if c.startswith("param_")]
            if param_cols:
                print("   Parameters:")
                for col in param_cols[:5]:
                    pname = col.replace("param_", "")
                    pval = row[col]
                    if isinstance(pval, float):
                        print(f"     {pname}: {pval:.6f}")
                    else:
                        print(f"     {pname}: {pval}")
        print(f"\nSummary: Total={len(df)}, Successful={len(df[df['error'] == ''])}")
        print(f"Mean {metric}: {df[opt_col].mean():.4f}, Best: {df[opt_col].max():.4f}")

    # ------------------------------------------------------------------
    # Bayesian search (simplified, n_jobs=1)
    # ------------------------------------------------------------------
    def bayesian_search(
        self,
        dataset: str,
        target_column: str,
        layouts: Optional[List[str]] = None,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        seed: int = 42,
        quiet: bool = True,
        save_results: bool = True,
        study_name: Optional[str] = None,
        split_id: str = "default",
        optimization_metric: str = "accuracy",
        prune: bool = False,
        n_jobs: int = 1
    ) -> pd.DataFrame:
        if n_jobs != 1:
            print("⚠️ Bayesian search forced to n_jobs=1 (SQLite single‑process).")
            n_jobs = 1
        if prune:
            print("⚠️ Pruning is not implemented. Ignoring --prune flag.")

        set_global_seed(seed)
        layouts = layouts or ["step_row", "packed", "packed_T", "step_sparse", 'attention_map']
        self._ensure_persistent_split(dataset, target_column, split_id, seed)

        # Use SQLite storage – safe for single process and works on Windows without symlink privileges
        db_path = self.hp_search_dir / "optuna_study.db"
        storage_url = f"sqlite:///{db_path}"
        study_name = study_name or f"{dataset}_bayesian"

        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            print(f"Resuming existing study: {study_name}")
        except KeyError:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction="maximize",
                load_if_exists=True
            )
            print(f"Created new study: {study_name}")

        results = []

        def objective(trial: Trial) -> float:
            tabnet_params = {
                "n_steps": trial.suggest_int("n_steps", 2, 10),
                "step_dim": trial.suggest_int("step_dim", 4, 64, log=True),
                "attn_dim": trial.suggest_int("attn_dim", 4, 64, log=True),
                "gamma": trial.suggest_float("gamma", 1.0, 2.5),
                "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            }
            layout_idx = trial.suggest_categorical("layout", list(range(len(layouts))))
            layout = layouts[layout_idx]

            cnn_params = {
                "learning_rate": trial.suggest_float("cnn_lr", 1e-4, 1e-2, log=True),
                "optimizer": trial.suggest_categorical("cnn_optimizer", ["adam", "sgd", "rmsprop"]),
                "dropout": trial.suggest_float("cnn_dropout", 0.1, 0.6),
                "epochs": trial.suggest_int("cnn_epochs", 30, 100),
            }

            if not quiet:
                print(f"\n[Trial {trial.number}] Layout={layout}")

            result = self.run_simple(
                dataset=dataset,
                target_column=target_column,
                mol_layout=layout,
                tabnet_params=tabnet_params,
                cnn_params=cnn_params,
                reuse_existing=False,
                quiet=quiet,
                seed=seed + trial.number,
                split_id=split_id,
                optimization_metric=optimization_metric
            )

            result["search_method"] = "bayesian"
            result["search_params"] = {**tabnet_params, "layout": layout, **cnn_params}
            result["trial"] = trial.number
            results.append(result)

            metric_value = result.get(optimization_metric, result["test"]["accuracy"])
            if result.get('error'):
                if not quiet:
                    print(f"  ❌ Trial {trial.number} failed: {result['error'][:100]}")
                return 0.0
            else:
                if not quiet:
                    print(f"  ✅ {optimization_metric}: {metric_value:.4f}")
                return metric_value

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=1,
            show_progress_bar=not quiet
        )

        df = self._results_to_dataframe(results, optimization_metric)

        if save_results:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_csv = self.hp_search_dir / f"bayesian_search_{dataset}_{ts}.csv"
            df.to_csv(out_csv, index=False)
            best_params_file = self.hp_search_dir / f"best_params_{dataset}_{ts}.json"
            with open(best_params_file, 'w') as f:
                json.dump({
                    "best_value": study.best_value,
                    "best_params": study.best_params,
                    "best_trial": study.best_trial.number,
                    "optimization_metric": optimization_metric
                }, f, indent=2)
            print(f"\n💾 Results saved to {out_csv} and {best_params_file}")

        print(f"\nBest {optimization_metric}: {study.best_value:.4f}")
        return df

def _run_trial_standalone(trial_info, dataset, target_column, base_seed,
                          split_id, optimization_metric, quiet, base_path):
    trial_idx, layout, params = trial_info
    api = SimplePipelineAPI(base_path=Path(base_path))
    try:
        result = api.run_simple(
            dataset=dataset,
            target_column=target_column,
            mol_layout=layout,
            tabnet_params=params,
            reuse_existing=False,
            quiet=quiet,
            seed=base_seed + trial_idx,
            split_id=split_id,
            optimization_metric=optimization_metric
        )
        result["search_method"] = "random"
        result["search_params"] = params
        result["trial"] = trial_idx
        if not quiet and not result.get('error'):
            metric_val = result.get(optimization_metric, result["test"]["accuracy"])
            print(f"  ✅ Trial {trial_idx+1}: {optimization_metric}={metric_val:.4f}")
        return result
    except Exception as e:
        if not quiet:
            print(f"  ❌ Trial {trial_idx+1} failed: {e}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter Search for TabNet→CNN→MOL Pipeline")
    subparsers = parser.add_subparsers(dest='command', required=True)

    rand_parser = subparsers.add_parser('random', help='Random search')
    rand_parser.add_argument('dataset')
    rand_parser.add_argument('--target')
    rand_parser.add_argument('--trials', type=int, default=50)
    rand_parser.add_argument('--jobs', type=int, default=1)
    rand_parser.add_argument('--seed', type=int, default=42)
    rand_parser.add_argument('--metric', default='accuracy',
                             choices=['accuracy', 'balanced_accuracy', 'f1_score', 'auroc'])

    bayes_parser = subparsers.add_parser('bayesian', help='Bayesian search')
    bayes_parser.add_argument('dataset')
    bayes_parser.add_argument('--target')
    bayes_parser.add_argument('--trials', type=int, default=50)
    bayes_parser.add_argument('--seed', type=int, default=42)
    bayes_parser.add_argument('--metric', default='accuracy',
                              choices=['accuracy', 'balanced_accuracy', 'f1_score', 'auroc'])
    bayes_parser.add_argument('--no-prune', action='store_true')

    run_parser = subparsers.add_parser('run', help='Single run')
    run_parser.add_argument('dataset')
    run_parser.add_argument('--target')
    run_parser.add_argument('--layout', choices=['step_row', 'packed', "packed_T", 'step_sparse', 'attention_map'], default='step_row')
    run_parser.add_argument('--seed', type=int, default=42)
    run_parser.add_argument('--no-reuse', action='store_true', help='Do not reuse existing results')

    args = parser.parse_args()
    api = SimplePipelineAPI()

    def get_target():
        if hasattr(args, 'target') and args.target:
            return args.target
        raw_path = api.raw_data_dir / f"{args.dataset}.csv"
        if not raw_path.exists():
            print(f"Error: Dataset {raw_path} not found", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(raw_path, nrows=1)
        common = ['target', 'label', 'class', 'diagnosis', 'outcome', 'result']
        for col in df.columns:
            if col.lower() in common:
                print(f"Auto-detected target column: {col}")
                return col
        target = df.columns[-1]
        print(f"Auto-detected target column (last column): {target}")
        return target

    if args.command == 'random':
        api.random_search(
            dataset=args.dataset,
            target_column=get_target(),
            n_trials=args.trials,
            n_jobs=args.jobs,
            seed=args.seed,
            quiet=False,
            optimization_metric=args.metric
        )
    elif args.command == 'bayesian':
        api.bayesian_search(
            dataset=args.dataset,
            target_column=get_target(),
            n_trials=args.trials,
            seed=args.seed,
            quiet=False,
            optimization_metric=args.metric,
            prune=not args.no_prune
        )
    elif args.command == 'run':
        result = api.run_simple(
            dataset=args.dataset,
            target_column=get_target(),
            mol_layout=args.layout,
            seed=args.seed,
            reuse_existing=not args.no_reuse,
            quiet=False   # Show progress
        )
        if result.get('error'):
            print(f"\n❌ Pipeline failed: {result['error']}")
        else:
            print("\nResults:")

            print("\nTrain:")
            print(f"  Accuracy: {result['train']['accuracy']:.2%}")
            print(f"  Balanced Accuracy: {result['train']['balanced_accuracy']:.2%}")
            print(f"  F1 Macro: {result['train']['f1_macro']:.2%}")

            print("\nTest:")
            print(f"  Accuracy: {result['test']['accuracy']:.2%}")
            print(f"  Balanced Accuracy: {result['test']['balanced_accuracy']:.2%}")
            print(f"  F1 Macro: {result['test']['f1_macro']:.2%}")

            print(f"\nLayout: {result['layout']}")

if __name__ == "__main__":
    main()