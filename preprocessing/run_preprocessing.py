# run_preprocessing.py
"""
Entry point for headless preprocessing execution.

This script is invoked by the Streamlit orchestrator (via subprocess) to
run the full preprocessing pipeline outside the UI thread. It loads the
configuration from environment variables (set by the Streamlit form), loads
the raw dataset, executes `run_preprocessing_pipeline()`, and then persists
the cleaned data as CSV and NumPy arrays.

The script prints detailed progress information to stdout, which can be
captured and displayed in the Streamlit logs, and exits with a non‑zero
status code if any step fails, ensuring failures are visible and debuggable.

**Role in the framework:**
It acts as the automated executor of the **Map** stage, guaranteeing that
the same preprocessing can be reproduced outside the Streamlit environment
(e.g., in a script or on a cluster) and that all artefacts are saved in a
standardised directory structure. This supports the paper’s emphasis on
**reproducibility** and **deterministic pipelines**.
"""

import os
import sys
import pandas as pd
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing.decisions import build_preprocessing_config_from_env, PreprocessingConfig
from preprocessing.pipeline import run_preprocessing_pipeline, PreprocessingPipeline

def main():
    """Main entry point for preprocessing."""
    print("=" * 60)
    print("TABNET PREPROCESSING PIPELINE")
    print("=" * 60)

    # Setup paths
    base_dir = Path(__file__).resolve().parent.parent
    dataset_name = os.environ.get("DATASET")
    if not dataset_name:
        raise ValueError("DATASET environment variable is not set.")
    print(f"Dataset: {dataset_name}")

    raw_path = base_dir / "data" / "raw" / f"{dataset_name}.csv"

    # AUDIT FIX: every caller of this script (benchmark_parallel.py,
    # hyperparameter_search.py) sets PROCESSED_DIR specifically to isolate
    # one fold/seed/layout's preprocessing output from every other
    # concurrently-running fold. This script previously ignored that
    # variable entirely and always wrote to the shared, dataset-level
    # data/processed/{DATASET}/ directory — meaning every fold's
    # preprocessing step would race to write the SAME file path under
    # Parallel(n_jobs=-1), and the guard check in run_agt2i_fold that reads
    # back from the isolated tmp_dir immediately afterward would find
    # nothing there. Mirrors the same override pattern already used by
    # train_tabnet.py's load_data().
    processed_env = os.environ.get("PROCESSED_DIR")
    processed_dir = Path(processed_env) if processed_env else (
        base_dir / "data" / "processed" / dataset_name
    )
    artifacts_dir = processed_dir / "artifacts"

    # Create directories
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print(f"Loading data from: {raw_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at: {raw_path}")

    df_raw = pd.read_csv(raw_path, sep=',')
    print(f"Raw data shape: {df_raw.shape}")

    # Build configuration from environment variables
    print("Building preprocessing configuration...")
    config = build_preprocessing_config_from_env()

    # Display configuration
    print("\n CONFIGURATION:")
    print(f"  • Target column: {config.target_column}")
    print(f"  • Features to remove: {len(config.features_to_remove)}")
    print(f"  • Missing threshold: {config.missing_threshold:.0%}")
    print(f"  • Numerical imputation: {config.numerical_missing_strategy}")
    print(f"  • Categorical handling: {config.categorical_missing_strategy}")
    print(f"  • Scaling: {config.scaling_strategy}")
    print(f"  • Config hash: {config.config_hash}")

    # Save configuration
    config.save(artifacts_dir)

    # ------------------------------------------------------------------
    # Check for custom split (e.g., for cross‑validation)
    # ------------------------------------------------------------------
    use_custom = os.environ.get("USE_CUSTOM_SPLIT", "false").lower() == "true"
    if use_custom:
        train_idx_path = os.environ.get("TRAIN_IDX_PATH")
        test_idx_path = os.environ.get("TEST_IDX_PATH")
        if not train_idx_path or not test_idx_path:
            raise ValueError(
                "USE_CUSTOM_SPLIT is true but TRAIN_IDX_PATH or TEST_IDX_PATH not set"
            )
        train_idx = np.load(train_idx_path)
        test_idx = np.load(test_idx_path)

        target = config.target_column
        df_train = df_raw.iloc[train_idx].reset_index(drop=True)
        df_test  = df_raw.iloc[test_idx].reset_index(drop=True)

        # Separate features and target for train and test
        X_train_raw = df_train.drop(columns=[target])
        y_train_raw = df_train[target]
        X_test_raw  = df_test.drop(columns=[target])
        y_test_raw  = df_test[target]

        # Use the presplit method – no internal split
        print("\n Starting preprocessing on custom train/test split...")
        pipeline = PreprocessingPipeline(config)
        X_train, X_test, y_train_enc, y_test_enc, metadata = pipeline.fit_presplit(
            X_train_raw, y_train_raw,
            X_test_raw, y_test_raw,
            artifacts_dir
        )

        # Build combined clean DataFrame for inspection
        df_train_clean = X_train.copy()
        df_train_clean[target] = y_train_enc
        df_test_clean = X_test.copy()
        df_test_clean[target] = y_test_enc
        df_clean = pd.concat([df_train_clean, df_test_clean], ignore_index=True)

        # Prepare a results dictionary compatible with later saving code
        results = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train_enc,
            'y_test': y_test_enc,
            'feature_names': pipeline.feature_names_,
            'pipeline': pipeline,
            'metadata': metadata
        }
    else:
        # Standard pipeline (internal train/test split)
        print("\n Starting preprocessing pipeline...")
        results = run_preprocessing_pipeline(config, df_raw, artifacts_dir)

        # Build combined dataset for inspection
        X_train = results['X_train']
        X_test = results['X_test']
        y_train = results['y_train']
        y_test = results['y_test']

        df_train_clean = X_train.copy()
        df_train_clean[config.target_column] = y_train
        df_test_clean = X_test.copy()
        df_test_clean[config.target_column] = y_test
        df_clean = pd.concat([df_train_clean, df_test_clean], ignore_index=True)

    # ------------------------------------------------------------------
    # Save processed data (common to both branches)
    # ------------------------------------------------------------------
    print("\n Saving processed data...")
    clean_data_path = processed_dir / "clean_data.csv"
    df_clean.to_csv(clean_data_path, index=False)

    # Use the final arrays from 'results'
    X_train = results['X_train']
    X_test = results['X_test']
    y_train = results['y_train']
    y_test = results['y_test']

    np.save(processed_dir / "X_train.npy", X_train.values.astype(np.float32) if isinstance(X_train, pd.DataFrame) else X_train.astype(np.float32))
    np.save(processed_dir / "X_test.npy", X_test.values.astype(np.float32) if isinstance(X_test, pd.DataFrame) else X_test.astype(np.float32))
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "y_test.npy", y_test)

    # Save feature names
    np.save(processed_dir / "feature_names.npy", results['feature_names'])

    # Final summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Original shape: {df_raw.shape}")
    print(f"Clean shape: {df_clean.shape}")
    print(f"Target classes: {len(np.unique(y_train))}")
    print(f"Transformers saved: {len(results['pipeline'].transformers)}")
    print(f"Artifacts directory: {artifacts_dir}")
    print(f"Clean data: {clean_data_path}")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n PREPROCESSING FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)