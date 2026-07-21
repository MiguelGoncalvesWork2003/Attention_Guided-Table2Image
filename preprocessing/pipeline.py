#pipeline.py
"""
Main preprocessing pipeline with reproducibility and persistence.

This module implements the **Map** stage of the Map–Optimize–Learn philosophy
in its most concrete form: it consumes a `PreprocessingConfig` and applies a
series of scikit‑learn‑compatible, fittable transformers to the raw tabular
data, producing clean numerical arrays ready for TabNet training.

Key properties:
  1. **No data leakage:** The pipeline is fitted exclusively on the training
     split; the test set is only ever transformed.
  2. **Full persistence:** Every fitted transformer, the target encoder, the
     configuration, and the metadata are saved to disk so that the exact
     preprocessing can be reloaded or applied to new data.
  3. **Deterministic ordering:** Transformers are applied in a fixed, principled
     order (column removal → high‑missing drop → imputation → encoding →
     scaling) that respects data dependencies.
  4. **Metadata generation:** A comprehensive dictionary captures dataset
     dimensions, class distributions, feature names, and transformation
     summaries—essential for later inspection and for writing the experimental
     protocol.

The `PreprocessingPipeline` class is designed to be both used in interactive
Streamlit workflows and in headless scripts via `run_preprocessing_pipeline()`.

**Role in the framework:**
- It produces the standardised tabular representation on which TabNet is
  trained (the “map” step).
- By persisting all artefacts, it guarantees that the subsequent
  “optimize” (TabNet training) and “learn” (CNN training) stages are based on
  exactly the same preprocessing, enabling controlled experimentation with
  layout geometry.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import json

from .decisions import PreprocessingConfig
from .transform import (
    ColumnSelector, HighMissingDropper, SmartImputer,
    CategoricalEncoder, FeatureScaler, TargetLabelEncoder,
    save_transformer, load_transformer, create_transformation_summary
)
from sklearn.model_selection import train_test_split

class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for TabNet → CNN with persistence.
    
    Key features:
    1. Fits on training data only (no data leakage)
    2. Persists all transformers for reproducibility
    3. Handles train/test splits consistently
    4. Creates comprehensive metadata
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.transformers = {}
        self.metadata = {}
        self.feature_names_ = None
        self.target_encoder_ = None
        
    def fit_transform(self, 
                     X: pd.DataFrame, 
                     y: pd.Series,
                     artifacts_dir: Path) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Fit transformers on training data and transform both train/test.
        
        Returns:
            Tuple of (X_train_processed, X_test_processed, y_train_encoded, y_test_encoded, metadata)
        """
        print(f"🚀 Starting TabNet preprocessing pipeline for {self.config.dataset_name}")
        print(f"📊 Original shape: {X.shape}")
        print(f"🎯 Target: {self.config.target_column}")
        
        # Create artifacts directory
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Split data FIRST (avoid data leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.split_ratio,
            random_state=self.config.random_seed,
            stratify=y if y.nunique() < 10 else None  # Stratify for classification
        )
        
        print(f"📈 Train/Test split: {len(X_train)}/{len(X_test)} samples")
            
        # Step 1: Remove user-selected features
        if self.config.features_to_remove:
            feature_remover = ColumnSelector(
                columns=self.config.features_to_remove,
                drop=True
            )
            X_train = feature_remover.fit_transform(X_train)
            X_test = feature_remover.transform(X_test)
            self.transformers['feature_remover'] = feature_remover
            print(f"🗑️  Removed {len(self.config.features_to_remove)} user-selected features")
        
        # Step 2: Drop high missing columns
        missing_dropper = HighMissingDropper(threshold=self.config.missing_threshold)
        X_train = missing_dropper.fit_transform(X_train)
        X_test = missing_dropper.transform(X_test)
        self.transformers['missing_dropper'] = missing_dropper
        print(f"🧹 Dropped {len(missing_dropper.columns_to_drop_)} columns with high missing values")
        
        # Step 3: Impute missing values
        imputer = SmartImputer(
            numerical_strategy=self.config.numerical_missing_strategy,
            categorical_strategy=self.config.categorical_missing_strategy
        )
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        self.transformers['imputer'] = imputer
        print(f"🔄 Imputed missing values ({self.config.numerical_missing_strategy}/{self.config.categorical_missing_strategy})")
        
        # Step 4: Encode categorical variables (TabNet prefers label encoding)
        if self.config.encode_categoricals:
            encoder = CategoricalEncoder(encoding_strategy=self.config.encoding_strategy)
            X_train = encoder.fit_transform(X_train)
            X_test = encoder.transform(X_test)
            self.transformers['encoder'] = encoder
            print(f"🔤 Encoded {len(encoder.categorical_columns_)} categorical columns")
        
        # Step 5: Scale numerical features
        scaler = FeatureScaler(scaling_strategy=self.config.scaling_strategy)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        self.transformers['scaler'] = scaler
        print(f"⚖️  Scaled numerical features ({self.config.scaling_strategy})")
        
        # Step 6: Encode target - FIXED
        self.target_encoder_ = TargetLabelEncoder()
        y_train_encoded = self.target_encoder_.fit_transform(y_train)
        y_test_encoded = self.target_encoder_.transform(y_test)
        print(f"🎯 Encoded target variable ({y_train.nunique()} classes)")
        
        # Store feature names
        self.feature_names_ = X_train.columns.tolist()
        
        # Create metadata
        self.metadata = self._create_metadata(
            X_train, X_test, y_train, y_test,
            y_train_encoded, y_test_encoded
        )
        
        # Save artifacts
        self._save_artifacts(artifacts_dir)
        
        print(f"✅ Preprocessing completed. Final shape: {X_train.shape}")
        print(f"💾 Artifacts saved to: {artifacts_dir}")
        
        return X_train, X_test, y_train_encoded, y_test_encoded, self.metadata
    
    def fit_presplit(self,
                    X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series,
                    artifacts_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Fit transformers on the provided training data and transform both
        train and test sets. No internal splitting is performed.
        """
        print(f"🚀 Starting TabNet preprocessing pipeline for {self.config.dataset_name}")
        print(f"📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        print(f"🎯 Target: {self.config.target_column}")

        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Remove user-selected features
        if self.config.features_to_remove:
            feature_remover = ColumnSelector(
                columns=self.config.features_to_remove,
                drop=True
            )
            X_train = feature_remover.fit_transform(X_train)
            X_test = feature_remover.transform(X_test)
            self.transformers['feature_remover'] = feature_remover
            print(f"🗑️  Removed {len(self.config.features_to_remove)} user-selected features")

        # Step 2: Drop high missing columns
        missing_dropper = HighMissingDropper(threshold=self.config.missing_threshold)
        X_train = missing_dropper.fit_transform(X_train)
        X_test = missing_dropper.transform(X_test)
        self.transformers['missing_dropper'] = missing_dropper
        print(f"🧹 Dropped {len(missing_dropper.columns_to_drop_)} columns with high missing values")

        # Step 3: Impute missing values
        imputer = SmartImputer(
            numerical_strategy=self.config.numerical_missing_strategy,
            categorical_strategy=self.config.categorical_missing_strategy
        )
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        self.transformers['imputer'] = imputer
        print(f"🔄 Imputed missing values")

        # Step 4: Encode categorical variables
        if self.config.encode_categoricals:
            encoder = CategoricalEncoder(encoding_strategy=self.config.encoding_strategy)
            X_train = encoder.fit_transform(X_train)
            X_test = encoder.transform(X_test)
            self.transformers['encoder'] = encoder
            print(f"🔤 Encoded {len(encoder.categorical_columns_)} categorical columns")

        # Step 5: Scale numerical features
        scaler = FeatureScaler(scaling_strategy=self.config.scaling_strategy)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        self.transformers['scaler'] = scaler
        print(f"⚖️  Scaled numerical features ({self.config.scaling_strategy})")

        # Step 6: Encode target
        self.target_encoder_ = TargetLabelEncoder()
        y_train_encoded = self.target_encoder_.fit_transform(y_train)
        y_test_encoded = self.target_encoder_.transform(y_test)
        print(f"🎯 Encoded target variable ({y_train.nunique()} classes)")

        # Store feature names
        self.feature_names_ = X_train.columns.tolist()

        # Create metadata
        self.metadata = self._create_metadata(
            X_train, X_test, y_train, y_test,
            y_train_encoded, y_test_encoded
        )

        # Save artifacts
        self._save_artifacts(artifacts_dir)

        print(f"✅ Preprocessing completed.")
        print(f"💾 Artifacts saved to: {artifacts_dir}")

        return X_train, X_test, y_train_encoded, y_test_encoded, self.metadata

    def _create_metadata(self,
                        X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series,
                        y_train_encoded: np.ndarray,
                        y_test_encoded: np.ndarray) -> Dict[str, Any]:
        """Create comprehensive preprocessing metadata."""
        
        return {
            'dataset': {
                'name': self.config.dataset_name,
                'original_features': len(self.feature_names_) + len(self.config.features_to_remove),
                'final_features': len(self.feature_names_),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'target_classes': len(np.unique(y_train_encoded)),
                'class_distribution': {
                    'train': dict(pd.Series(y_train).value_counts().sort_index()),
                    'test': dict(pd.Series(y_test).value_counts().sort_index())
                }
            },
            'preprocessing': {
                'config_hash': self.config.config_hash,
                'transformers_summary': create_transformation_summary(self.transformers),
                'feature_names': self.feature_names_,
                'steps_applied': list(self.transformers.keys())
            },
            'statistics': {
                'train_mean': X_train.mean().to_dict() if not X_train.empty else {},
                'train_std': X_train.std().to_dict() if not X_train.empty else {},
                'test_mean': X_test.mean().to_dict() if not X_test.empty else {},
                'test_std': X_test.std().to_dict() if not X_test.empty else {},
            }
        }
    
    def _save_artifacts(self, artifacts_dir: Path):
        """Save all transformers and metadata."""
        
        # Save configuration
        self.config.save(artifacts_dir)
        
        # Save transformers
        transformers_dir = artifacts_dir / "transformers"
        transformers_dir.mkdir(exist_ok=True)
        
        for name, transformer in self.transformers.items():
            save_transformer(transformer, transformers_dir / f"{name}.pkl")
        
        # Save target encoder
        if self.target_encoder_:
            save_transformer(self.target_encoder_, transformers_dir / "target_encoder.pkl")
        
        # Save metadata
        metadata_path = artifacts_dir / "preprocessing_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        # Save feature names
        if self.feature_names_:
            np.save(artifacts_dir / "feature_names.npy", self.feature_names_)
    
    @classmethod
    def load_from_artifacts(cls, artifacts_dir: Path) -> "PreprocessingPipeline":
        """Load pipeline from saved artifacts."""
        
        # Load configuration
        config = PreprocessingConfig.load(artifacts_dir / "preprocessing_config.json")
        pipeline = cls(config)
        
        # Load transformers
        transformers_dir = artifacts_dir / "transformers"
        if transformers_dir.exists():
            for pkl_file in transformers_dir.glob("*.pkl"):
                name = pkl_file.stem
                pipeline.transformers[name] = load_transformer(pkl_file)
        
        # Load metadata
        metadata_path = artifacts_dir / "preprocessing_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                pipeline.metadata = json.load(f)
        
        # Load feature names
        feature_names_path = artifacts_dir / "feature_names.npy"
        if feature_names_path.exists():
            pipeline.feature_names_ = np.load(feature_names_path, allow_pickle=True).tolist()
        
        return pipeline
    
    def transform_new_data(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted transformers."""
        
        if not self.transformers:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")
        
        X_transformed = X_new.copy()
        
        # Apply transformers in order
        transformer_order = ['feature_remover', 'missing_dropper', 'imputer', 'encoder', 'scaler']
        
        for name in transformer_order:
            if name in self.transformers:
                X_transformed = self.transformers[name].transform(X_transformed)
        
        # Ensure correct feature order
        if self.feature_names_:
            X_transformed = X_transformed[self.feature_names_]
        
        return X_transformed

def run_preprocessing_pipeline(config: PreprocessingConfig,
                               df: pd.DataFrame,
                               artifacts_dir: Path) -> Dict[str, Any]:
    """
    High-level function to run the preprocessing pipeline.
    """
    import sys
    #print("DEBUG: Columns in loaded dataset:", list(df.columns), file=sys.stderr)   
    if config.target_column not in df.columns:
        raise ValueError(f"Target column '{config.target_column}' not found in dataset")

    X = df.drop(columns=[config.target_column])
    y = df[config.target_column]

    # Create and run pipeline (split happens inside fit_transform)
    pipeline = PreprocessingPipeline(config)
    X_train, X_test, y_train_encoded, y_test_encoded, metadata = pipeline.fit_transform(
        X, y, artifacts_dir
    )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train_encoded,
        'y_test': y_test_encoded,
        'feature_names': pipeline.feature_names_,
        'metadata': metadata,
        'pipeline': pipeline,
        'artifacts_dir': artifacts_dir
    }