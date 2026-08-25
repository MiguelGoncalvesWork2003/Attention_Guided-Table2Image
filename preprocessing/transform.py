#transform.py
"""
Fittable, serialisable transformation components.

This module provides a collection of custom scikit‑learn‑compatible transformers
that implement each atomic preprocessing step. Every transformer is designed to:

  - Be fitted on training data only and applied consistently to any future data.
  - Support pickle serialisation for persistence and reproducible reloading.
  - Handle edge cases (e.g., unseen categories, missing values) gracefully and
    with explicit warnings.
  - Expose internal state (e.g., `columns_to_drop_`, `numerical_columns_`) so
    that transformation summaries can be built automatically.

The included transformers are:
  - `ColumnSelector` – explicitly drop or keep a set of columns.
  - `HighMissingDropper` – drop columns whose missing‑value ratio exceeds a threshold.
  - `SmartImputer` – impute numerical and categorical columns with separate strategies.
  - `CategoricalEncoder` – label‑encode categorical features (preferred for TabNet).
  - `FeatureScaler` – apply standard, robust, or min‑max scaling to numerical features.
  - `TargetLabelEncoder` – encode the target variable (not a scikit‑learn
    transformer, but follows the same fit/transform paradigm).

Helper functions `save_transformer`, `load_transformer`, and
`create_transformation_summary` support the pipeline’s persistence and
metadata generation.

**Role in the framework:**
These transformers are the building blocks of the **Map** stage. By isolating
each transformation into a modular, testable, and serialisable unit, the
preprocessing pipeline remains transparent, auditable, and fully reproducible—
exactly as described in the paper’s experimental protocol.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, 
    LabelEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

# ============================================================================
# Custom Fittable Transformers
# ============================================================================

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select or drop specified columns."""
    
    def __init__(self, columns: List[str], drop: bool = False):
        self.columns = columns
        self.drop = drop
        self.selected_columns_ = None
    
    def fit(self, X: pd.DataFrame, y=None):
        if self.drop:
            self.selected_columns_ = [col for col in X.columns if col not in self.columns]
        else:
            self.selected_columns_ = self.columns
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_columns_]
    
    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Path) -> "ColumnSelector":
        with open(path, 'rb') as f:
            return pickle.load(f)

class HighMissingDropper(BaseEstimator, TransformerMixin):
    """Drop columns with high missing ratio."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.columns_to_drop_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        missing_ratio = X.isna().mean()
        self.columns_to_drop_ = missing_ratio[missing_ratio >= self.threshold].index.tolist()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns_to_drop_)

class SmartImputer(BaseEstimator, TransformerMixin):
    """Intelligent imputation with persistence."""
    
    def __init__(self, 
                 numerical_strategy: str = "median",
                 categorical_strategy: str = "explicit"):
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.numerical_imputer_ = None
        self.categorical_imputer_ = None
        self.numerical_columns_ = []
        self.categorical_columns_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        # Identify column types
        self.numerical_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create numerical imputer
        if self.numerical_columns_:
            self.numerical_imputer_ = SimpleImputer(strategy=self.numerical_strategy)
            self.numerical_imputer_.fit(X[self.numerical_columns_])
        
        # Create categorical imputer
        if self.categorical_columns_ and self.categorical_strategy != "drop":
            if self.categorical_strategy == "explicit":
                strategy = "constant"
                fill_value = "MISSING"
            else:
                strategy = "most_frequent"
                fill_value = None
            
            self.categorical_imputer_ = SimpleImputer(
                strategy=strategy, 
                fill_value=fill_value
            )
            self.categorical_imputer_.fit(X[self.categorical_columns_])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # Impute numerical columns
        if self.numerical_columns_ and self.numerical_imputer_ is not None:
            X[self.numerical_columns_] = self.numerical_imputer_.transform(
                X[self.numerical_columns_]
            )
        
        # Impute categorical columns
        if self.categorical_columns_:
            if self.categorical_strategy == "drop":
                X = X.drop(columns=self.categorical_columns_)
            elif self.categorical_imputer_ is not None:
                X[self.categorical_columns_] = self.categorical_imputer_.transform(
                    X[self.categorical_columns_]
                )
        
        return X

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, encoding_strategy: str = "label"):
        self.encoding_strategy = encoding_strategy
        self.label_encoders_ = {}
        self.ordinal_encoder_ = None
        self.categorical_columns_ = []
        # AUDIT FIX: per-column most-frequent training category, used to
        # replace unseen test categories (Section 5.2: "unseen test
        # categories are replaced by the most frequent training category").
        self.most_frequent_encoded_ = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        self.categorical_columns_ = X.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        if self.encoding_strategy == "label":
            for col in self.categorical_columns_:
                le = LabelEncoder()
                # Handle missing values by converting to string
                col_str = X[col].astype(str)
                le.fit(col_str)
                self.label_encoders_[col] = le
                # Record the most frequent training category's encoded value,
                # for unseen categories at transform time.
                most_frequent_value = col_str.mode(dropna=True)
                if len(most_frequent_value) > 0:
                    self.most_frequent_encoded_[col] = int(
                        le.transform([most_frequent_value.iloc[0]])[0]
                    )
                else:
                    self.most_frequent_encoded_[col] = 0
        elif self.encoding_strategy == "ordinal":
            self.ordinal_encoder_ = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            self.ordinal_encoder_.fit(X[self.categorical_columns_])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        if self.encoding_strategy == "label":
            for col, encoder in self.label_encoders_.items():
                if col in X.columns:
                    # Handle unseen categories
                    mapping = {cls: idx for idx, cls in enumerate(encoder.classes_)}

                    X_col = X[col].astype(str)

                    unseen_mask = ~X_col.isin(encoder.classes_)
                    if unseen_mask.any():
                        warnings.warn(
                            f"Unseen categories in {col}, encoding as the "
                            f"most frequent training category "
                            f"(Section 5.2)."
                        )

                    fallback = self.most_frequent_encoded_.get(col, 0)
                    X[col] = X_col.map(mapping).fillna(fallback).astype(int)
        
        elif self.encoding_strategy == "ordinal" and self.ordinal_encoder_ is not None:
            X[self.categorical_columns_] = self.ordinal_encoder_.transform(
                X[self.categorical_columns_]
            )
        
        return X

class FeatureScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, scaling_strategy: str = "standard"):
        self.scaling_strategy = scaling_strategy
        self.scaler_ = None
        self.numerical_columns_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        self.numerical_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not self.numerical_columns_ or self.scaling_strategy == "none":
            return self
        
        if self.scaling_strategy == "standard":
            self.scaler_ = StandardScaler()
        elif self.scaling_strategy == "robust":
            self.scaler_ = RobustScaler()
        elif self.scaling_strategy == "minmax":
            self.scaler_ = MinMaxScaler()
        
        if self.scaler_:
            self.scaler_.fit(X[self.numerical_columns_])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.numerical_columns_ or self.scaling_strategy == "none":
            return X
        
        X = X.copy()
        if self.scaler_:
            X[self.numerical_columns_] = self.scaler_.transform(
                X[self.numerical_columns_]
            )
        
        return X

class TargetLabelEncoder:
    """
    Simple target encoder (not a scikit-learn transformer).
    This only encodes the target variable, not features.
    """
    
    def __init__(self):
        self.encoder_ = None
        self.classes_ = None
        self.target_name_ = None
    
    def fit(self, y: pd.Series):
        """Fit encoder to target variable."""
        self.target_name_ = y.name if hasattr(y, 'name') else 'target'
        self.encoder_ = LabelEncoder()
        self.encoder_.fit(y)
        self.classes_ = self.encoder_.classes_
        return self
    
    def transform(self, y: pd.Series) -> np.ndarray:
        """Transform target variable."""
        try:
            return self.encoder_.transform(y)
        except ValueError:
            # If there are unseen labels, handle them by assigning -1
            # or map to existing classes
            y_series = pd.Series(y).copy()
            unseen_mask = ~y_series.isin(self.encoder_.classes_)
            y_series[unseen_mask] = self.encoder_.classes_[0]  # Map to first class
            return self.encoder_.transform(y_series)
    
    def fit_transform(self, y: pd.Series) -> np.ndarray:
        """Fit and transform target variable."""
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y_encoded: np.ndarray) -> pd.Series:
        """Inverse transform encoded target."""
        return pd.Series(
            self.encoder_.inverse_transform(y_encoded),
            name=self.target_name_
        )

# ============================================================================
# Utility Functions
# ============================================================================

def save_transformer(transformer: BaseEstimator, path: Path):
    """Save transformer to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(transformer, f)

def load_transformer(path: Path) -> BaseEstimator:
    """Load transformer from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def create_transformation_summary(transformers: Dict[str, BaseEstimator]) -> Dict[str, Any]:
    """Create summary of fitted transformers."""
    summary = {}
    
    for name, transformer in transformers.items():
        if hasattr(transformer, 'columns_to_drop_'):
            summary[name] = {
                'type': 'HighMissingDropper',
                'dropped_columns': transformer.columns_to_drop_,
                'threshold': transformer.threshold
            }
        
        elif hasattr(transformer, 'numerical_columns_'):
            if hasattr(transformer, 'scaling_strategy'):
                summary[name] = {
                    'type': 'FeatureScaler',
                    'strategy': transformer.scaling_strategy,
                    'scaled_columns': transformer.numerical_columns_
                }
            elif hasattr(transformer, 'numerical_strategy'):
                summary[name] = {
                    'type': 'SmartImputer',
                    'numerical_strategy': transformer.numerical_strategy,
                    'categorical_strategy': transformer.categorical_strategy,
                    'numerical_columns': transformer.numerical_columns_,
                    'categorical_columns': transformer.categorical_columns_
                }
        
        elif hasattr(transformer, 'label_encoders_'):
            summary[name] = {
                'type': 'CategoricalEncoder',
                'strategy': transformer.encoding_strategy,
                'encoded_columns': list(transformer.label_encoders_.keys())
            }
    
    return summary