#data_inspector.py
"""
Pure data inspection and descriptive analysis module.

This module is responsible for **characterising the raw tabular dataset without
any transformation, side effects, or data leakage**. It provides the structural
and statistical information needed by the downstream decision‑making stage
(Phase 2) to define a reproducible and model‑aware preprocessing configuration.

The inspection functions answer questions such as:
  - What are the column types, missing value distributions, and cardinalities?
  - Are there constant or high‑cardinality features that should be handled?
  - Is the dataset suitable for the TabNet→CNN pipeline (e.g., class balance,
    sample size)?

All outputs are dictionaries or pandas DataFrames that can be visualised in the
Streamlit interface or used programmatically to build the `PreprocessingConfig`.

**Role in the Map–Optimize–Learn pipeline:**
- **Map:**  Informs the mapping of raw tabular data into a clean numerical space
  by identifying which features require special treatment.
- **Optimize:**  Provides the evidence for heuristic decisions (e.g., drop
  thresholds, encoding strategies) so that the layout construction is grounded
  in the dataset’s actual properties rather than arbitrary defaults.

No modification of the input `DataFrame` is permitted in this module.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

def compute_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive statistics without modifying the data.
    
    Returns:
        Dictionary with dataset statistics
    """
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "missing_percentage": (df.isna().mean() * 100).round(2).to_dict(),
        "n_unique": df.nunique().to_dict(),
        "numeric_summary": df.select_dtypes(include=[np.number]).describe().transpose().to_dict() 
                         if df.select_dtypes(include=[np.number]).shape[1] > 0 else None,
        "categorical_summary": df.select_dtypes(include=['object', 'category']).describe().transpose().to_dict() 
                             if df.select_dtypes(include=['object', 'category']).shape[1] > 0 else None,
        "sample_rows": df.head(5).to_dict('records')
    }

def identify_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify column types without transformation.
    
    Returns:
        Dictionary with column type groups
    """
    return {
        "numerical": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "binary": [col for col in df.columns if df[col].nunique() == 2],
        "constant": [col for col in df.columns if df[col].nunique() == 1],
        "high_cardinality": [col for col in df.select_dtypes(include=['object', 'category']).columns 
                           if df[col].nunique() > 50]
    }

def compute_feature_metrics(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
    """
    Compute per-feature metrics for decision making.
    
    Args:
        df: Input dataframe
        target_col: Optional target column for correlation
        
    Returns:
        DataFrame with feature metrics
    """
    features = df.columns.tolist()
    if target_col and target_col in features:
        features.remove(target_col)
    
    metrics = []
    for feature in features:
        col_metrics = {
            'feature': feature,
            'dtype': str(df[feature].dtype),
            'missing_pct': (df[feature].isna().mean() * 100).round(2),
            'n_unique': df[feature].nunique(),
            'is_constant': df[feature].nunique() == 1,
        }
        
        # Add type-specific metrics
        if pd.api.types.is_numeric_dtype(df[feature]):
            col_metrics.update({
                'mean': df[feature].mean(),
                'std': df[feature].std(),
                'min': df[feature].min(),
                'max': df[feature].max(),
                'median': df[feature].median(),
                'skew': df[feature].skew() if df[feature].std() > 0 else 0,
            })
            
            if target_col and pd.api.types.is_numeric_dtype(df[target_col]):
                col_metrics['correlation'] = df[feature].corr(df[target_col])
        
        metrics.append(col_metrics)
    
    return pd.DataFrame(metrics)

def validate_dataset_for_ml_pipeline(df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
    """
    Validate dataset suitability for TabNet pipeline.
    
    Returns:
        Dictionary with validation results and recommendations
    """
    validation = {
        'is_valid': True,
        'warnings': [],
        'recommendations': []
    }
    
    # Check sample size
    if len(df) < 100:
        validation['warnings'].append('Small sample size (<100) may lead to unstable results')
        validation['recommendations'].append('Consider data augmentation or collect more samples')
    
    # Check class balance if target specified
    if target_col and target_col in df.columns:
        class_counts = df[target_col].value_counts()
        min_class_ratio = class_counts.min() / class_counts.sum()
        
        if min_class_ratio < 0.1:
            validation['warnings'].append(f'Class imbalance detected (minority class: {min_class_ratio:.1%})')
            validation['recommendations'].append('Consider stratified sampling or class weights')
    
    # Check for high cardinality categorical features
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    high_card_cols = [col for col in cat_cols if df[col].nunique() > 50]
    
    if high_card_cols:
        validation['warnings'].append(f'High cardinality features: {high_card_cols}')
        validation['recommendations'].append('Consider target encoding or dimensionality reduction')
    
    # Check for constant features
    constant_features = [col for col in df.columns if df[col].nunique() == 1]
    if constant_features:
        validation['warnings'].append(f'Constant features detected: {constant_features}')
        validation['recommendations'].append('Remove constant features')
    
    validation['is_valid'] = True
    validation['has_warnings'] = len(validation['warnings']) > 0
    
    return validation

def generate_inspection_report(df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
    """
    Generate comprehensive inspection report for decision phase.
    
    Returns:
        Complete inspection report
    """
    return {
        'statistics': compute_dataset_statistics(df),
        'column_types': identify_column_types(df),
        'feature_metrics': compute_feature_metrics(df, target_col),
        'validation': validate_dataset_for_ml_pipeline(df, target_col),
        'summary': {
            'n_samples': len(df),
            'n_features': len(df.columns) - (1 if target_col and target_col in df.columns else 0),
            'n_numerical': len(df.select_dtypes(include=[np.number]).columns),
            'n_categorical': len(df.select_dtypes(include=['object', 'category']).columns),
            'missing_total': df.isna().sum().sum(),
            'missing_percentage': (df.isna().sum().sum() / (len(df) * len(df.columns)) * 100).round(2),
        }
    }