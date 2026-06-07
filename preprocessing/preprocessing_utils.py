#preprocessing_utils.py
"""
Streamlit‑specific utilities for displaying preprocessing artefacts.

This module separates the **presentation layer** from the preprocessing logic
to keep the UI code clean and the pipeline implementation agnostic of any
front‑end. It provides functions to:

  - Render a summary of the preprocessing metadata (feature counts, class
    distributions, configuration) inside Streamlit.
  - Load and preview the clean data after transformation.
  - Validate that all expected preprocessing outputs exist and are not corrupt.

**Role in the framework:**
While not part of the core Map–Optimize–Learn algorithm, this module supports
the **interpretability and reproducibility** goals of the paper by giving users
a direct, visual inspection of the preprocessing outcomes. It is the bridge
between the backend pipeline and the interactive application mentioned in the
paper’s introduction and conclusion.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any
import streamlit as st

def display_preprocessing_summary(processed_dir: Path) -> None:
    """Display preprocessing summary in Streamlit."""
    
    metadata_path = processed_dir / "preprocessing_metadata.json"
    if not metadata_path.exists():
        st.warning("Preprocessing metadata not found.")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    st.subheader("📝 Preprocessing Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Original Features", metadata['dataset']['original_features'])
    with col2:
        st.metric("Final Features", metadata['dataset']['final_features'])
    with col3:
        st.metric("Training Samples", metadata['dataset']['train_samples'])
    with col4:
        st.metric("Test Samples", metadata['dataset']['test_samples'])
    
    # Display configuration
    with st.expander("⚙️ Preprocessing Configuration"):
        config_path = processed_dir / "artifacts" / "preprocessing_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            st.json(config, expanded=False)
        else:
            st.info("Configuration file not found")
    
    # Display transformers summary
    with st.expander("🔧 Applied Transformations"):
        if 'transformers_summary' in metadata['preprocessing']:
            transformers = metadata['preprocessing']['transformers_summary']
            
            for name, details in transformers.items():
                st.write(f"**{name}** ({details['type']})")
                if 'dropped_columns' in details:
                    st.write(f"Dropped: {len(details['dropped_columns'])} columns")
                if 'scaled_columns' in details:
                    st.write(f"Scaled: {len(details['scaled_columns'])} columns")
                if 'encoded_columns' in details:
                    st.write(f"Encoded: {len(details['encoded_columns'])} columns")
                st.write("---")
        else:
            st.info("Transformer details not available")
    
    # Display class distribution
    with st.expander("🎯 Class Distribution"):
        if 'class_distribution' in metadata['dataset']:
            train_dist = metadata['dataset']['class_distribution']['train']
            test_dist = metadata['dataset']['class_distribution']['test']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Training Set**")
                st.write(pd.Series(train_dist))
            with col2:
                st.write("**Test Set**")
                st.write(pd.Series(test_dist))
        else:
            st.info("Class distribution not available")

def load_clean_data_preview(processed_dir: Path, n_rows: int = 5) -> pd.DataFrame:
    """Load and return preview of clean data."""
    clean_data_path = processed_dir / "clean_data.csv"
    if clean_data_path.exists():
        return pd.read_csv(clean_data_path, nrows=n_rows)
    return None

def get_preprocessing_artifacts(processed_dir: Path) -> Dict[str, Path]:
    """Get dictionary of preprocessing artifact paths."""
    artifacts = {}
    
    # Check for key files
    key_files = {
        'clean_data': 'clean_data.csv',
        'X_train': 'X_train.npy',
        'X_test': 'X_test.npy',
        'y_train': 'y_train.npy',
        'y_test': 'y_test.npy',
        'feature_names': 'feature_names.npy',
        'metadata': 'preprocessing_metadata.json',
        'config': 'artifacts/preprocessing_config.json'
    }
    
    for name, filename in key_files.items():
        path = processed_dir / filename
        if path.exists():
            artifacts[name] = path
    
    return artifacts

def validate_preprocessing_outputs(processed_dir: Path) -> Dict[str, Any]:
    """Validate that preprocessing outputs exist and are valid."""
    validation = {
        'is_valid': True,
        'missing_files': [],
        'warnings': [],
        'available_artifacts': []
    }
    
    # Required files
    required_files = [
        'clean_data.csv',
        'X_train.npy',
        'X_test.npy',
        'y_train.npy',
        'y_test.npy'
    ]
    
    for filename in required_files:
        path = processed_dir / filename
        if not path.exists():
            validation['missing_files'].append(filename)
            validation['is_valid'] = False
        else:
            validation['available_artifacts'].append(filename)
    
    # Check file sizes
    for filename in validation['available_artifacts']:
        path = processed_dir / filename
        if path.stat().st_size == 0:
            validation['warnings'].append(f"{filename} is empty")
    
    # Check numpy arrays can be loaded
    for npy_file in ['X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy']:
        path = processed_dir / npy_file
        if path.exists():
            try:
                data = np.load(path, allow_pickle=True)
                if data.size == 0:
                    validation['warnings'].append(f"{npy_file} contains no data")
            except Exception as e:
                validation['warnings'].append(f"Failed to load {npy_file}: {e}")
    
    return validation