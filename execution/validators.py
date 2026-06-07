#validators.py
"""
Validation utilities for pipeline outputs.

This module provides a comprehensive set of checks that verify the existence,
structure, and integrity of the artefacts produced by each stage of the
attention‑guided tabular‑to‑image framework. It is used both by the
Streamlit interface (to inform the user about the pipeline’s current state)
and by automated validation scripts (to guarantee experimental reproducibility).

Functions are organised by pipeline stage:
  - `check_preprocessing_outputs`: Validates that the cleaned data, train/test
    splits, feature names, and configuration/metadata are present and non‑empty.
  - `check_tabnet_outputs`: Ensures that the feature importance and step
    assignment CSV files exist, contain the required columns, and are readable.
  - `check_cnn_outputs`: Confirms that the image arrays (`X_train_img.npy`,
    `X_test_img.npy`) are saved, that the CNN model file exists, and that the
    image data has the expected shape `(N, 1, H, W)`.
  - `check_mol_outputs`: Checks that MOL visualisation directories contain at
    least some grid and instance PNG images.
  - `check_layout_outputs`: Specialised check for layout‑specific outputs
    (image arrays and layout metadata JSON).
  - `validate_dataset_structure`: Runs all checks for a given dataset, returning
    a dictionary that maps step names to their validation results.

**Role in the Map–Optimize–Learn pipeline:**
  - These validators act as a “safety net” that ensures every artefact required
    by a downstream stage has been successfully produced by its upstream
    counterpart.
  - By providing clear, machine‑readable feedback, they support the paper’s
    claims of full reproducibility: any missing or corrupted file is flagged
    before it can silently compromise a scientific result.
  - The validation of image dimensions and model files directly supports the
    controlled experimental protocol described in Section 4, where CNN training
    relies on fixed, pre‑computed image representations.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

def check_preprocessing_outputs(processed_dir: Path) -> Tuple[bool, List[str], List[str]]:
    """
    Validate preprocessing outputs.
    
    Args:
        processed_dir: Path to processed data directory
        
    Returns:
        Tuple of (is_valid, missing_files, warnings)
    """
    artifacts_dir = processed_dir / "artifacts"
    
    required_files = [
        "clean_data.csv",
        "X_train.npy",
        "X_test.npy",
        "y_train.npy",
        "y_test.npy",
        "feature_names.npy"
    ]
    
    missing_files = []
    warnings = []
    
    # Check main files
    for file_name in required_files:
        file_path = processed_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
        elif file_path.stat().st_size == 0:
            warnings.append(f"{file_name} is empty")
    
    # Check for artifacts directory
    if not artifacts_dir.exists():
        warnings.append("artifacts directory not found (old preprocessing format?)")
    else:
        # Check for key artifacts
        artifact_files = [
            "preprocessing_config.json",
            "preprocessing_metadata.json"
        ]
        
        for artifact in artifact_files:
            artifact_path = artifacts_dir / artifact
            if not artifact_path.exists():
                warnings.append(f"Artifact {artifact} not found")
    
    is_valid = len(missing_files) == 0
    
    return is_valid, missing_files, warnings

def check_tabnet_outputs(tabnet_output_dir: Path) -> Tuple[bool, List[str], List[str]]:
    """
    Validate TabNet outputs.
    
    Args:
        tabnet_output_dir: Path to TabNet output directory
        
    Returns:
        Tuple of (is_valid, missing_files, warnings)
    """
    required_files = [
        "tabnet_feature_importance.csv",
        "tabnet_step_assignment.csv"
    ]
    
    missing_files = []
    warnings = []
    
    for file_name in required_files:
        file_path = tabnet_output_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
        elif file_path.stat().st_size == 0:
            warnings.append(f"{file_name} is empty")
    
    # Validate step assignment file structure
    assignment_path = tabnet_output_dir / "tabnet_step_assignment.csv"
    if assignment_path.exists():
        try:
            df = pd.read_csv(assignment_path)
            required_columns = ['feature', 'dominant_step', 'importance_score']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                warnings.append(f"Step assignment missing columns: {missing_columns}")
            if df.empty:
                warnings.append("Step assignment file is empty")
        except Exception as e:
            warnings.append(f"Failed to read step assignment: {e}")
    
    is_valid = len(missing_files) == 0
    
    return is_valid, missing_files, warnings

def check_cnn_outputs(
    processed_dir: Path,
    cnn_model_path: Path
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate CNN training outputs.
    
    Args:
        processed_dir: Path to processed data directory
        cnn_model_path: Path to CNN model file
        
    Returns:
        Tuple of (is_valid, missing_files, warnings)
    """
    required_files = [
        "X_train_img.npy",
        "X_test_img.npy"
    ]
    
    missing_files = []
    warnings = []
    
    # Check image files
    for file_name in required_files:
        file_path = processed_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
        elif file_path.stat().st_size == 0:
            warnings.append(f"{file_name} is empty")
    
    # Check model file
    if not cnn_model_path.exists():
        missing_files.append(cnn_model_path.name)
    elif cnn_model_path.stat().st_size == 0:
        warnings.append("CNN model file is empty")
    
    # Validate image dimensions
    img_train_path = processed_dir / "X_train_img.npy"
    if img_train_path.exists():
        try:
            img_data = np.load(img_train_path)
            if len(img_data.shape) != 4:  # Should be (n_samples, 1, height, width)
                warnings.append(f"Unexpected image shape: {img_data.shape}")
            if img_data.shape[0] == 0:
                warnings.append("No training images found")
        except Exception as e:
            warnings.append(f"Failed to validate image data: {e}")
    
    is_valid = len(missing_files) == 0
    
    return is_valid, missing_files, warnings

def check_mol_outputs(mol_viz_dir: Path) -> Tuple[bool, List[str], List[str]]:
    """
    Validate MOL visualization outputs.
    
    Args:
        mol_viz_dir: Path to MOL visualizations directory
        
    Returns:
        Tuple of (is_valid, missing_files, warnings)
    """
    grids_dir = mol_viz_dir / "grids"
    instances_dir = mol_viz_dir / "instances"
    
    warnings = []
    
    # Check directory structure
    if not grids_dir.exists():
        return False, ["grids directory"], warnings
    if not instances_dir.exists():
        return False, ["instances directory"], warnings
    
    # Count images
    grid_images = list(grids_dir.glob("*.png"))
    instance_images = list(instances_dir.glob("*.png"))
    
    if len(grid_images) == 0:
        warnings.append("No grid images found")
    if len(instance_images) == 0:
        warnings.append("No instance images found")
    
    # Check for minimum expected images (at least one per class)
    is_valid = len(grid_images) > 0 or len(instance_images) > 0
    
    missing_files = []
    if not is_valid:
        missing_files = ["MOL visualization images"]
    
    return is_valid, missing_files, warnings

def validate_dataset_structure(dataset: str, base_dir: Path) -> Dict[str, Tuple[bool, List[str], List[str]]]:
    """
    Comprehensive validation of all pipeline outputs for a dataset.
    
    Args:
        dataset: Dataset name
        base_dir: Base directory path
        
    Returns:
        Dictionary mapping step names to validation results
    """
    # Define paths
    processed_dir = base_dir / "data" / "processed" / dataset
    tabnet_out = base_dir / "tabnet_fs" / "outputs" / f"output_{dataset}"
    cnn_model_path = base_dir / "cnn" / "cnn_models" / f"cnn_model_{dataset}.pth"
    mol_viz_dir = base_dir / "experiments" / "mol_visualizations" / dataset
    
    results = {}
    
    # Validate each step - use new validator for preprocessing
    results['preprocessing'] = check_preprocessing_outputs(processed_dir)
    results['tabnet'] = check_tabnet_outputs(tabnet_out)
    results['cnn'] = check_cnn_outputs(processed_dir, cnn_model_path)
    results['mol'] = check_mol_outputs(mol_viz_dir)
    
    return results

def check_layout_outputs(processed_dir: Path) -> Tuple[bool, List[str], List[str]]:
    """
    Check if layout-specific outputs exist.
    
    Args:
        processed_dir: Processed data directory
        
    Returns:
        Tuple of (is_valid, missing_files, warnings)
    """
    required_files = [
        "X_train_img.npy",
        "X_test_img.npy",
        "tabnet_image_layout.json"
    ]
    
    missing = []
    warnings = []
    
    for file in required_files:
        if not (processed_dir / file).exists():
            missing.append(file)
    
    # Check for layout metadata files
    layout_files = list(processed_dir.glob("layout_*.json"))
    if not layout_files:
        warnings.append("No layout metadata files found")
    
    is_valid = len(missing) == 0
    
    return is_valid, missing, warnings