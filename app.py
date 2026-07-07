#app.py
"""
Attention‑Guided Tabular‑to‑Image Pipeline – Interactive Dashboard.

Streamlit application for the deterministic attention-guided tabular-to-image
framework described in the paper. It provides a fully interactive, zero‑code
interface to execute every stage of the pipeline, from raw data inspection to
trained CNN evaluation and interpretable visualisations.

The dashboard functions as an **orchestration layer** that mirrors exactly
the logic of the command‑line API (`api.py`), but presents results in real
time and enables exploratory analysis of layout strategies and hyperparameters.
All heavy computation is delegated to the dedicated execution modules via
`execution.runner.run_step()`, ensuring that the dashboard, the CLI, and the
paper’s experiments share the same code base.

**Phases of the interactive workflow:**

1. **Data Loading & Inspection (Phase 1 & 2)**
   - Select an existing benchmark dataset or upload a new CSV.
   - Inspect raw statistics, missing values, class distribution.
   - Configure target column, features to remove, and preprocessing
     parameters (missing imputation, scaling, encoding).

2. **Pipeline Execution (Phase 3 – Map, Optimize, Learn)**
   - Choose a spatial layout strategy (`step_row`, `packed`, `packed_T`,
     `step_sparse`, `attention_map`) that defines how TabNet’s attention
     structure is projected to 2D.
   - Adjust TabNet hyperparameters (number of steps, attention dimension,
     sparsity, learning rate, etc.) directly in the UI.
   - Execute the full end‑to‑end pipeline with a single click:
       * **Map:** `run_preprocessing.py` → `tabnet_image_builder.py`
       * **Optimize:** `train_tabnet.py` (attention‑guided layout derivation)
       * **Learn:** `train_cnn.py` → `evaluate_cnn.py` (CNN training & test)
   - Optionally reuse existing preprocessing and TabNet outputs to speed up
     layout comparisons, while CNN models and visualisations are always
     regenerated for fairness.

3. **Results & Visualisations (Step 7)**
   - Displays accuracy, balanced accuracy, F1‑score, Cohen’s κ, and a full
     classification report – all computed by `evaluate_cnn.py` and loaded
     from the standard JSON results file.
   - Confusion matrix rendered as a Seaborn heatmap.
   - AG‑T2I image grids (per class, train & test) showing the actual pixel
     representations produced by the layout.
   - TabNet feature‑step assignment table with per‑step feature groups.
   - Download buttons for processed data, metrics, confusion matrices, and
     all generated plots.

**Design principles:**
- **No model or metric computation** inside the dashboard – it only
  coordinates existing scripts and displays their outputs.
- **Full compatibility** with the `SimplePipelineAPI`; any run from the
  dashboard is reproducible via `python api.py run`.
- **Stateful session management** ensures that dataset, target column,
  layout, and parameter choices persist across UI re‑renders.

This dashboard serves as both a demonstration tool for the paper’s
interpretability claims and a practical experimentation environment for
researchers exploring attention‑guided tabular‑to‑image transformations.
"""

import streamlit as st
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import io
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json

from execution.runner import run_step, PipelineStepError
from execution.validators import validate_dataset_structure

from preprocessing.preprocessing_utils import (
    display_preprocessing_summary,
    load_clean_data_preview,
    validate_preprocessing_outputs
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

BASE = Path(__file__).resolve().parent

DATA_DIR = BASE / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_BASE = DATA_DIR / "processed"

PREPROCESS_DIR = BASE / "preprocessing"
TABNET_DIR = BASE / "tabnet_fs"
IMAGE_DIR = BASE / "image_builder"
CNN_DIR = BASE / "cnn"

EXPERIMENTS_DIR = BASE / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
MOL_VIZ_BASE = EXPERIMENTS_DIR / "mol_visualizations"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="TabNet → CNN → MOL Pipeline",
    page_icon=None,
    layout="wide"
)

st.title("TabNet → CNN → MOL Pipeline Dashboard")
st.markdown("---")

# Sessioin Initialization
if 'dataset_loaded' not in st.session_state:
    st.session_state['dataset_loaded'] = False
if 'pipeline_run' not in st.session_state:
    st.session_state['pipeline_run'] = False
if 'dataset_changed' not in st.session_state:
    st.session_state['dataset_changed'] = False
if 'reuse_prev' not in st.session_state:
    st.session_state['reuse_prev'] = False
if 'preprocessing_params_set' not in st.session_state:
    st.session_state['preprocessing_params_set'] = False
if 'target_column' not in st.session_state:
    st.session_state['target_column'] = None
if 'preprocessing_params' not in st.session_state:
    st.session_state['preprocessing_params'] = {}
if 'last_dataset' not in st.session_state:
    st.session_state['last_dataset'] = None
if 'uploaded_dataset_name' not in st.session_state:
    st.session_state['uploaded_dataset_name'] = None
if 'target_column_history' not in st.session_state:
    st.session_state['target_column_history'] = {}
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = None
if 'dataset' not in st.session_state:
    st.session_state['dataset'] = None
if 'raw_path' not in st.session_state:
    st.session_state['raw_path'] = None
if 'selected_dataset' not in st.session_state:
    st.session_state['selected_dataset'] = None
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'mol_layout' not in st.session_state:
    st.session_state['mol_layout'] = 'step_row'
if 'layout_params' not in st.session_state:
    st.session_state['layout_params'] = {}
if 'seed' not in st.session_state:
    st.session_state['seed'] = 42
if 'train_cnn' not in st.session_state:
    st.session_state['train_cnn'] = True

def get_safe_dataset_key(dataset_name):
    """Convert dataset name to a safe key for session state"""
    safe_key = ''.join(c if c.isalnum() else '_' for c in dataset_name)
    if safe_key and not safe_key[0].isalpha():
        safe_key = 'ds_' + safe_key
    return safe_key

def check_existing_results(dataset, layout):
    """Check which pipeline components already have outputs for this dataset and layout"""
    validation_results = validate_dataset_structure(dataset, BASE)
    
    # CNN models are NEVER reused - always train new
    # MOL visualizations are NEVER reused - always generate new
    return {
        'preprocessing': validation_results['preprocessing'][0],
        'tabnet': validation_results['tabnet'][0],
        'cnn': False,  # NEVER reuse CNN
        'mol': False   # NEVER reuse MOL visualizations
    }

def save_uploaded_file(uploaded_file, raw_dir):
    """Save uploaded file to raw data directory"""
    path = raw_dir / uploaded_file.name
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

# PHASE 1: Dataset Selection & Upload
st.header("Phase 1: Dataset Selection & Loading")

AVAILABLE_DATASETS = [
    "Cancer", "Glass", "Iris", "Thyroid", "Diabetes", "Gene", "Soybean",
    "Heart", "Horse", "Forest_Cover_Type", "Poker_Hand"
]

data_source = st.radio(
    "Choose data source", 
    ["Use existing dataset", "Upload new dataset"],
    key="data_source_radio"
)

dataset = None
raw_path = None

if data_source == "Use existing dataset":
    selected_dataset = st.selectbox(
        "Select dataset", 
        AVAILABLE_DATASETS, 
        key="existing_dataset_select"
    )
    
    if 'selected_dataset' not in st.session_state or st.session_state['selected_dataset'] != selected_dataset:
        st.session_state['selected_dataset'] = selected_dataset
        st.session_state['dataset_changed'] = True
        st.session_state['dataset_loaded'] = False
        st.session_state['preprocessing_params_set'] = False
        st.session_state['pipeline_run'] = False
        st.session_state['uploaded_dataset_name'] = None
    else:
        st.session_state['dataset_changed'] = False
    
    dataset = selected_dataset
    raw_path = RAW_DATA_DIR / f"{dataset}.csv"

else:  # Upload new CSV
    uploaded_file = st.file_uploader(
        "Upload a CSV dataset", 
        type=["csv"], 
        key="file_uploader"
    )

    if uploaded_file is not None:
        dataset = Path(uploaded_file.name).stem

        candidate_path = RAW_DATA_DIR / uploaded_file.name

        if st.session_state.get('uploaded_dataset_name') != dataset:
            st.session_state['uploaded_dataset_name'] = dataset
            st.session_state['dataset_changed'] = True
            st.session_state['dataset_loaded'] = False
            st.session_state['preprocessing_params_set'] = False
            st.session_state['pipeline_run'] = False
            st.session_state['target_column'] = None

        st.session_state['reuse_prev'] = False

        if candidate_path.exists():
            st.warning("File already exists.")
            if not st.checkbox("Replace existing file?", key="replace_file"):
                st.stop()

        # Save uploaded file
        raw_path = save_uploaded_file(uploaded_file, RAW_DATA_DIR)
        st.success(f"Dataset '{uploaded_file.name}' saved.")

# Load dataset button
if dataset is not None and raw_path is not None and raw_path.exists():
    load_dataset_btn = st.button("Load Dataset", key="load_dataset")
    if load_dataset_btn:
        try:
            df_raw = pd.read_csv(raw_path)
            st.session_state['df_raw'] = df_raw
            st.session_state['dataset'] = dataset
            st.session_state['raw_path'] = str(raw_path)
            st.session_state['dataset_loaded'] = True
            st.session_state['preprocessing_params_set'] = False
            st.session_state['pipeline_run'] = False
            st.session_state['last_dataset'] = dataset
            st.success(f"Dataset '{dataset}' loaded successfully!")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")

# PHASE 2: Dataset Inspection & Preprocessing Configuration
if st.session_state['dataset_loaded']:
    st.header("Phase 2: Dataset Inspection & Preprocessing Configuration")
    
    # Get data from session state
    df_raw = st.session_state['df_raw']
    dataset = st.session_state['dataset']
    
    # Display raw data inspection
    st.subheader("Raw Data Inspection")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("**Dataset Preview:**")
    with col2:
        st.write(f"**Shape:** {df_raw.shape[0]} × {df_raw.shape[1]}")
    
    st.dataframe(df_raw.head(), width='stretch')
    
    st.subheader("Summary Statistics")
    with st.expander("View Summary Statistics"):
        st.dataframe(df_raw.describe(include="all").transpose(), width='stretch')
    
    st.subheader("Missing Values Analysis")
    missing_pct = df_raw.isna().mean().sort_values(ascending=False) * 100
    missing_df = missing_pct.to_frame("missing_%")
    missing_df = missing_df.T
    st.dataframe(missing_df, width='stretch')
    
    # Target column selection
    st.subheader("Target Column Selection")
    
    dataset_safe_key = get_safe_dataset_key(dataset)
    target_selector_key = f"target_col_{dataset_safe_key}"
    
    # Get default target column
    if dataset in st.session_state['target_column_history']:
        default_target = st.session_state['target_column_history'][dataset]
    elif st.session_state.get('target_column') and st.session_state['target_column'] in df_raw.columns:
        default_target = st.session_state['target_column']
    else:
        default_target = df_raw.columns[-1] if len(df_raw.columns) > 0 else df_raw.columns[0]
    
    try:
        default_idx = df_raw.columns.tolist().index(default_target)
    except ValueError:
        default_idx = 0
    
    target_col = st.selectbox(
        "Select target column for classification", 
        df_raw.columns.tolist(),
        index=default_idx,
        key=target_selector_key
    )
    
    # Store in session state
    st.session_state['target_column'] = target_col
    st.session_state['target_column_history'][dataset] = target_col
    
    # Display target distribution
    st.subheader("Target Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    target_counts = df_raw[target_col].value_counts()
    target_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    
    # Feature selection dropdown
    st.subheader("Feature Selection")
    
    # Get all feature columns (exclude target column)
    feature_columns = [col for col in df_raw.columns if col != target_col]
    
    if feature_columns:
        # Create multi-select for features to remove
        features_to_remove = st.multiselect(
            "Select features to remove from the dataset:",
            options=feature_columns,
            default=[],
            help="These features will be excluded from the machine learning pipeline"
        )
        
        # Show summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Features", len(feature_columns))
        with col2:
            st.metric("Features to Remove", len(features_to_remove))
        
        # Show what will be kept
        if features_to_remove:
            kept_features = [col for col in feature_columns if col not in features_to_remove]
            with st.expander(f"✓ Features that will be kept ({len(kept_features)})"):
                st.write(", ".join(kept_features))
            
            with st.expander(f"✗ Features that will be removed ({len(features_to_remove)})"):
                st.write(", ".join(features_to_remove))
        else:
            st.success("All features will be kept for analysis")
    else:
        st.warning("No features found (only target column exists)")
        features_to_remove = []
    
    # Reuse previous results option
    st.subheader("Reuse Previous Results")
    
    # Check existing results
    existing_results = check_existing_results(dataset, 'step_row')  # Default layout
    
    # Display what can be reused
    st.write("Previous results found for this dataset:")
    
    col1, col2 = st.columns(2)  # Only show preprocessing and TabNet
    with col1:
        st.metric("Preprocessing", "✓" if existing_results['preprocessing'] else "✗")
    with col2:
        st.metric("TabNet", "✓" if existing_results['tabnet'] else "✗")
    
    # Note about CNN and MOL
    st.info("Note: CNN models and MOL visualizations are never reused. They are always generated anew for each layout.")
    
    # Reuse checkbox
    if existing_results['preprocessing'] or existing_results['tabnet']:
        reuse_prev = st.checkbox(
            "Reuse existing preprocessing and TabNet results",
            value=st.session_state.get('reuse_prev', True),
            key="reuse_prev_checkbox"
        )
        st.session_state['reuse_prev'] = reuse_prev
        
        if reuse_prev:
            st.info("Will reuse preprocessing and TabNet results where available. CNN and MOL will always be regenerated.")
        else:
            st.info("Will regenerate all results from scratch.")
    else:
        st.info("No previous results found for this dataset.")
        st.session_state['reuse_prev'] = False
    
    # Preprocessing configuration
    st.subheader("Preprocessing Configuration")
    with st.form("preprocessing_config"):
        st.write("Configure preprocessing parameters:")
        
        drop_threshold = st.slider(
            "Drop columns with missing ratio ≥", 
            0.0, 1.0, 0.5, 0.05,
            key="drop_threshold"
        )
        
        cat_strategy = st.selectbox(
            "Categorical missing values handling",
            ["Treat as category", "Drop categorical columns"],
            key="cat_strategy"
        )
        
        num_strategy = st.selectbox(
            "Numerical missing values handling",
            ["Median", "Mean", "Zero"],
            key="num_strategy"
        )
        
        scaling_strategy = st.selectbox(
            "Feature scaling",
            ["Standard", "Robust", "MinMax", "None"],
            index=0,
            key="scaling_strategy"
        )
        
        submitted = st.form_submit_button("Save Preprocessing Configuration")
        
        if submitted:
            st.session_state['preprocessing_params'] = {
                'target_col': target_col,
                'drop_threshold': drop_threshold,
                'cat_strategy': cat_strategy,
                'num_strategy': num_strategy,
                'scaling_strategy': scaling_strategy,
                'features_to_remove': features_to_remove,
                'mol_layout': st.session_state.get('mol_layout', 'step_row'),
                'layout_params': st.session_state.get('layout_params', {}),
                'reuse_prev': st.session_state.get('reuse_prev', False),
                'seed': st.session_state.get('seed', 42)
            }
            st.session_state['preprocessing_params_set'] = True
            
            # Store in environment variables
            os.environ["TARGET_COL"] = target_col
            os.environ["DROP_THRESHOLD"] = str(drop_threshold)
            os.environ["CAT_MISSING"] = "explicit" if cat_strategy == "Treat as category" else "drop"
            os.environ["NUM_MISSING"] = num_strategy.lower()
            os.environ["SCALING"] = scaling_strategy.lower()
            os.environ["ENCODE_CATEGORICALS"] = "true"  # Always true for TabNet
            
            # Store features to remove in environment variable (as comma-separated string)
            features_to_remove_str = ",".join(features_to_remove) if features_to_remove else ""
            os.environ["FEATURES_TO_REMOVE"] = features_to_remove_str
            
            # Store layout in environment
            os.environ["MOL_LAYOUT"] = st.session_state.get('mol_layout', 'step_row')
            
            # Store seed (CNN always trained fresh)
            os.environ["SEED"] = str(st.session_state.get('seed', 42))
            
            st.success("Preprocessing configuration saved!")
        
# PHASE 3: Pipeline Execution
if st.session_state['preprocessing_params_set']:
    st.header("Phase 3: Pipeline Execution")
    
    # Layout Configuration Section
    st.subheader("Layout Strategy Configuration")
    
    # Layout selection
    available_layouts = ["step_row", "packed", "packed_T", "step_sparse", "attention_map"]
    current_layout = st.session_state.get('mol_layout', 'step_row')
    selected_layout = st.selectbox(
        "Select layout strategy",
        options=available_layouts,
        index=available_layouts.index(current_layout) if current_layout in available_layouts else 0,
        help="How TabNet's learned structure is mapped to 2D images"
    )

    # Layout parameters
    layout_params = st.session_state.get('layout_params', {})
    
    if selected_layout == "packed":
        target_width = st.slider(
            "Target width",
            min_value=1,
            max_value=64,
            value=layout_params.get('target_width', 16),
            help="Width of the packed grid"
        )
        layout_params['target_width'] = target_width
        
    elif selected_layout == "step_sparse":
        columns_per_step = st.slider(
            "Columns per step",
            min_value=1,
            max_value=32,
            value=layout_params.get('columns_per_step', 10),
            help="Fixed number of columns allocated to each step"
        )
        layout_params['columns_per_step'] = columns_per_step

    # TabNet Configuration Section
    st.subheader("TabNet Configuration")
    with st.expander("TabNet Parameters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tabnet_n_steps = st.slider(
                "Number of steps (n_steps)",
                min_value=1,
                max_value=10,
                value=6,
                help="Number of sequential decision steps in TabNet"
            )
            
            tabnet_step_dim = st.slider(
                "Step dimension (n_d)",
                min_value=4,
                max_value=64,
                value=8,
                help="Dimension of decision layer in each step"
            )
            
            tabnet_attn_dim = st.slider(
                "Attention dimension (n_a)",
                min_value=4,
                max_value=64,
                value=8,
                help="Dimension of attention layer in each step"
            )
            
        with col2:
            tabnet_gamma = st.slider(
                "Gamma (feature reuse)",
                min_value=1.0,
                max_value=2.0,
                value=1.5,
                step=0.1,
                help="Controls feature reuse across steps"
            )
            
            tabnet_lambda_sparse = st.number_input(
                "Sparsity lambda",
                min_value=0.0,
                max_value=0.01,
                value=1e-4,
                format="%e",
                help="Sparsity regularization weight"
            )
            
            tabnet_mask_type = st.selectbox(
                "Mask type",
                ["sparsemax", "entmax"],
                index=0,
                help="Type of masking function"
            )
            
        with col3:
            tabnet_learning_rate = st.number_input(
                "Learning rate",
                min_value=1e-5,
                max_value=0.1,
                value=2e-2,
                format="%e",
                help="Learning rate for optimization"
            )
            
            tabnet_batch_size = st.selectbox(
                "Batch size",
                [16, 32, 64, 128],
                index=1,
                help="Training batch size"
            )
            
            tabnet_max_epochs = st.slider(
                "Max epochs",
                min_value=10,
                max_value=500,
                value=100,
                help="Maximum training epochs"
            )
            
        # Store TabNet parameters in session state
        st.session_state['tabnet_params'] = {
            'n_steps': tabnet_n_steps,
            'step_dim': tabnet_step_dim,
            'attn_dim': tabnet_attn_dim,
            'gamma': tabnet_gamma,
            'lambda_sparse': tabnet_lambda_sparse,
            'mask_type': tabnet_mask_type,
            'learning_rate': tabnet_learning_rate,
            'batch_size': tabnet_batch_size,
            'max_epochs': tabnet_max_epochs
        }

    # Detailed explanations
    with st.expander("Understanding TabNet Configuration"):
        st.markdown("""
        ### Key TabNet Parameters
        
        TabNet learns **which features matter and when** using sequential attention.
        The learned feature-step assignments are later reused to construct CNN images.
        
        1. **n_steps** (default: 5):
        - Number of sequential decision steps
        - Each step attends to a different subset of features
        - More steps = more capacity, but slower training
        
        2. **step_dim (n_d)** (default: 8):
        - Decision representation size
        - Controls model capacity per step
        - Larger = more expressive, but more parameters
        
        3. **attn_dim (n_a)** (default: 8):
        - Attention representation size
        - Controls how selectively features are weighted
        - Larger = more selective feature attention
        
        4. **gamma** (default: 1.5):
        - Feature reuse coefficient
        - Higher values allow features to appear in multiple steps
        - Higher = features can be used in multiple steps
        - Lower = each feature used in fewer steps
        
        5. **lambda_sparse** (default: 1e-4):
        - Sparsity regularization
        - Encourages compact and interpretable feature selection
        - Higher = more sparse, fewer features used
        
        6. **mask_type** (default: "sparsemax"):
        - "sparsemax": hard, sparse feature selection
        - "entmax": smoother, more distributed attention
        """)

    # Store layout settings
    st.session_state['mol_layout'] = selected_layout
    st.session_state['layout_params'] = layout_params
    
    # Important note about CNN and MOL
    st.warning("⚠️ Important: CNN models and MOL visualizations are always generated fresh for each layout and are never reused. This ensures fair comparisons across different layout strategies.")
    
    # Display current configuration
    st.info(f"**Current Layout**: {selected_layout}")
    
    # Run Pipeline Button
    if st.button("Run Full Pipeline", type="primary", key="run_pipeline"):
        st.session_state['pipeline_run'] = True

# Pipeline execution block
if st.session_state.get('pipeline_run', False):
    # Get parameters from session state
    dataset = st.session_state.get('dataset')
    preprocessing_params = st.session_state.get('preprocessing_params', {})
    
    if not dataset or not preprocessing_params:
        st.error("Missing dataset or preprocessing parameters.")
        st.stop()
    
    reuse_prev = preprocessing_params.get('reuse_prev', False)
    SEED = preprocessing_params.get('seed', 42)
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Dataset paths
    DATASET = dataset
    os.environ["DATASET"] = DATASET
    
    # Pass layout parameters to environment
    mol_layout = st.session_state.get('mol_layout', 'step_row')
    layout_params = st.session_state.get('layout_params', {})
    
    # Set TabNet environment variables
    tabnet_params = st.session_state.get('tabnet_params', {})
    if tabnet_params:
        os.environ["TABNET_N_STEPS"] = str(tabnet_params.get('n_steps', 6))
        os.environ["TABNET_STEP_DIM"] = str(tabnet_params.get('step_dim', 8))
        os.environ["TABNET_ATTN_DIM"] = str(tabnet_params.get('attn_dim', 8))
        os.environ["TABNET_GAMMA"] = str(tabnet_params.get('gamma', 1.5))
        os.environ["TABNET_LAMBDA_SPARSE"] = str(tabnet_params.get('lambda_sparse', 1e-4))
        os.environ["TABNET_MASK_TYPE"] = tabnet_params.get('mask_type', 'sparsemax')
        os.environ["TABNET_LEARNING_RATE"] = str(tabnet_params.get('learning_rate', 2e-2))
        os.environ["TABNET_BATCH_SIZE"] = str(tabnet_params.get('batch_size', 32))
        os.environ["TABNET_MAX_EPOCHS"] = str(tabnet_params.get('max_epochs', 100))
    
    # Ensure environment variables are set for layout
    os.environ["MOL_LAYOUT"] = mol_layout
    os.environ["SEED"] = str(SEED)
    
    if mol_layout == "packed" and 'target_width' in layout_params:
        os.environ["PACKED_TARGET_WIDTH"] = str(layout_params['target_width'])
    elif mol_layout == "step_sparse" and 'columns_per_step' in layout_params:
        os.environ["SPARSE_COLUMNS_PER_STEP"] = str(layout_params['columns_per_step'])
    
    # Define paths for this dataset
    PROCESSED_DIR = PROCESSED_BASE / DATASET
    TABNET_OUT = TABNET_DIR / "outputs" / f"output_{DATASET}"
    CNN_MODELS_DIR = CNN_DIR / "cnn_models"
    RESULTS_CSV = RESULTS_DIR / f"{DATASET}_results.csv"
    MOL_VIZ_DIR = MOL_VIZ_BASE / DATASET / mol_layout
    MOL_GRIDS_DIR = MOL_VIZ_DIR / "grids"
    
    # Create directories
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABNET_OUT.mkdir(parents=True, exist_ok=True)
    CNN_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check existing results - only for preprocessing and TabNet
    existing_results = check_existing_results(dataset, mol_layout)
    
    # ==================================================
    # Step 1: Preprocessing execution
    # ==================================================
    st.subheader("Step 1: Preprocessing")
    st.caption(
    "Cleans the raw dataset and produces a fully numerical, scaled representation "
    "suitable for TabNet training. This step does not involve any spatial modeling."
)
    
    if reuse_prev and existing_results['preprocessing']:
        st.info("✓ Using existing preprocessed data")
    else:
        status_text.text("Running: Preprocessing")
        try:
            success, output, metadata = run_step(
                name="Preprocessing",
                script_path=PREPROCESS_DIR / "run_preprocessing.py",
                env_vars=os.environ.copy()#,
                #timeout=600
            )
            
            if success:
                st.success("✓ Preprocessing completed")
                if output:
                    with st.expander("Preprocessing logs"):
                        st.text(output[:2000])
            else:
                st.error("✗ Preprocessing failed")
                st.code(output)
                st.stop()
                
        except PipelineStepError as e:
            st.error(f"✗ Preprocessing failed: {e}")
            st.stop()
    
    progress_bar.progress(20)
    
    # ==================================================
    # Step 2: TabNet Training
    # ==================================================
    st.subheader("Step 2: TabNet Training")
    st.caption(
    "Learns sparse, step-wise feature importance and decision masks from tabular data. "
    "These learned structures define *what* information will later be visualized."
)
    
    if reuse_prev and existing_results['tabnet']:
        st.info("✓ Using existing TabNet outputs")
    else:
        status_text.text("Running: TabNet Training")
        try:
            success, output, metadata = run_step(
                name="TabNet Training",
                script_path=TABNET_DIR / "train_tabnet.py",
                env_vars=os.environ.copy()#,
                #timeout=900
            )
            
            if success:
                from execution.validators import check_tabnet_outputs
                is_valid, missing, warnings = check_tabnet_outputs(TABNET_OUT)
                
                if is_valid:
                    st.success("✓ TabNet training completed successfully")
                else:
                    st.warning(f"⚠ TabNet training completed but some outputs missing: {missing}")
                
                if output:
                    with st.expander("TabNet logs"):
                        st.text(output[:3000])
                        
            else:
                st.error("✗ TabNet training failed")
                st.code(output)
                st.stop()
                
        except PipelineStepError as e:
            st.error(f"✗ TabNet training failed: {e}")
            st.stop()
    
    progress_bar.progress(40)
    
    # ==================================================
    # Step 3: Image builder
    # ==================================================
    st.subheader("Step 3: Image Building")
    st.caption(
    "Projects TabNet's learned feature-step structure into 2D images according to the "
    "selected layout strategy. This step controls *how* tabular information is spatially organized."
)

    # Check existing images for this layout
    layout_image_exists = (PROCESSED_DIR / "X_train_img.npy").exists()
    
    if reuse_prev and layout_image_exists:
        st.info("✓ Using existing CNN image tensors")
    else:
        status_text.text("Running: Image Building")
        try:
            success, output, metadata = run_step(
                name="Image Building",
                script_path=IMAGE_DIR / "tabnet_image_builder.py",
                env_vars=os.environ.copy()#,
                #timeout=300
            )
            
            if success:
                st.success("✓ Image building completed")
                if output:
                    with st.expander("Image builder logs"):
                        st.text(output[:2000])
            else:
                st.error("✗ Image building failed")
                st.code(output)
                st.stop()
                
        except PipelineStepError as e:
            st.error(f"✗ Image building failed: {e}")
            st.stop()
    
    progress_bar.progress(60)
    
    # ==================================================
    # Step 4: CNN Training
    # ==================================================
    st.subheader("Step 4: CNN Training")
    st.caption(
    "Trains a convolutional neural network on the generated images. "
    "Models are always trained fresh for each layout to ensure fair comparisons."
)

    status_text.text(f"Running: CNN Training for layout '{mol_layout}'")
    try:
        success, output, metadata = run_step(
            name="CNN Training",
            script_path=CNN_DIR / "train_cnn.py",
            env_vars=os.environ.copy()#,
            #timeout=600
        )
        
        if success:
            st.success("✓ CNN training completed")
            if output:
                with st.expander("CNN training logs"):
                    st.text(output[:2000])
        else:
            st.error("✗ CNN training failed")
            st.code(output)
            st.stop()
            
    except PipelineStepError as e:
        st.error(f"✗ CNN training failed: {e}")
        st.stop()
    
    progress_bar.progress(80)
    
    # ==================================================
    # Step 5: CNN Evaluation
    # ==================================================
    st.subheader("Step 5: CNN Evaluation")
    st.caption(
    "Evaluates the trained CNN on held-out test data to measure "
    "classification performance under the current layout."
)

    status_text.text("Running: CNN Evaluation")
    try:
        success, output, metadata = run_step(
            name="CNN Evaluation",
            script_path=CNN_DIR / "evaluate_cnn.py",
            env_vars=os.environ.copy()#,
            #timeout=300
        )
        
        if success:
            st.success("✓ CNN evaluation completed")
            if output:
                with st.expander("CNN evaluation logs"):
                    st.text(output[:2000])
        else:
            st.warning("⚠ CNN evaluation may be incomplete")
            if output:
                st.text(output[:1000])
            
    except PipelineStepError as e:
        st.warning(f"⚠ CNN evaluation failed: {e}")
    
    progress_bar.progress(90)
    
    # ==================================================
    # Step 6: MOL Visualizations
    # ==================================================
    st.subheader("Step 6: MOL Visualizations")
    st.caption(
    "Generates model-oriented visualizations (MOL) to analyze how learned representations "
    "and spatial layouts differ across classes. Always generated fresh for each layout."
)

    status_text.text("Running: MOL Visualization")
    try:
        success, output, metadata = run_step(
            name="MOL Visualization",
            script_path=IMAGE_DIR / "mol_visualizations.py",
            env_vars=os.environ.copy()#,
            #timeout=300
        )
        
        if success:
            st.success("✓ MOL visualizations completed")
            if output:
                with st.expander("MOL visualization logs"):
                    st.text(output[:2000])
        else:
            st.warning("⚠ MOL visualizations may be incomplete")
            if output:
                st.text(output[:1000])
            
    except PipelineStepError as e:
        st.warning(f"⚠ MOL visualizations failed: {e}")
    
    progress_bar.progress(95)
    
    # ==================================================
    # Step 7: Load and Display Results
    # ==================================================
    st.subheader("Step 7: Results Analysis")
    st.caption(
        "Aggregates quantitative metrics and visual diagnostics to support comparison "
        "across layouts and experimental configurations."
    )

    try:
        results_file = PROCESSED_DIR / f"cnn_evaluation_results_{mol_layout}.json"
        
        if results_file.exists():
            # Load evaluation results (computed by evaluate_cnn.py)
            with open(results_file, 'r') as f:
                eval_results = json.load(f)
            
            # Store in session state for visualization
            st.session_state['results'] = {
                "y_test": np.array(eval_results.get("y_test", [])),
                "y_pred": np.array(eval_results.get("y_pred", [])),
                "y_prob": np.array(eval_results.get("y_prob", [])) if "y_prob" in eval_results else None,
                "accuracy": eval_results.get("accuracy", 0),
                "balanced_accuracy": eval_results.get("balanced_accuracy", 0),
                "f1_score": eval_results.get("f1_score", 0),
                "cohen_kappa": eval_results.get("cohen_kappa", 0),
                "confusion_matrix": np.array(eval_results.get("confusion_matrix", [])),
                "classification_report": eval_results.get("classification_report", {}),
                "n_classes": eval_results.get("n_classes", 0),
                "correct_predictions": eval_results.get("correct_predictions", 0),
                "total_samples": eval_results.get("total_samples", 0)
            }
            
            st.success(
                f"✓ Evaluation Results Loaded: "
                f"Accuracy = {eval_results.get('accuracy', 0):.2%}"
            )
        else:
            st.error(
                "CNN evaluation results were not found. "
                "Please verify that evaluate_cnn.py completed successfully. "
                f"Expected results at: {results_file}"
            )
            st.info(
                "The evaluation should be run as part of the pipeline. "
                "If it failed, try running the pipeline again."
            )
            
    except Exception as e:
        st.error(f"Error loading evaluation results: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())
    
    progress_bar.progress(100)
    
    # ==================================================
    # Display Results Section
    # ==================================================
    st.header("Results & Visualizations")
    
    # Display pipeline configuration
    st.subheader("Pipeline Configuration")
    col_config1, col_config2, col_config3 = st.columns(3)
    with col_config1:
        st.metric("Dataset", DATASET)
    with col_config2:
        st.metric("Layout", mol_layout)
    with col_config3:
        st.metric("Seed", SEED)
    
    # Display TabNet Configuration
    st.subheader("TabNet Configuration Used")
    if tabnet_params:
        tabnet_config_df = pd.DataFrame(
            [(k, str(v)) for k, v in tabnet_params.items()],
            columns=["Parameter", "Value"]
        )
        st.dataframe(tabnet_config_df, width='stretch')
        
    # Display Preprocessed Data Preview
    st.subheader("Preprocessed Data Preview")
    try:
        processed_df = load_clean_data_preview(PROCESSED_DIR, n_rows=5)
        if processed_df is not None:
            st.dataframe(processed_df, width='stretch')
    except Exception as e:
        st.warning(f"Could not load preprocessed data: {e}")
    
    # TabNet Feature Assignments
    st.subheader("TabNet Feature Assignments")
    step_path = TABNET_OUT / "tabnet_step_assignment.csv"
    if step_path.exists():
        step_df = pd.read_csv(step_path)

        if "step_distribution" in step_df.columns:
            def format_distribution(x):
                try:
                    values = ast.literal_eval(x) if isinstance(x, str) else x

                    # Pretty formatting with 2 decimals
                    return "[" + ", ".join(f"{float(v):.2f}" for v in values) + "]"

                except Exception:
                    return x

            step_df["step_distribution"] = step_df["step_distribution"].apply(format_distribution)

        st.dataframe(step_df, width='stretch')
        
        with st.expander("Step Distribution with Features"):
            for s in range(step_df["dominant_step"].max() + 1):
                feats = step_df[step_df["dominant_step"] == s]["feature"].tolist()
                if len(feats) > 0:
                    # Show features instead of count
                    display_text = f"**Step {s}:** {', '.join(map(str, feats))}"
                    st.markdown(display_text)
    else:
        st.warning(f"TabNet step assignment file not found")
    
    # Performance Metrics
    st.subheader("Performance Metrics")
    
    results = st.session_state.get('results')
    if results:
        y_test = results['y_test']
        y_pred = results['y_pred']
        accuracy = results['accuracy']
        cm = results['confusion_matrix']
        report = results['classification_report']
        n_classes = results['n_classes']
        balanced_acc = results.get('balanced_accuracy', 0)
        f1 = results.get('f1_score', 0)
        kappa = results.get('cohen_kappa', 0)
        
        # Display key metrics (already computed, just display)
        col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
        with col_metrics1:
            st.metric("Accuracy", f"{accuracy:.2%}")
        with col_metrics2:
            st.metric("Balanced Accuracy", f"{balanced_acc:.2%}")
        with col_metrics3:
            st.metric("Macro F1-Score", f"{f1:.2%}")
        with col_metrics4:
            st.metric("Cohen's Kappa", f"{kappa:.3f}")
        
        # Classification Report (pre-computed)
        st.write("**Classification Report:**")
        report_df = pd.DataFrame(report).transpose()
        
        # Remove accuracy row if present
        if 'accuracy' in report_df.index:
            report_df = report_df.drop('accuracy')
        
        # Format percentage columns for display
        for col in ['precision', 'recall', 'f1-score']:
            if col in report_df.columns:
                report_df[col] = report_df[col].apply(
                    lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x
                )
        
        st.dataframe(report_df, width='stretch')
        
        # Confusion Matrix (pre-computed)
        st.write("**Confusion Matrix**")
        n_classes = cm.shape[0]

        # Compact sizing rule
        figsize = (3.2, 3.0) if n_classes <= 5 else (4.0, 3.6)
        fontsize = 7 if n_classes > 5 else 8

        fig, ax = plt.subplots(figsize=figsize, dpi=80)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar=False,
            annot_kws={"size": fontsize}
        )

        ax.set_xlabel("Predicted", fontsize=fontsize)
        ax.set_ylabel("True", fontsize=fontsize)
        ax.set_title(f"Confusion Matrix\n{mol_layout}", fontsize=fontsize+1)

        ax.tick_params(axis="both", labelsize=fontsize)

        plt.tight_layout(pad=0.4)

        st.pyplot(fig, width="content")
        plt.close(fig)

        # Save confusion matrix figure for download
        cm_buf = io.BytesIO()
        fig.savefig(cm_buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.05)
        cm_buf.seek(0)

        st.session_state["cm_plot_buffer"] = cm_buf

        # Save confusion matrix as DataFrame
        st.session_state["cm_dataframe"] = pd.DataFrame(
            cm,
            index=[f"T{i}" for i in range(n_classes)],
            columns=[f"P{i}" for i in range(n_classes)]
        )
        
    # MOL Grids per class
    st.subheader("MOL Image Grids")

    if not MOL_GRIDS_DIR.exists():
        st.warning(f"MOL grids directory not found:\n{MOL_GRIDS_DIR}")

    else:
        # Debug info
        st.caption(f"Reading MOL grids from: {MOL_GRIDS_DIR}")
        # Get all png files
        all_grid_files = sorted(MOL_GRIDS_DIR.glob("*.png"))

        if not all_grid_files:
            st.warning("No MOL grid PNG files found.")
            st.write("Expected files like:")
            st.code("train_class_0.png")
            st.code("test_class_0.png")

        else:
            # Show discovered files
            with st.expander("Detected MOL grid files"):
                for f in all_grid_files:
                    st.write(f.name)
            # Determine available classes directly from filenames
            detected_classes = set()

            for f in all_grid_files:
                name = f.stem
                parts = name.split("_")
                try:
                    cls_idx = parts.index("class") + 1
                    cls = parts[cls_idx]
                    detected_classes.add(cls)
                except Exception:
                    continue
            detected_classes = sorted(detected_classes)

            if not detected_classes:
                st.warning("Could not detect class IDs from MOL grid filenames.")

            for cls in detected_classes:
                st.markdown(f"### Class {cls}")
                train_img = MOL_GRIDS_DIR / f"train_class_{cls}.png"
                test_img = MOL_GRIDS_DIR / f"test_class_{cls}.png"

                col1, col2 = st.columns(2)
                with col1:
                    st.write("Train")
                    if train_img.exists():
                        st.image(str(train_img), width='stretch')
                    else:
                        st.error(f"Missing:\n{train_img.name}")
                with col2:
                    st.write("Test")
                    if test_img.exists():
                        st.image(str(test_img), width='stretch')
                    else:
                        st.error(f"Missing:\n{test_img.name}")
    
    # Download outputs section
    st.header("Download Outputs")
    
    # Create columns for download options
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Download Processed Data
        if (PROCESSED_DIR / "clean_data.csv").exists():
            try:
                processed_df = pd.read_csv(PROCESSED_DIR / "clean_data.csv")
                csv_processed = processed_df.to_csv(index=False)
                st.download_button(
                    "Processed Data",
                    data=csv_processed,
                    file_name=f"{dataset}_processed_data.csv",
                    mime="text/csv",
                    help="Download the cleaned and preprocessed dataset"
                )
            except Exception as e:
                st.error(f"Error loading processed data: {e}")
        else:
            st.info("No processed data")
    
    with col2:
        # Download Confusion Matrix
        try:
            if 'cm_dataframe' in st.session_state:
                cm_csv = st.session_state['cm_dataframe'].to_csv()
                st.download_button(
                    "Confusion Matrix",
                    data=cm_csv,
                    file_name=f"{dataset}_confusion_matrix_{mol_layout}.csv",
                    mime="text/csv",
                    help="Download the confusion matrix in CSV format"
                )
            else:
                st.info("No confusion matrix")
        except Exception as e:
            st.error(f"Error creating confusion matrix: {e}")
    
    with col3:
        # Download Classification Report
        try:
            if results:
                report_df_full = pd.DataFrame(report).transpose()
                if 'accuracy' in report_df_full.index:
                    report_df_full = report_df_full.drop('accuracy')
                
                report_csv = report_df_full.to_csv()
                
                st.download_button(
                    "Classification Report",
                    data=report_csv,
                    file_name=f"{dataset}_classification_report_{mol_layout}.csv",
                    mime="text/csv",
                    help="Download detailed classification report"
                )
        except Exception as e:
            st.error(f"Error creating classification report: {e}")
    
    with col4:
        # Download TabNet Feature Assignments
        try:
            if step_path.exists():
                with open(step_path, "r") as f:
                    assignments_content = f.read()
                st.download_button(
                    "Feature Assignments",
                    data=assignments_content,
                    file_name=f"{dataset}_feature_assignments.csv",
                    mime="text/csv",
                    help="Download TabNet feature step assignments"
                )
            else:
                st.info("No feature assignments")
        except Exception as e:
            st.error(f"Error loading feature assignments: {e}")
    
    # Additional download options
    with st.expander("Additional Download Options"):
        col_add1, col_add2, col_add3 = st.columns(3)
        
        with col_add1:
            # Download all TabNet outputs
            try:
                if TABNET_OUT.exists():
                    import zipfile
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                        with zipfile.ZipFile(tmp_zip.name, 'w') as zipf:
                            for tabnet_file in TABNET_OUT.glob("*"):
                                if tabnet_file.is_file():
                                    zipf.write(tabnet_file, tabnet_file.name)
                        
                        with open(tmp_zip.name, 'rb') as f:
                            st.download_button(
                                "All TabNet Outputs",
                                data=f.read(),
                                file_name=f"{dataset}_tabnet_outputs.zip",
                                mime="application/zip",
                                help="Download all TabNet output files as a ZIP archive"
                            )
            except Exception as e:
                st.error(f"Error creating TabNet zip: {e}")
        
        with col_add2:
            # Download all metrics in JSON format
            try:
                import json
                if results:
                    metrics_dict = {
                        'dataset': dataset,
                        'layout': mol_layout,
                        'seed': SEED,
                        'accuracy': accuracy,
                        'balanced_accuracy': balanced_acc,
                        'macro_f1_score': f1,
                        'cohens_kappa': kappa,
                        'confusion_matrix': cm.tolist(),
                        'timestamp': pd.Timestamp.now().isoformat()
                    }
                    
                    metrics_json = json.dumps(metrics_dict, indent=2)
                    
                    st.download_button(
                        "Metrics (JSON)",
                        data=metrics_json,
                        file_name=f"{dataset}_metrics_{mol_layout}.json",
                        mime="application/json",
                        help="Download all metrics in JSON format"
                    )
            except Exception as e:
                st.error(f"Error creating JSON: {e}")
        
        with col_add3:
            # Download all plots as PNG
            try:
                import tempfile
                import zipfile
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                    with zipfile.ZipFile(tmp_zip.name, 'w') as zipf:
                        # Save confusion matrix heatmap
                        if 'cm_plot_buffer' in st.session_state:
                            cm_buf = st.session_state['cm_plot_buffer']
                            zipf.writestr(f"{dataset}_confusion_matrix_{mol_layout}.png", cm_buf.getvalue())
                        
                        # Save target distribution plot
                        target_buf = io.BytesIO()
                        fig_target = plt.figure()
                        target_col = st.session_state.get('target_column', 'Unknown')
                        target_counts = df_raw[target_col].value_counts()
                        target_counts.plot(kind="bar")
                        plt.xticks(rotation=45, ha="right")
                        plt.tight_layout()
                        plt.savefig(target_buf, format='png', dpi=100, bbox_inches='tight')
                        target_buf.seek(0)
                        zipf.writestr(f"{dataset}_target_distribution.png", target_buf.getvalue())
                        plt.close(fig_target)
                    
                    with open(tmp_zip.name, 'rb') as f:
                        st.download_button(
                            "All Plots",
                            data=f.read(),
                            file_name=f"{dataset}_plots_{mol_layout}.zip",
                            mime="application/zip",
                            help="Download all generated plots as PNG files"
                        )
            except Exception as e:
                st.error(f"Error creating plots zip: {e}")

# Reset button
if st.session_state.get('pipeline_run', False):
    if st.button("Reset Pipeline", type="secondary"):
        for key in ['pipeline_run', 'dataset_loaded', 'preprocessing_params_set']:
            st.session_state[key] = False
        st.rerun()

# Footer
st.markdown("---")
st.caption("TabNet → CNN → MOL Pipeline Dashboard | Thesis Project | UI Orchestration Layer")