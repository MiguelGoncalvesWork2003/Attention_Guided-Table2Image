
# ATTENTION-GUIDED TABULAR-TO-IMAGE FRAMEWORK – SCRIPT DOCSTRINGS

# This file collects the recommended introductory docstrings for every script in the pipeline.  Copy the appropriate block to the top of each corresponding Python file to improve readability and alignment with the Map-Optimize-Learn paradigm described in the accompanying paper.

# ---------------------------
# decisions.py

Preprocessing configuration builder.

This module translates human‑readable choices (from the Streamlit UI or
environment variables) into a **deterministic, serialisable, and hashable
`PreprocessingConfig` object**. The configuration defines *how* every step of
the preprocessing pipeline will be executed, ensuring **full reproducibility**
across runs and allowing the entire transformation to be audited and recomputed
from the raw data.

The `PreprocessingConfig` dataclass captures:
  - Which features to remove (user‑specified),
  - Missing‑value handling strategies for numerical and categorical columns,
  - Encoding and scaling schemes (all chosen to be TabNet‑compatible),
  - Model‑aware settings (e.g., sparse representation, no categorical embedding),
  - Data‑splitting ratio and random seed.

**Role in the Map–Optimize–Learn pipeline:**
- **Map:**  The configuration explicitly defines the transformation map from the
  raw feature space to the normalised tensor that will be consumed by TabNet and,
  later, converted into an image.
- **Optimize:**  The deterministic configuration replaces an external
  metaheuristic search for optimal preprocessing. Instead, decisions are
  grounded in dataset inspection and principled heuristics, making the
  preprocessing part of the “optimize” stage transparent and debuggable.

The module also provides `build_preprocessing_config_from_env()` to bridge
Streamlit inputs and the `PreprocessingConfig` constructor.


# --------------------------- 
# pipeline.py

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


# ---------------------------
# preprocessing_utils.py

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


# ---------------------------
# run_preprocessing.py

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


# ---------------------------
# transform.py

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


# ---------------------------
# train_tabnet.py

TabNet training script for the attention-guided tabular-to-image framework.

This script implements the **supervised attention learning stage** stage of the proposed attention-guided
tabular-to-image transformation: it trains an interpretable TabNet model to
learn supervised, task-specific feature attention. The resulting sparse masks
serve as the single source of truth for all subsequent deterministic spatial
layouts.

Key design choices:
  - Only authentic, gradient-based feature attention is used – no synthetic or
    fallback assignments are ever created.
  - A dedicated validation split ensures reliable early stopping.
  - The training procedure is fully reproducible (fixed seed, deterministic
    CUDA settings) and all hyperparameters are recorded.
  - All output artefacts (feature importance, stepwise masks, step assignments,
    trained model, configuration, and metrics) are saved in a structured
    directory, enabling read‑only interpretation and deterministic layout
    construction in later stages.

When executed, it:
  1. Loads the preprocessed tabular data (numpy arrays produced by the
     preprocessing pipeline).
  2. Creates and trains a `TabNetClassifier` with the configured number of
     decision steps `K`.
  3. Extracts the 3D attention masks of shape (K, N, F) and the global
     feature importance vector.
  4. Computes per‑feature step assignments by averaging masks across samples
     and taking the argmax step.
  5. Evaluates train / test accuracy and persists all artefacts.

Relation to the paper:
  The step assignments and importance scores are used by the **Layout Builder**
  to construct the CNN‑compatible image representations. The training
  process embodies the "implicit optimisation" described in Section 4:
  instead of a population‑based metaheuristic, the layout geometry emerges
  directly from supervised learning of feature attention.

Requires: `pytorch-tabnet`, preprocessed data in `data/processed/<dataset>/`.


# ---------------------------
# unified_layouts.py (layout strategies base classes)

Unified, object‑oriented design of layout strategies for attention‑guided
tabular‑to‑image generation.

This module provides a clean, extensible hierarchy for defining how TabNet’s
step assignments and importance scores are translated into 2D image coordinates.
It abstracts the layout logic away from the builder, making it easy to
experiment with new spatial organisations without modifying the core pipeline.

Classes:
  - `BaseLayoutStrategy` – groups features by dominant step, optionally
    collapses empty rows, and defines the interface (`compute_image_shape`,
    `map_feature`).
  - `StepRowLayout` – each TabNet decision step occupies one row; features
    within a step are placed left‑to‑right in descending importance.
  - `PackedLayout` – flattens all features into a dense grid sorted by global
    importance, maximising spatial locality for CNNs.
  - `StepSparseLayout` – allocates a fixed number of columns per step,
    preserving empty slots for interpretability.

Factory functions (`create_layout`, `create_layout_from_config`) enable
instantiation by name, facilitating integration with the Streamlit interface
and configuration files.

All strategies are **deterministic**: given the same step assignment DataFrame,
they always produce the same coordinate mapping. They are explicitly designed
to be “CNN‑agnostic” – they specify only the spatial layout, leaving the actual
image generation and CNN training to separate modules.

Relation to the paper:
  The three layout strategies correspond to the AG‑T2I‑S, AG‑T2I‑G, and
  AG‑T2I‑P variants evaluated in the experimental section. This module
  therefore provides the exact implementations used in the empirical comparison,
  enabling full reproducibility.


# ---------------------------
# layouts/__init__.py (package initialiser)

Layout strategies package.

Re‑exports the base layout class and all concrete strategies from
`unified_layouts.py`, together with helper functions for dynamic creation and
validation. This package is the public interface used by the Streamlit
orchestrator and the image generation stage.

Usage:
  from layouts import StepRowLayout, create_layout

  layout = create_layout("step_row", step_df)
  shape = layout.compute_image_shape()   # (channels, height, width)
  row, col = layout.map_feature(step, local_rank)


# ---------------------------
# cnn_model.py

Convolutional neural network architecture for attention‑guided
tabular‑to‑image representations.

This module defines the `TabNetCNN` class, a lightweight CNN designed to
process the 2D image representations produced by the deterministic layout
stage. The architecture is intentionally kept simple and fixed across all
experiments to ensure that any performance differences originate from the
layout geometry rather than from model capacity.

Architecture design:
  - Small branch (≤16 pixels): 8→16 channels, 2×2 convolutions.
  - Medium branch (≤100 pixels): 16→32 channels, 3×3 convolutions.
  - Default branch (>100 pixels): 32→64 channels, 3×3 convolutions.
  All branches end with adaptive average pooling to 1×1, followed by a
  dropout layer and a linear classifier.

The model requires the final image height and width to select the appropriate
branch, but the spatial dimensions are determined entirely by the chosen layout
strategy; the CNN itself does not impose any prior on feature organisation.

**Role in the Map–Optimize–Learn pipeline:**
  - **Learn:** The CNN is trained exclusively on the frozen image
    representations, with no gradient flow back to TabNet or the layout
    builder. This strict decoupling enables controlled experimentation
    with different spatial layouts while keeping the learner identical.
  - The architecture is shared across all tabular‑to‑image baselines
    (IGTD, naive reshape, AG‑T2I variants) to guarantee fair comparison.


# ---------------------------
# train_cnn.py

CNN training script for attention‑guided tabular‑to‑image representations.

This script implements the **CNN learning stage** of the attention-guided tabular-to-image framework: it trains a convolutional neural network on the image
representations previously produced by a deterministic, attention‑guided
layout. The training process is fully decoupled from both the TabNet
feature‑attention model and the layout builder.

Key properties:
  - Loads `X_train_img.npy` (and optionally `X_val_img.npy`) from the
    processed data directory, together with the encoded labels.
  - Instantiates a `TabNetCNN` with the exact spatial dimensions of the
    generated images, ensuring the architecture matches the layout geometry
    without any resizing or interpolation.
  - Uses a fixed hyperparameter set (learning rate, optimizer, dropout,
    batch size, epochs) read from environment variables, with a
    `ReduceLROnPlateau` scheduler for stable convergence.
  - Saves the best model checkpoint (based on validation accuracy) together
    with a JSON configuration file that records all hyperparameters and
    the final image shape.

The script expects the image arrays to be 4D `(N, C, H, W)`, where `C=1`
(single‑channel grayscale images). This format is the direct output of the
layout projection, ensuring no information is lost or distorted.

**Role in the Map–Optimize–Learn pipeline:**
  - After the **Map** stage (preprocessing → image generation) and the
    **Optimize** stage (TabNet training → layout derivation), this script
    performs the final supervised learning step using a CNN.
  - No feedback is ever passed from the CNN back to the earlier stages,
    preserving the controlled experimental protocol described in Section 4.
  - The saved configuration and model checkpoint enable fully reproducible
    evaluation, which is carried out by `evaluate_cnn.py`.


# ---------------------------
# evaluate_cnn.py

CNN evaluation script — the single authority for computing and saving
classification metrics on the test set.

This script loads a trained `TabNetCNN` model (from a checkpoint produced
by `train_cnn.py`) and evaluates it on the held‑out test images. It
computes a comprehensive set of metrics using the centralised
`evaluation.metrics` module and saves all results in a structured format
for both programmatic consumption and the Streamlit UI.

Workflow:
  1. Load the CNN configuration JSON (image shape, hyperparameters) and
     the model checkpoint.
  2. Load `X_test_img.npy` and `y_test.npy`; validate tensor dimensions
     and normalise labels to 0‑based indexing.
  3. Perform batched inference with `torch.no_grad()` to obtain predicted
     classes and class probabilities.
  4. Call `compute_classification_metrics` to produce accuracy, balanced
     accuracy, macro precision/recall/F1, Cohen’s kappa, confusion matrix,
     classification report, and (for binary problems) ROC‑AUC.
  5. Save the full metrics dictionary as JSON, and also export the raw
     predictions, probabilities, confusion matrix, and classification
     report as separate files for downstream analysis.
  6. Print a summary table for quick inspection.

**Role in the Map–Optimize–Learn pipeline:**
  - Constitutes the **evaluation** sub‑stage of **Learn**, delivering the
    performance numbers that populate Table 1 and the ablation studies in
    the paper.
  - The output files are the authoritative source of all reported CNN
    metrics, guaranteeing that the same numbers can be reproduced from the
    saved artefacts without re‑running the entire pipeline.
  - By separating evaluation from training, the script reinforces the
    pipeline’s modularity and reproducibility: the model is never modified
    during evaluation, and all metrics are computed in a standardised,
    library‑based manner.


# ---------------------------
# tabnet_image_builder.py

Tabular‑to‑image projection (the **deterministic feature-to-image projection** stage of the attention-guided tabular-to-image framework).

This script is the central implementation of the attention‑guided
tabular‑to‑image transformation described in Section 4 of the paper.
It converts the preprocessed tabular data into fixed, CNN‑compatible
image representations using a deterministic spatial layout derived
entirely from the frozen TabNet step assignments.

Workflow:
  1. Load the preprocessed numerical arrays (`X_train.npy`, `X_test.npy`,
     feature names) and the saved TabNet step assignment CSV.
  2. Apply an importance cutoff (default 0.005) to discard features with
     negligible attention; this step prevents noise pixels from diluting
     the image signal.
  3. Instantiate a layout strategy (`step_row`, `packed`, or `step_sparse`)
     via the unified layout interface. The layout defines the image
     dimensions and the mapping from each feature to a pixel coordinate.
  4. For every sample, place the feature value at the assigned (row, col)
     location, producing a single‑channel grayscale image of shape
     `(C=1, H, W)`.
  5. Save the resulting image arrays (`X_train_img.npy`, `X_test_img.npy`)
     and a JSON metadata file that records the layout geometry, step groups,
     and feature ordering.

Key properties:
  - **Fully deterministic:** The same step assignments and layout choice
    always produce the same image coordinates. No randomness or learning
    is involved in this stage.
  - **Decoupled:** The image builder does not depend on the CNN or any
    downstream learner. It operates solely on the artefacts produced by
    the TabNet training.
  - **Reproducible:** All parameters (layout name, importance cutoff) are
    captured in the metadata file, enabling exact regeneration of the
    experimental images.

**Role in the Map–Optimize–Learn framework:**
  This script executes the **Map** step after the **Optimize** step
  (TabNet training) has completed. It bridges the interpretable tabular
  model and the CNN learner, materialising the layout geometry that
  embodies the supervised, task‑specific feature attention. The image
  arrays it writes are directly consumed by `train_cnn.py` and
  `evaluate_cnn.py`.


# ---------------------------
# metrics.py
 
Metrics computation module for the attention-guided tabular-to-image framework.

This module provides pure, reusable functions for evaluating classification
performance. It enforces a clean separation between metric computation and
presentation logic (e.g., Streamlit UI), ensuring that all evaluation code is
centralised, testable, and reproducible.

Key functions:
  - `compute_classification_metrics`: Computes accuracy, balanced accuracy,
    macro precision/recall/F1, Cohen’s kappa, confusion matrix, and a full
    classification report. For binary problems, it optionally computes ROC-AUC
    and ROC curve data.
  - `format_metrics_for_display`: Converts the raw metrics dictionary into
    human-readable percentage/string values suitable for UI or paper tables.
  - `save_metrics_to_json`: Serialises the metrics dictionary to JSON,
    automatically handling numpy array conversion.

**Role in the Map–Optimize–Learn pipeline:**
  - **Learn:** After training the CNN classifier on the image representations,
    this module evaluates its predictions against the ground truth.
  - The metrics are used to populate Table 1 and the ablation tables in the
    paper, providing standardised, comparable performance figures across
    baselines and layout strategies.
  - The separation of computation from UI ensures that the exact same metrics
    can be generated in both interactive (Streamlit) and script‑based workflows,
    contributing to the full reproducibility of the reported results.

# ---------------------------
# mol_visualizations.py

Visual diagnostics for attention‑guided tabular‑to‑image representations.

This script generates a comprehensive set of interpretability plots that
qualitatively assess the quality of the MOL image representations and the
spatial structure induced by the learned attention. It produces exactly the
kind of visualisations referenced in Sections 5 and 6 of the paper.

Output categories:
  1. **Class grids** – a 3×3 (or similar) grid of images for each class,
     allowing visual inspection of intra‑class consistency and inter‑class
     differences. Horizontal lines corresponding to TabNet step boundaries
     are overlaid when layout metadata is available.
  2. **Single‑instance images** – high‑resolution visualisations of
     individual test samples, useful for detailed examination of
     feature‑to‑pixel mappings.
  3. **Step analysis** – bar charts showing the average activation and the
     number of features assigned to each TabNet step, connecting the spatial
     layout back to the attention structure.
  4. **Average images per class** – the pixel‑wise mean of all training (or
     test) images belonging to a class, revealing which spatial regions are
     consistently activated for each category.

All images are normalised globally across the training and test sets to
ensure consistent colour scales, and the output is saved in a directory
hierarchy (`experiments/mol_visualizations/<dataset>/<layout>/`) for easy
navigation.

**Role in the Map–Optimize–Learn pipeline:**
  - These visualisations are the primary **interpretability evidence** in the
    paper. They demonstrate that the attention‑guided spatial layouts preserve
    semantic structure (e.g., class‑specific regions) and that the step‑wise
    organisation remains legible in the final image representation.
  - The script operates after the image builder (`tabnet_image_builder.py`)
    and can be run independently; it does not require the CNN to be trained.
    This reinforces the claim that interpretability is a property of the
    layout itself, not of the downstream convolutional model.
  - Together with the interactive Streamlit tool, the visualisations support
    the paper’s argument that supervised, model‑aware spatialisation yields
    transparent and analysable representations.


# ---------------------------
# runner.py

Generic subprocess‑based pipeline runner with error handling and logging.

This module encapsulates all subprocess execution logic used to orchestrate the
distinct stages of the attention‑guided tabular‑to‑image framework. It provides
a uniform, reproducible interface for invoking preprocessing, TabNet training,
layout construction, CNN training, and visualisation steps as separate,
isolated processes.

Key features:
  - `run_step`: Executes a single Python script as a subprocess, capturing
    stdout and stderr, enforcing timeouts, and optionally raising a custom
    `PipelineStepError` on failure.
  - `run_multiple_steps`: Sequentially runs a list of steps, with the ability
    to stop on first error.
  - `prepare_environment`: Builds a consistent `PYTHONPATH` and encoding
    environment, ensuring that subprocesses import project modules correctly.
  - `clean_output`: Normalises special Unicode characters for safe display in
    terminal and Streamlit logs.

**Role in the Map–Optimize–Learn pipeline:**
  - Serves as the execution backbone of the interactive Streamlit application
    and of any head‑less experiment scripts.
  - Guarantees that the **exact same Python code** is invoked regardless of
    whether the pipeline is run interactively or from the command line – a
    crucial point for the paper’s emphasis on reproducibility.
  - The subprocess isolation ensures that no global state leaks between stages,
    preventing, for example, accidental feedback from the CNN stage into the
    TabNet optimisation.


# ---------------------------
# validators.py

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


# ---------------------------
# api.py

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
  generate the qualitative plots used in the paper.

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


# ---------------------------
# app.py

Attention‑Guided Tabular‑to‑Image Pipeline – Interactive Dashboard.

Streamlit application for the deterministic attention-guided tabular-to-image framework described
in the paper. It provides a fully interactive, zero‑code interface to execute
every stage of the pipeline, from raw data inspection to trained CNN
evaluation and interpretable visualisations.

The dashboard functions as an **orchestration layer** that mirrors exactly
the logic of the command‑line API (`pipeline_api.py`), but presents results
in real time and enables exploratory analysis of layout strategies and
hyperparameters. All heavy computation is delegated to the dedicated
execution modules via `execution.runner.run_step()`, ensuring that the
dashboard, the CLI, and the paper’s experiments share the same code base.

**Phases of the interactive workflow:**

1. **Data Loading & Inspection (Phase 1 & 2)**
   - Select an existing benchmark dataset or upload a new CSV.
   - Inspect raw statistics, missing values, class distribution.
   - Configure target column, features to remove, and preprocessing
     parameters (missing imputation, scaling, encoding).

2. **Pipeline Execution (Phase 3 – Map, Optimize, Learn)**
   - Choose a spatial layout strategy (`step_row`, `packed`, `step_sparse`)
     that defines how TabNet’s attention structure is projected to 2D.
   - Adjust TabNet hyperparameters (number of steps, attention dimension,
     sparsity, learning rate, etc.) directly in the UI.
   - Execute the full end‑to‑end pipeline with a single click:
       * **Map:** `run_preprocessing.py` → `tabnet_image_builder.py`
       * **Optimize:** `train_tabnet.py` (attention‑guided layout derivation)
       * **Learn:** `train_cnn.py` → `evaluate_cnn.py` (CNN training & test)
   - Optionally reuse existing preprocessing and TabNet outputs to speed up
     layout comparisons, while CNN models and MOL visualisations are always
     regenerated for fairness.

3. **Results & Visualisations (Step 7)**
   - Displays accuracy, balanced accuracy, F1‑score, Cohen’s κ, and a full
     classification report – all computed by `evaluate_cnn.py` and loaded
     from the standard JSON results file.
   - Confusion matrix rendered as a Seaborn heatmap.
   - MOL image grids (per class, train & test) showing actual pixel
     representations produced by the layout.
   - TabNet feature‑step assignment table with per‑step feature groups.
   - Download buttons for processed data, metrics, confusion matrices, and
     all generated plots.

**Design principles:**
- **No model or metric computation** inside the dashboard – it only
  coordinates existing scripts and displays their outputs.
- **Full compatibility** with the `SimplePipelineAPI`; any run from the
  dashboard is reproducible via `python pipeline_api.py run`.
- **Stateful session management** ensures that dataset, target column,
  layout, and parameter choices persist across UI re‑renders.

This dashboard serves as both a demonstration tool for the paper’s
interpretability claims and a practical experimentation environment for
researchers exploring attention‑guided tabular‑to‑image transformations.


# ----------------------------------------------
# ARCHIVE
# analyser_of_tabnet_structure.py

Read‑only interpreter for TabNet’s learned internal structure.

This module performs **post‑hoc analysis** of the TabNet artefacts produced
by `train_tabnet.py`. It never recomputes assignments, trains a model, or
generates synthetic data. Its sole purpose is to extract structural insights
that support the interpretability analysis in the paper (Section 5).

The `TabNetStructureInterpreter` class:
  - Loads the **step assignment CSV** (the single source of truth) together
    with feature importance, configuration, and optional 3D masks.
  - Computes structural metrics: feature utilisation ratio, step‑balance Gini,
    normalised step entropy, and the number of active steps.
  - Interprets step roles (Specialist, Focused, High‑impact, Balanced) based on
    feature count and average importance.
  - Optionally identifies *critical features* – those whose attention value is
    consistently the maximum within their assigned step – using the raw 3D masks.
  - Generates publication‑ready visualisations: step distribution, importance by
    step, structural metric bar charts, and step hierarchy plots.

All outputs are strictly derived from the saved artefacts, guaranteeing
reproducibility and full traceability.

Usage:
  python build_tabnet_structure.py <dataset> --visualize

Relation to the paper:
  The structural metrics and visualisations serve as qualitative evidence that
  TabNet’s attention masks induce meaningful feature groupings, which are then
  transferred to the image layout. The interpreter thus bridges the
  “Optimize” (TabNet training) and “Map” (layout construction) stages by
  quantifying the quality of the learned attention structure.


# ---------------------------
# analyze_tabnet_structure_light.py

Compact, read‑only structural analyser for TabNet artefacts.

This script provides a lightweight alternative to `analyser_of_tabnet_structure.py`
for quick inspection of a trained TabNet model. It computes basic statistics,
feature‑importance rankings, and step‑assignment distributions without any
external visualisation dependencies (except matplotlib for optional plots).

The `TabNetStructureAnalyzer` class:
  - Loads the same artefacts (importance, step assignments, configuration,
    optional masks) as the full interpreter.
  - Reports importance statistics (mean, std, Gini coefficient) and step
    utilisation rates.
  - Generates a comprehensive JSON report that can be consumed by downstream
    notebooks or the Streamlit UI.

This module is intended for programmatic use, e.g., within automated
evaluation pipelines, or when only a numerical summary is required.

Relation to the paper:
  It complements the interactive application by providing a lightweight,
  script‑friendly entry point for inspecting TabNet’s attention structure,
  reinforcing the claim that the layout’s spatial semantics are grounded in
  quantifiable, task‑aware feature relevance.


# ---------------------------
# layout_builder.py

Deterministic spatial layout builder for attention‑guided tabular‑to‑image
representations.

This module implements the **deterministic feature-to-image projection** stage of the attention-guided tabular-to-image framework: it converts the
frozen TabNet step assignments and feature importance scores into concrete
two‑dimensional grid layouts. Once produced, these layouts serve as the fixed
canvas onto which individual tabular samples are projected to form
CNN‑compatible images.

The `LayoutBuilder` class is designed with a strict “no learning, no
interpretation, no CNN constraints” philosophy:
  - It loads the **authentic** step assignments and importance from saved CSVs
    (produced by `train_tabnet.py`).
  - All layout functions are purely deterministic – given the same assignments,
    the same pixel coordinates are always produced.
  - No feedback from the CNN stage is used, and no hyperparameter search is
    performed on the layout itself.

Provided layout strategies include:
  - `importance_grid` – square grid sorted by global importance.
  - `packed` – row‑major packing, optionally grouped by step.
  - `step_rows` – each TabNet step becomes one image row.
  - `step_sparse` – step‑separated columns with empty slots preserved.

For every layout, the builder exports a CSV mapping feature → (row, col),
a NumPy matrix of the same shape filled with importance scores, and a summary
JSON.

Relation to the paper:
  The deterministic mapping from learned attention to image coordinates is
  the core contribution of the framework. This module is the exact
  implementation of the attention‑to‑grid mapping described in Section 4,
  ensuring that the spatial organisation of the CNN input is directly
  interpretable through its connection to TabNet’s supervised feature masks.


# ---------------------------
# data_inspector.py


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


# --------------------------- 
# metadata.py
Legacy structured logging and configuration module for TabNet experiments.
Originally designed to capture experiment metadata, configuration hashes,
and training histories in a reproducible format.

In the current Map–Optimize–Learn framework, logging and artifact
persistence are handled directly by `train_tabnet.py` (which saves configs,
metrics, and summaries as part of its output directory). The dataclasses
`TabNetConfig` and `TabNetResults`, along with `TabNetExperimentLogger`,
are not imported or instantiated anywhere in the execution pipeline.

Retained for reference and possible standalone experiments.