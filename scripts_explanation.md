# Attention‑Guided Tabular‑to‑Image (AG‑T2I) – Script Reference

This document describes every important Python module in the repository.  
The pipeline follows the **Map‑Optimize‑Learn** philosophy:

* **Map** – Preprocessing and projection of tabular features into 2D images.  
* **Optimize** – TabNet's supervised feature attention.  
* **Learn** – CNN training and evaluation on the generated images.  

Each script’s place in this flow is clearly indicated, so you can see exactly
how the attention‑guided representations are built, trained, and evaluated.

---

## Preprocessing

### `preprocessing/decisions.py`
**Preprocessing configuration builder.**  
Creates a deterministic `PreprocessingConfig` from UI choices or environment
variables.  
Captures feature removal, imputation, encoding, scaling, and split settings.  

- Defines the transformation that turns raw data into the numeric
  tensor used by TabNet.  
- Replaces heuristic decisions with inspectable, data‑driven
  rules.

### `preprocessing/pipeline.py`
**Main preprocessing pipeline.**  
Applies a fixed sequence of transformers (column drop → imputation → encoding →
scaling) to the raw dataset, producing clean NumPy arrays.  

- Fitted exclusively on training data to prevent leakage.  
- Saves every fitted transformer and metadata for perfect reproducibility.  
- Delivers the standardised tabular input for TabNet.

### `preprocessing/transform.py`
**Fittable, serialisable transformers.**  
Atomic scikit‑learn‑style steps: `ColumnSelector`, `HighMissingDropper`,
`SmartImputer`, `CategoricalEncoder`, `FeatureScaler`, `TargetLabelEncoder`.  

- Each transformer can be pickled and reloaded.  
- The building blocks that keep preprocessing transparent and
  auditable.

### `preprocessing/run_preprocessing.py`
**Headless execution entry point.**  
Invoked by the Streamlit orchestrator (or directly) to run the full
preprocessing pipeline.  

- Reads configuration from environment variables.  
- Saves the cleaned data as CSV and NumPy files.  
- **Map** – automates the preparation of the tabular data for subsequent
  stages.

### `preprocessing/preprocessing_utils.py`
**Streamlit‑specific display helpers.**  
Renders preprocessing summaries, previews the transformed data, and validates
output files – all without touching the pipeline logic.  

- Supports the interpretability and reproducibility goals of the project.

---

## TabNet Training

### `tabnet_fs/train_tabnet.py`
**Supervised attention learning.**  
Trains a `TabNetClassifier` to produce sparse, task‑specific feature attention
masks.  

- Only authentic, gradient‑based attention – no synthetic fallback.  
- Exports: feature importance, step‑wise masks, step assignments, configuration
  and evaluation metrics.  
- **Optimize** – provides the attention structure that defines every subsequent
  AG‑T2I layout.

---

## Layout Strategies (image construction)

### `tabnet_fs/layouts/unified_layouts.py`
**Layout strategy hierarchy.**  
Contains `StepRowLayout`, `PackedLayout`, `StepSparseLayout`, and
`AttentionMapLayout` that translate frozen TabNet assignments into 2D pixel
coordinates.  

- Fully deterministic – same input always yields the same coordinates.  
- CNN‑agnostic – only specifies *where* each feature goes, not the pixel value.  
- **Map** – the bridge between the attention model and the image builder.

### `tabnet_fs/layouts/__init__.py`
**Layout package initialiser.**  
Re‑exports all layout classes and factory functions (`create_layout`,
`validate_layout_name`) for clean imports across the project.

### `image_builder/tabnet_image_builder.py`
**Tabular‑to‑image projection (core Map stage).**  
Takes the preprocessed data and frozen step assignments, applies a chosen layout,
and produces fixed CNN‑compatible images.  

- For coordinate‑based layouts, discards features below an importance cutoff
  (θ=0.005).  
- Splits the training fold into training (80%) and validation (20%) subsets
  before building images, using stratified sampling and a fixed random seed.
  The validation images are used exclusively for CNN early stopping.  
- Saves `X_train_img.npy`, `X_val_img.npy`, `X_test_img.npy` and layout metadata.  
- Completely decoupled from the CNN – no gradient flow exists between them.  
- The AttentionMap layout keeps all features and applies robust percentile‑based
  normalisation to scale pixel values to [0,1].

---

## CNN Architecture, Training & Evaluation

### `cnn/cnn_model.py`
**Lightweight CNN (`TabNetCNN`).**  
A simple architecture that automatically scales convolution sizes to the image
dimensions.  

- Three branches (small/medium/large) based on pixel count.  
- Shared across all tabular‑to‑image baselines for fair comparison.  
- The supervised model that receives the fixed AG‑T2I images.

### `cnn/train_cnn.py`
**CNN training script.**  
Trains the `TabNetCNN` on the generated images using fixed hyperparameters and
a `ReduceLROnPlateau` scheduler.  

- Loads training and validation images from the isolated output directory.  
- Saves the best model checkpoint (based on validation accuracy).  
- Also computes and saves training‑set metrics for over‑fitting analysis.  
- Performs the final supervised learning step (**Learn**).

### `cnn/evaluate_cnn.py`
**CNN evaluation script.**  
Loads the best checkpoint, runs inference on the test set, and computes a
comprehensive set of metrics (accuracy, precision, recall, F1, ROC‑AUC, etc.)
using the shared `running_all_models.metrics`.  

- Exports predictions, probabilities, confusion matrix, and classification
  report.  
- The single source of truth for all CNN performance numbers.

---

## Supporting Infrastructure

### `evaluation/metrics.py`
**Centralised metric computation.**  
Provides `compute_classification_metrics` (accuracy, balanced accuracy, macro
precision/recall/F1, Cohen’s κ, ROC‑AUC) and serialisation helpers.  

- Used by both the CNN evaluator and the Streamlit dashboard.  
- Keeps calculation separate from presentation.

### `execution/runner.py`
**Subprocess‑based runner.**  
Executes individual Python scripts as isolated processes with timeout and error
capture.  

- Guarantees the CLI and the dashboard use the exact same code.  
- Prevents accidental state leakage between pipeline stages.

### `execution/validators.py`
**Output validation utilities.**  
Checks the existence, structure, and integrity of all pipeline artefacts
(preprocessed data, TabNet outputs, CNN images, visualisations).  

- Acts as a safety net – no downstream stage runs with missing or corrupt files.

---

## AG‑T2I Visualisations

### `image_builder/mol_visualizations.py`
(File name kept for historical continuity – the script generates **AG‑T2I
interpretability plots**.)

Creates visual diagnostics directly from the projected images:
- Class grids (e.g., 3×3 samples per class)
- Single‑instance images
- Step‑activation bar charts
- Per‑class average images

These plots are used in the paper to show that the attention‑guided layout
preserves semantic structure and that step‑wise organisation is visible even
before CNN training.  
They are saved in `experiments/mol_visualizations/<dataset>/<layout>/`.

---

## Entry Points & Dashboard

### `api.py`
**Command‑line and Python API.**  
The single headless entry point for the entire AG‑T2I pipeline.  
- `run` – full pipeline for one dataset/layout.  
- `random` – parallel random hyperparameter search.  
- `bayesian` – Optuna‑based Bayesian optimisation.  
- Manages persistent splits and writes results to the standard directory tree.

### `app.py`
**Streamlit interactive dashboard.**  
Zero‑code interface that orchestrates the same scripts as `api.py`.  

- Phase 1/2: data loading and inspection.  
- Phase 3: pipeline execution with configurable layout and hyperparameters.  
- Phase 4: results – metrics, confusion matrix, **AG‑T2I image grids**.  
- All computation delegated to the subprocess runner.

---

## Running All Models (Benchmark & Hyper‑Parameter Optimisation)

### `running_all_models/benchmark.py`
**Original sequential benchmark (superseded by `benchmark_parallel.py`).**  
Evaluates all baseline models and AG‑T2I variants using a 5‑fold cross‑validation
loop with multiple random seeds.  
- Processes datasets one seed and one fold at a time – simpler but slower than
  the parallel version.  
- Still useful for small‑scale tests and debugging.

### `running_all_models/models_factory.py`
**Baseline model factory.**  
Provides `get_models()` and `get_tuned_models()` that return pre‑configured
classifiers for the benchmark comparison.  

- Tree ensembles: XGBoost, LightGBM, CatBoost.  
- Tabular deep learning: TabNetClassifier, FT‑Transformer (lite).  
- Tabular‑to‑image baselines: IGTD‑inspired and Naive Reshape, both using a
  shared lightweight CNN (`TabNetCNN`) for fair comparison.  
- Models can be instantiated with default parameters or with tuned
  hyperparameters loaded from JSON files.

### `running_all_models/metrics.py`
**Extended evaluation metrics.**  
Contains `compute_extended_metrics` and `get_wrong_cases` used throughout the
benchmark and AG‑T2I evaluation.  

- Computes accuracy, balanced accuracy, macro/weighted precision, recall, F1,
  and ROC‑AUC (if probabilities are provided).  
- Provides a DataFrame of misclassified samples for error analysis.

### `running_all_models/utils.py`
**Reproducibility and statistics helpers.**  

- `set_seed(seed)` – fixes Python, NumPy, PyTorch, and CUDA random states.  
- `mean_std_ci(scores)` – returns mean, standard deviation, and 95% confidence
  interval.

### `running_all_models/statistical_tests.py`
**Statistical significance analysis.**  
Loads aggregated benchmark output (`*_raw.csv` files) and performs paired
t‑tests, Wilcoxon signed‑rank tests with Holm‑Bonferroni correction, and the
Friedman test with Nemenyi post‑hoc.  

- Produces p‑value matrices and rankings as CSV and LaTeX tables.

---

### `running_all_models/benchmark_parallel.py`
**Optimised parallel benchmark with TabNet caching and dynamic output directory.**  
Evaluates all baseline models (XGBoost, LightGBM, CatBoost, TabNet,
FT‑Transformer, IGTD‑inspired, Naive Reshape) and all five AG‑T2I layouts
(StepRow, PackedRow, PackedCol, StepSparse, AttentionMap) across multiple seeds
and folds **in parallel**.

**Key features (current implementation):**

- **Global preprocessing** – runs once per dataset before any parallel tasks,
  avoiding race conditions on Windows. All folds reuse the same imputed and
  scaled arrays.
- **Caching of TabNet training** – per (fold, tabnet_params), TabNet training
  is performed only once and reused across all five AG‑T2I layouts, reducing
  total runtime by up to 80%.
- **Internal validation split** – the image builder splits the training fold
  into 80% training / 20% validation before building images. The CNN uses
  validation images for early stopping.
- **Dynamic output directory** – respects the `RESULTS_DIR` environment variable,
  allowing the hyper‑parameter search to isolate its results from regular
  benchmarks.
- **Robust to tiny folds** – guards against training sets with only one sample
  (which would crash TabNet’s batch normalisation).
- **`--agt2i` flag** – when set, only AG‑T2I models are evaluated; all baseline
  models are skipped.
- **Parallel execution** – all tasks (models × seeds × folds) are submitted
  simultaneously to `joblib.Parallel` with configurable `--workers`.

**Usage:**
```bash
python running_all_models/benchmark_parallel.py --dataset Cancer --workers 8
python running_all_models/benchmark_parallel.py --agt2i          # AG‑T2I only
```

---

### `running_all_models/hyperparameter_search.py`
**Parallel hyper‑parameter optimisation for all models.**  
Tunes every baseline model (randomised search with 3‑fold CV) and every AG‑T2I
layout (Bayesian optimisation with 3‑fold CV) **concurrently**.

**Current implementation details:**

- **Same CV protocol for all methods** – both baselines and AG‑T2I use a
  stratified 3‑fold inner cross‑validation, ensuring a fair comparison.
- **AG‑T2I evaluation reuses cached TabNet** – the objective function calls
  `run_agt2i_fold` from the benchmark script, so the first evaluation of a
  (fold, tabnet_params) pair trains TabNet once; subsequent evaluations are
  almost free.
- **Global preprocessing** – performed once per dataset before any parallel
  tuning, eliminating race conditions.
- **Separate Optuna databases per layout** – allows all layouts to be tuned in
  parallel without SQLite locking conflicts.
- **Robust to pipeline failures** – failed trials (e.g., due to tiny training
  sets) return a score of 0.0 instead of crashing the whole optimisation.
- **Saves best parameters as JSON** – ready to be loaded by `benchmark_parallel.py`
  for the final outer 5‑fold evaluation.
- **Automatic benchmark after tuning** – unless `--skip-benchmark` is passed,
  the tuned parameters are immediately used in a full benchmark run, with
  results saved to a separate `results_hyperparameter` directory.

**Usage:**
```bash
python running_all_models/hyperparameter_search.py --fresh --workers 8
python running_all_models/hyperparameter_search.py --dataset Cancer --skip-benchmark
```

---

## Archive (historical / auxiliary)

### `archive/analyser_of_tabnet_structure.py`
Deep, read‑only interpreter of TabNet artefacts (structural metrics, step
roles, critical features).

### `archive/analyze_tabnet_structure_light.py`
Lightweight numerical summary of TabNet attention structure.

### `archive/layout_builder.py`
Earlier standalone layout builder (superseded by `unified_layouts.py` and
`tabnet_image_builder.py`). Retained for reference.

### `archive/data_inspector.py`
Pure descriptive analysis of raw tabular data – no transformation or side
effects.

### `archive/metadata.py`
Legacy logging module; not used in the current pipeline.