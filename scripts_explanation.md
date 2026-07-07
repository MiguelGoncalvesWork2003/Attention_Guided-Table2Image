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
**Tabular‑to‑image projection.**  
The core **Map** stage: takes the preprocessed data and frozen step assignments,
applies a chosen layout, and produces fixed CNN‑compatible images.  

- Applies an importance cutoff to discard low‑attention features.  
- Saves `X_train_img.npy`, `X_test_img.npy`, and layout metadata.  
- Completely decoupled from the CNN – no gradient flow exists between them.

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

- Also computes and saves training‑set metrics for over‑fitting analysis.  
- Performs the final supervised learning step.

### `cnn/evaluate_cnn.py`
**CNN evaluation script.**  
Loads the best checkpoint, runs inference on the test set, and computes a
comprehensive set of metrics (accuracy, precision, recall, F1, ROC‑AUC, etc.)
using the shared `running_all_models.metrics`.  

- Exports predictions, probabilities, confusion matrix, and classification
  report.  
- The single source of truth for all CNN performance
  numbers.

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

---

## Running All Models

### `running_all_models/benchmark.py`
**Original sequential benchmark (superseded by `benchmark_parallel.py`).**  
Evaluates all baseline models and AG‑T2I variants using a 5‑fold cross‑validation
loop with multiple random seeds.  
- Processes datasets one seed and one fold at a time – simpler but slower than
  the parallel version.  
- Preprocessing (imputation + scaling) is performed inside each fold, with no
  caching of intermediate results.  
- AG‑T2I variants are evaluated via the `SimplePipelineAPI` with
  `reuse_existing=False` to guarantee fresh training every run.  
- Still useful for small‑scale tests and debugging, but for production
  benchmarking the parallel script is strongly preferred.

### `running_all_models/models_factory.py`
**Baseline model factory.**  
Provides the `get_models()` function that returns a dictionary of pre‑configured
classifiers for the benchmark comparison.  

- Tree ensembles: XGBoost, LightGBM, CatBoost.  
- Tabular deep learning: TabNetClassifier, FT‑Transformer (lite).  
- Tabular‑to‑image baselines: IGTD‑inspired and Naive Reshape, both using a
  shared lightweight CNN (`TabNetCNN`) for fair comparison.  
- All models are instantiated with reasonable default hyperparameters and ready
  to be trained directly on the tabular data (or scaled data, for the neural
  models).

### `running_all_models/metrics.py`
**Extended evaluation metrics.**  
Contains two functions used throughout the benchmark and AG‑T2I evaluation:  

- `compute_extended_metrics(y_true, y_pred, y_proba)` – calculates accuracy,
  balanced accuracy, precision (macro & weighted), recall (macro & weighted),
  F1 (macro & weighted), and ROC‑AUC (if probabilities are provided).  
- `get_wrong_cases(y_true, y_pred, ...)` – returns a DataFrame of misclassified
  samples, including original indices and decoded class names for easier
  inspection.  

These functions ensure that every model is evaluated with the same set of
metrics and that error analysis is reproducible.

### `running_all_models/utils.py`
**Reproducibility and statistics helpers.**  

- `set_seed(seed)` – fixes Python, NumPy, and PyTorch random states (and
  CUDA deterministic flags).  
- `mean_std_ci(scores, confidence=0.95)` – returns the mean, standard
  deviation, and 95% confidence interval for an array of metric scores.  

Both utilities are used by the benchmark scripts to guarantee repeatable
experiments and to report aggregated results with confidence intervals.

### `running_all_models/statistical_tests.py`
**Statistical significance analysis for benchmark results.**  
Loads the aggregated benchmark output (`*_raw.csv` files) and performs:  

- Paired t‑tests and Wilcoxon signed‑rank tests between every pair of models,
  with Holm‑Bonferroni correction for multiple comparisons.  
- Friedman test (non‑parametric ANOVA) across all models, followed by a manual
  Nemenyi post‑hoc test when the Friedman test is significant.  
- Saves the resulting p‑value matrices and rankings as CSV and LaTeX tables.  

This script produces the statistical evidence needed to support the claims of
superiority or equivalence among models in the paper.

### `running_all_models/benchmark_parallel.py`
**Optimized parallel benchmark with caching.**  
Replaces the old sequential benchmark. It evaluates every baseline model
(XGBoost, LightGBM, CatBoost, Random Forest, MLP, TabNet, FT‑Transformer) and
all five AG‑T2I variants across multiple seeds and folds *in parallel*.  

- All tasks are submitted immediately, keeping all CPU cores fully utilised.  
- Preprocessing (imputation + scaling) is cached per fold for neural models,
  avoiding repeated heavy computation.  
- TabNet training is cached per (dataset, seed, fold) so that the five AG‑T2I
  layouts reuse the same attention model – reducing total runtime by up to
  80% for AG‑T2I.  
- Optional GPU support: automatically moves PyTorch models (TabNet, CNN,
  FT‑Transformer) to CUDA if available.  
- Outputs per‑fold metrics and an aggregated summary (CSV + LaTeX).  

**Usage:**  
```bash
python running_all_models/benchmark_parallel.py [--dataset Cancer] [--workers 8]

running_all_models/hyperparameter_search.py
Per‑model hyperparameter tuning with 3‑fold CV.
Performs a systematic search for the best hyperparameters of every model
in the benchmark. To ensure a fair comparison, all models use exactly the
same 3‑fold stratified cross‑validation protocol.

Tree ensembles & MLP: RandomizedSearchCV with publication‑informed grids.

TabNet & FT‑Transformer: manual random search with cross‑validation (20
iterations).

AG‑T2I (five layouts): custom CV loop that runs the full pipeline
(preprocessing → TabNet → image building → CNN) for each fold and averages
the accuracy.

Results are saved as JSON, ready to be loaded by the benchmark script for
final evaluation with tuned parameters.

**Usage:**  
```bash
python running_all_models/hyperparameter_search.py Cancer --target Class --agt2i_trials 20

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