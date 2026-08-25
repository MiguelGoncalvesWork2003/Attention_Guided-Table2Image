# Script Reference

Technical reference for every script in the pipeline: what it does, what it
reads, what it writes, and the environment variables that control it. For
project overview, installation, and quick-start commands, see `README.md`
instead — this document assumes you already know what the project does and
need to know how a specific script works.

Organised by pipeline stage, in the order data actually flows through them.

---

## 1. Preprocessing (`preprocessing/`)

### `run_preprocessing.py`

Entry point for preprocessing. Reads `DATASET` and `TARGET_COL` (required),
loads `data/raw/{DATASET}.csv`, and dispatches to one of two branches
depending on `USE_CUSTOM_SPLIT`:

- **Standard branch** (`USE_CUSTOM_SPLIT` unset or `false`): builds a
  `PreprocessingConfig` from environment variables via `decisions.py`, then
  calls `pipeline.py`'s `run_preprocessing_pipeline()`, which internally
  splits the data using `PreprocessingConfig.split_ratio` (default 0.3).
- **Custom-split branch** (`USE_CUSTOM_SPLIT=true`): reads pre-computed
  `TRAIN_IDX_PATH` / `TEST_IDX_PATH` `.npy` files and calls
  `PreprocessingPipeline.fit_presplit()` directly, bypassing the internal
  split entirely. This is the branch every fold-aware caller (the ablation
  orchestrators, the main benchmark) actually uses, since they need to
  preprocess a *specific* train/test partition, not a fresh random one.

Writes to `PROCESSED_DIR` (default `data/processed/{DATASET}`, but every
real caller overrides this to a fold- or experiment-specific directory):
`X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`,
`feature_names.npy`, plus an `artifacts/` subfolder with the fitted
transformer objects.

Key environment variables: `DATASET`, `TARGET_COL`, `PROCESSED_DIR`,
`OUTPUT_DIR`, `USE_CUSTOM_SPLIT`, `TRAIN_IDX_PATH`, `TEST_IDX_PATH`,
`DROP_THRESHOLD`, `CAT_MISSING`, `NUM_MISSING`, `SCALING`,
`ENCODE_CATEGORICALS`.

### `decisions.py`

Pure configuration logic — no I/O. `build_preprocessing_config_from_env()`
reads the environment variables above and returns a validated
`PreprocessingConfig` dataclass. Accepts both the short internal codes
(`CAT_MISSING=explicit`/`drop`, used by every headless script) and the raw
Streamlit UI label strings (`"Treat as category"`/`"Drop categorical
columns"`, used only if `app.py` ever passes the label directly) — both
resolve to the same config either way.

### `transform.py`

The actual transformer classes: `HighMissingDropper` (drops columns above
`missing_threshold`, fit on train only), `SmartImputer` (median for
numerical, explicit-category or drop for categorical), `CategoricalEncoder`
(label-encodes categorical columns; unseen test categories map to the most
frequent *training* category's own code, not a sentinel value),
`StandardScaler`-equivalent scaling. Each implements `fit`/`transform`
separately so leakage is structurally prevented — `fit` never sees test
data in either the standard or custom-split branch.

### `pipeline.py`

`PreprocessingPipeline` chains the `transform.py` classes into one
`fit_transform()` (standard branch) or `fit_presplit()` (custom-split
branch) call. Both methods apply the identical sequence: remove
user-selected features → drop high-missing columns → impute → encode
categoricals → scale — fit exclusively on the training partition passed
in, transform-only on the test partition.

### `preprocessing_utils.py`

Streamlit display helpers only (`display_preprocessing_summary`,
`load_clean_data_preview`, `validate_preprocessing_outputs`). Used by
`app.py`; no role in the headless pipeline.

---

## 2. TabNet Training (`tabnet_fs/`)

### `train_tabnet.py`

Trains a `TabNetClassifier` on the preprocessed training fold, then
extracts and aggregates its attention masks into the per-feature
statistics every downstream layout depends on.

Reads `X_train.npy`/`y_train.npy` from `PROCESSED_DIR`. Internally splits
80/20 (stratified, `random_state` = the `SEED` env var) into a
model-fitting set and an early-stopping validation set — **and persists
both index arrays** (`cnn_train_idx.npy`, `cnn_val_idx.npy`) to its output
directory, so the CNN stage can provably reuse the identical partition
rather than reconstructing it independently.

Computes, per feature `j`: `a_{j,k}` (mean attention at step `k` across the
training set), `s_j` (TabNet's own aggregate global-importance score, an
externally-defined quantity from the library, not derived from `a_{j,k}`),
and `k_j*` (the dominant step, ties broken toward the lowest index).

Writes to `OUTPUT_DIR/tabnet_output/` (or the legacy default
`tabnet_fs/outputs/output_{DATASET}/` if `OUTPUT_DIR` is unset):
`tabnet_step_assignment.csv`, `tabnet_feature_importance.csv`,
`tabnet_masks.npy`, `tabnet_model.zip`, `cnn_train_idx.npy`,
`cnn_val_idx.npy`, `tabnet_config.json`, `summary.json`.

Key environment variables: `DATASET`, `TARGET_COL`, `SEED`, `PROCESSED_DIR`,
`OUTPUT_DIR`, `TABNET_N_STEPS`, `TABNET_STEP_DIM`, `TABNET_ATTN_DIM`,
`TABNET_GAMMA`, `TABNET_LAMBDA_SPARSE`, `TABNET_MASK_TYPE`,
`TABNET_LEARNING_RATE`, `TABNET_BATCH_SIZE`, `TABNET_MAX_EPOCHS`,
`TABNET_PATIENCE`.

---

## 3. Image Construction (`image_builder/`)

### `tabnet_image_builder.py`

Turns TabNet's per-feature statistics into pixel arrays, according to
whichever layout `MOL_LAYOUT` selects.

Reads the persisted split indices from `TABNET_IDX_DIR` (falls back to
reconstructing the same 80/20 split independently — mathematically
identical given the same `PROCESSED_DIR` and `SEED` — if that directory
isn't set or doesn't contain them yet, which happens during hyperparameter
search when TabNet training and image building run in the same call).
Reads the step-assignment table from `TABNET_STEP_CSV_PATH` (or the
default TabNet output location).

Filters features below `IMPORTANCE_CUTOFF` (default 0.005 — see
`run_e4_threshold_sensitivity.py`), then dispatches to `unified_layouts.py`
to compute each retained feature's pixel coordinate, and writes the
resulting tensors.

Writes to `OUTPUT_DIR`: `X_train_img.npy`, `X_val_img.npy`,
`X_test_img.npy`, `y_train.npy` (reduced to the retained split), `y_val.npy`,
`y_test.npy`, and `tabnet_layout_{tag}.json` — the geometry metadata
(`image_shape`, `n_features_retained`, `sparsity`, `degenerate_1d`, and
more) that `analyse_ablations.py`'s E1 section and the E4 threshold sweep
both read back.

Key environment variables: `DATASET`, `MOL_LAYOUT`, `SEED`,
`IMPORTANCE_CUTOFF`, `PROCESSED_DIR`, `OUTPUT_DIR`, `TABNET_IDX_DIR`,
`TABNET_STEP_CSV_PATH`, plus two pairs specific to individual layouts:
`BASE_LAYOUT`/`PERMUTATION_SEED` (only read when `MOL_LAYOUT=shuffled`,
the E2 permutation control) and `AM_VARIANT` (only read when
`MOL_LAYOUT=attention_map`; one of `full`/`flat`/`1row`/`nonorm`, the E3
decomposition).

### `unified_layouts.py`

The five (six, counting the permutation-control wrapper) layout classes,
each exposing `compute_image_shape()` and either `map_feature()` or
`map_feature_by_name()`:

- `StepRowLayout` — one row per decision step, columns sorted by
  within-step importance.
- `StepSparseLayout` — fixed-width variant of StepRow; wraps within the
  band rather than truncating when a step has more features than the
  configured width.
- `PackedLayout` (`transpose=False`/`True` for row-major/column-major) —
  sorts by **global** importance only, ignoring step boundaries entirely,
  making it a genuinely step-agnostic baseline against the step-aware
  layouts above.
- `AttentionMapLayout` — the full `(K, F)` attention-weighted matrix.
  Accepts a `variant` parameter (`full`/`flat`/`1row`/`nonorm`) for the E3
  decomposition; `flat` replaces the per-step attention distribution with
  uniform weighting, `1row` collapses the K rows to their mean, `nonorm`
  sets `skip_normalization` so `tabnet_image_builder.py` leaves pixel
  values on the standardised scale instead of rescaling to [0,1].
- `ShuffledLayout` — wraps any of the coordinate-based layouts above and
  applies a fixed random permutation to the feature-to-pixel assignment
  (not the coordinates themselves), for the E2 permutation control. Its
  `base_name` property lets the image builder route packed layouts, which
  only implement `map_feature_by_name()`, correctly even when wrapped.

`create_layout(name, step_df, **kwargs)` is the factory function every
other script should call rather than instantiating these classes directly.

### `mol_visualizations.py`

Per-run qualitative diagnostics: class grids, single-instance images,
step-activation charts, per-class average images. Independent of the
metrics pipeline — reads the same image tensors but never contributes to
any reported number. Normalises purely for its own display purposes.

---

## 4. CNN Training and Evaluation (`cnn/`)

### `train_cnn.py`

Trains a CNN on the images `tabnet_image_builder.py` produced. Reads
training data from `IMAGE_DIR` (shared, read-only across every
architecture that might train on the same images) and writes its
checkpoint and config to `IMAGE_DIR/arch_{CNN_ARCH}/` (per-architecture,
so E1's four architectures never overwrite each other's results on the
same images). `CNN_ARCH` defaults to `tabnetcnn` — every call that doesn't
set it explicitly behaves exactly as if this split didn't exist.

Selects the best checkpoint by macro-averaged ROC-AUC on an internal
validation split (not accuracy), with early stopping (`CNN_PATIENCE`,
default 20 epochs). cuDNN deterministic mode is enabled.

Writes `best_model_{DATASET}_{LAYOUT}_seed{SEED}.pth` and
`cnn_config_{DATASET}_{LAYOUT}_seed{SEED}.json` (including `architecture`
and `n_parameters`) to `TASK_OUTPUT_DIR`.

Key environment variables: `DATASET`, `MOL_LAYOUT`, `SEED`, `N_CLASSES`
(overrides the fallback `y_train.max()+1`, which undercounts when a rare
class is absent from this specific fold), `OUTPUT_DIR`, `CNN_ARCH`,
`CNN_LEARNING_RATE`, `CNN_OPTIMIZER`, `CNN_DROPOUT`, `CNN_BATCH_SIZE`,
`CNN_EPOCHS`, `CNN_PATIENCE`, plus the same `BASE_LAYOUT`/
`PERMUTATION_SEED`/`AM_VARIANT` trio as the image builder, needed here
only to reconstruct the matching directory tag.

### `evaluate_cnn.py`

Loads the checkpoint `train_cnn.py` saved, reconstructs the matching
architecture via `cnn_architectures.build_model()` (reading which one was
actually trained from the saved config, not assuming `tabnetcnn`),
evaluates on the held-out test images from `IMAGE_DIR`, and writes
`cnn_evaluation_results_{LAYOUT}.json` to `TASK_OUTPUT_DIR` —
`{accuracy, balanced_accuracy, f1_macro, f1_weighted, roc_auc, ...}`, plus
`architecture` and `n_parameters` for provenance.

**This exact file path —
`{IMAGE_DIR}/{layout_tag}/arch_{CNN_ARCH}/cnn_evaluation_results_{LAYOUT}.json`
— is read back independently by three different places**:
`benchmark_parallel.py`'s `run_agt2i_fold()` (main benchmark and HPO),
`analyse_ablations.py`'s E1 section, and each E3/E4/E6 orchestrator. If you
ever change this path again, all four need updating together — a mismatch
here fails silently (empty dict, every metric defaults to `NaN`) rather
than raising, which is exactly what happened once already this project.

### `cnn_model.py`

`TabNetCNN` — the reference architecture. Two convolutional blocks (no
spatial downsampling — global pooling happens immediately after), channel
width bucketed by pixel count (`(8,16)`/`(16,32)`/`(32,64)` for
small/medium/large images), 3×3 kernels except 2×2 for the small bucket.

### `cnn_architectures.py`

The architecture registry for E1. `ARCHITECTURES` maps a name to an
`nn.Module` class: `tabnetcnn` (re-exports the reference architecture
above), `deep_cnn` (three blocks, spatial downsampling), `small_resnet`
(residual blocks, constant resolution), `pixel_mlp` (flattened input, no
spatial structure at all — the control that can't exploit adjacency,
regardless of layout). `build_model(name, n_classes, input_channels,
image_height, image_width, dropout)` is the factory function; every other
script imports from here rather than the individual classes.

---

## 5. Orchestration (`running_all_models/`)

### `hyperparameter_search.py`

Tunes every baseline (`tune_single_model`) and every AG-T2I layout
(`tune_agt2i_layout`), then runs the final benchmark.

Baselines: `RandomizedSearchCV` with a custom scorer
(`make_scorer(_safe_auc, response_method="predict_proba")` — not
`needs_proba`, which is a silently-ignored dead parameter in current
scikit-learn), 3-fold stratified CV over the *full* dataset (not nested —
declared, not hidden, in the thesis). `HPO_SUBSAMPLE_SIZE` (default
20,000) applies only to Poker Hand and Forest Cover Type, only for this
tuning step; the final benchmark always trains on the complete outer fold.

AG-T2I layouts: Optuna (TPE, the library default sampler), jointly tuning
TabNet and CNN hyperparameters, 25 trials, one independent study per
layout — see `run_e6_shared_backbone.py` if you specifically want a shared
backbone instead.

`--fresh` deletes `best_params/`, Optuna study databases, and the TabNet
cache before starting. `--model` accepts one or more values, mixing
baseline names and `AG-T2I-{layout}` freely in the same call.

### `benchmark_parallel.py`

The lower-level execution engine `hyperparameter_search.py` calls for the
final evaluation step. `run_dataset_benchmark()` builds the outer 5-fold
split — **independently for each of the three training seeds**
(`StratifiedKFold(random_state=seed)`, not a single global seed shared
across all three) — and dispatches each `(model, seed, fold)` combination
to `run_model_on_fold()` (baselines) or `run_agt2i_fold()` (AG-T2I
layouts).

`run_agt2i_fold()` caches TabNet training per `(dataset, seed, fold,
tabnet_params)` via `_fold_id()` — a content hash, not just a fold index,
since different AG-T2I layouts may have different tuned TabNet
hyperparameters and must not share a cache entry. `load_agt2i_params()`
reads a dataset's tuned parameters back out of `best_params/{dataset}.json`
into `(tabnet_params, cnn_params)` pairs, keyed by layout — every ablation
orchestrator that reuses a "already-tuned" backbone calls this function
rather than re-parsing the JSON itself.

### `benchmark.py`

An earlier, sequential predecessor to `benchmark_parallel.py`. **Not used
for any result reported in the thesis** — its own docstring explains why
(five seeds instead of three, a different orchestration path through
`api.py`). Not deleted in case of intentional exploratory use, but treat
any numbers it produces as belonging to a different, uncompared protocol.

### `models_factory.py`

Builds every baseline model. Two things worth knowing if you're extending
this file: `TabNetClassifier` here is a **local subclass**
(`class TabNetClassifier(ClassifierMixin, _TabNetClassifierBase)`) that
adds the `ClassifierMixin` the real `pytorch-tabnet` library's class
doesn't inherit — without it, `is_classifier()` returns `False` and
scikit-learn's `response_method="predict_proba"` scoring rejects it as
"a regressor". `T2I_CNN` and `FTTransformerWrapper` need the same
`ClassifierMixin`-before-`BaseEstimator` ordering for the same reason
(Python's MRO resolves `__sklearn_tags__()` from whichever mixin comes
first), plus an explicit `self.classes_ = np.arange(self.n_classes)` in
`fit()`, since their output width is fixed by construction and shouldn't
be inferred from `np.unique(y)` on a single fold.

`T2I_CNN` accepts an `arch` parameter (default `tabnetcnn`) so the
baseline image methods can also run under E1's alternative architectures.

### `metrics.py`

`compute_extended_metrics()` — the single implementation of every
classification metric used throughout the project. Multiclass ROC-AUC
always passes `labels=list(range(n_classes))` explicitly; without it,
`roc_auc_score` raises whenever a fold's test split happens to be missing
one class, which is common on small or high-cardinality datasets (Glass,
Soybean).

### `statistical_tests.py`

Implements the protocol described in the thesis exactly: Friedman omnibus
test first; post-hoc comparisons run *only* if it rejects, and then only
as Wilcoxon signed-rank against a single control method with
Holm–Bonferroni correction across that reduced family — not all-pairs,
which would inflate the correction factor past the point where any
difference could reach significance at this sample size. Reports the
attainable p-value floor (`2/2^N` raw, `m × 2/2^N` after correction)
alongside every result, and two comparison families (`complete`: every
dataset, models present on all of them; `common`: every model, datasets
where all completed) so neither incomplete baselines nor incomplete
datasets get silently dropped without saying so.

---

## 6. Execution Utilities (`execution/`)

### `runner.py`

`run_step(name, script_path, env_vars)` — every subprocess call in this
project goes through this one function, returning `(success, output,
metadata)`. Centralising this is why fixing a path or environment-variable
bug in one place propagates correctly everywhere that calls it.

### `validators.py`

Sanity checks for the Streamlit dashboard (`check_tabnet_outputs`,
`validate_dataset_structure`) — confirms expected output files exist and
required columns are present before the UI tries to display them.

---

## 7. Ablation Orchestrators (project root)

Full description of what each isolates is in `README.md`'s
[Ablation Studies](README.md#ablation-studies) section. This is the
implementation detail for each:

### `run_e2_permutation_control.py`

For each `(dataset, base_layout, permutation_seed)`: reconstructs the
*exact* outer split `benchmark_parallel.py` used for the requested
`(seed, fold)` and looks up the *exact* cached TabNet backbone via the same
`_fold_id()`/`load_agt2i_params()` functions `benchmark_parallel.py` itself
uses — never re-derived, always imported, so the two can't drift apart.
Builds `MOL_LAYOUT=shuffled` images from that frozen backbone. Prints a
specific `hyperparameter_search.py` command and skips cleanly if the
matching main-benchmark run doesn't exist yet.

### `run_e3_am_decomposition.py`

Trains one TabNet per dataset (reusing `AG-T2I-attention_map`'s own tuned
hyperparameters), then generates all four AM variants from that one frozen
backbone — not four independent retrainings, which would confound the
variant being tested with ordinary TabNet training-run variation.

### `run_e4_threshold_sensitivity.py`

Trains one TabNet per dataset (reusing `AG-T2I-step_row`'s tuned
hyperparameters), then applies each `θ` value in `{0, 0.001, 0.005, 0.01,
0.02}` as a pure post-hoc filter on that same frozen attention output —
`θ` only ever affects which features clear the retention bar after
training, never TabNet's own training, so sharing the backbone here is a
faithful match to what the parameter actually does, not just a speed
optimisation.

### `run_e6_shared_backbone.py`

Trains one TabNet per `(dataset, fold)` — reusing the *standalone* TabNet
baseline's tuned hyperparameters, a deliberate and documented choice among
several defensible options (see the script's own docstring) — then
generates all five AG-T2I layouts from that one shared backbone, with each
layout's own tuned CNN hyperparameters applied downstream.

### `analyse_ablations.py`

Aggregates all five experiments (E1 included) into printed summaries and
LaTeX table snippets, written to `running_all_models/results/`. This is
the canonical, maintained implementation — `analyse_e1.py` still runs
standalone but is not where future fixes will go; use
`analyse_ablations.py --only e1` instead.

---

## 8. Diagnostics (project root)

### `verify_bugs.py`

Read-only, no network calls, no git commands. Checks that a specific list
of past fixes are actually present in the files on disk — not just that
the files exist, but that each carries the fingerprint of its
corresponding fix (a unique string that only appears in the corrected
version). Run this after pulling changes, before trusting that a fix
described in a changelog actually made it into your working copy.

---

## A note on drift

Several bugs found during this project's development were not logic
errors but **two places agreeing on a convention and then one of them
changing without the other** — a results path moved when architecture
support was added, and the function reading it back wasn't updated to
match; a script's default metric name didn't match what the metrics module
actually calls that value. None of these raised an exception; each
produced silently wrong or missing numbers instead. When you change a file
path, a JSON key name, or a directory-naming convention anywhere in this
codebase, grep for every other place that constructs or reads the same
path before assuming the change is complete.