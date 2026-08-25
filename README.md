# Attention-Guided Tabular-to-Image Representations

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AG-T2I** transforms tabular data into CNN-compatible images using **supervised feature attention** learned by TabNet.

The layout is derived from the model's own attention masks — no iterative optimisation, no metaheuristic search. Layout *generation*, given a fixed, already-trained TabNet, is reproducible; the hyperparameter search that selects that TabNet is a stochastic process like any other, reproducible only given the same random seed (see [Reproducibility](#reproducibility) below).

---

# Pipeline at a Glance

```text
           Raw Tabular Data
                  │
          Preprocessing
                  │
   TabNet Training (Frozen)
                  │
     Attention Aggregation
                  │
     Layout Generation (5 variants)
                  │
        Image Construction
                  │
     Lightweight CNN Training
                  │
      Evaluation & Metrics
```

See the accompanying thesis for the complete mathematical formulation and the full experimental protocol.

---

# Features

- Layout generation from frozen TabNet attention — no iterative optimisation, no metaheuristic search.
- Five AG-T2I layout strategies: StepRow, StepSparse, PackedRow, PackedCol, AttentionMap.
- Baseline suite:
  - Tree ensembles: XGBoost, LightGBM, CatBoost, Random Forest
  - Neural: TabNet, MLP, FTT-lite (a reduced FT-Transformer reimplementation — see [Benchmarks](#benchmarks))
  - Image-based: real IGTD, MDS-Layout, DeepInsight, Naive Reshape
- Five ablation experiments isolating specific factors in the framework — see [Ablation Studies](#ablation-studies).
- End-to-end benchmarking pipeline with 5-fold cross-validation, 3 seeds.
- Preprocessing: imputation, encoding, standardisation (neural models only; tree ensembles receive imputed, unscaled data).
- Hyperparameter optimisation via Randomized Search (baselines) and Bayesian Optimisation / Optuna (AG-T2I).
- Statistical significance testing: Friedman omnibus test, Wilcoxon signed-rank against a single control, Holm–Bonferroni correction.
- Interactive Streamlit dashboard.

---

# Repository Structure

```text
.
├── api.py
├── app.py
├── preprocessing/
│   ├── run_preprocessing.py
│   ├── decisions.py
│   ├── transform.py
│   ├── pipeline.py
│   └── preprocessing_utils.py
├── tabnet_fs/
│   └── train_tabnet.py
├── image_builder/
│   ├── tabnet_image_builder.py
│   ├── unified_layouts.py
│   └── mol_visualizations.py
├── cnn/
│   ├── train_cnn.py
│   ├── evaluate_cnn.py
│   ├── cnn_model.py
│   └── cnn_architectures.py       # architecture registry for E1 (layout transfer)
├── execution/
│   ├── runner.py
│   └── validators.py
├── running_all_models/
│   ├── benchmark_parallel.py
│   ├── benchmark.py               # superseded by benchmark_parallel.py — see its own docstring
│   ├── hyperparameter_search.py
│   ├── models_factory.py
│   ├── metrics.py
│   ├── utils.py
│   └── statistical_tests.py
├── run_e2_permutation_control.py  # ablation orchestrators — see Ablation Studies
├── run_e3_am_decomposition.py
├── run_e4_threshold_sensitivity.py
├── run_e6_shared_backbone.py
├── analyse_ablations.py           # aggregates E1–E4, E6 into thesis-ready tables
├── analyse_e1.py                  # superseded by analyse_ablations.py --only e1
├── verify_bugs.py                 # read-only diagnostic; see its own docstring
├── external/
│   └── IGTD/
├── experiments/
├── data/
├── cache/
├── figures/
├── requirements.txt
└── README.md
```

> **One item still needs your confirmation.** `running_all_models/utils.py` exists but wasn't part of the codebase reviewed in producing this update — if it's load-bearing for anything described above, let me know what it contains and this section (and `scripts_explanation.md`) should describe it properly rather than omit it.

---

# Requirements

- Python **3.10+**
- PyTorch ≥ 2.0
- pytorch-tabnet
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- Streamlit *(optional)*
- Optuna *(optional)*
- pyDeepInsight *(for the DeepInsight baseline)*
- scipy
- astropy

Install everything with

```bash
pip install -r requirements.txt
```

---

# Installation

```bash
git clone https://github.com/MiguelGoncalvesWork2003/Attention_Guided-Table2Image.git

cd Attention_Guided-Table2Image

pip install -r requirements.txt
```

## IGTD

Download the original IGTD repository and place its `Scripts` folder inside

```text
external/IGTD/Scripts/
```

The benchmark automatically detects `IGTD_Functions.py`.

## DeepInsight

Install the official Python implementation

```bash
pip install pyDeepInsight
```

No further configuration is required.

---

# Quick Start

## Run the complete AG-T2I pipeline

```bash
python api.py run Cancer --target Class --layout step_row --seed 42
```

This performs:

1. Data preprocessing
2. TabNet training
3. Attention extraction
4. Layout generation
5. Image construction
6. CNN training
7. Evaluation

## Launch the dashboard

```bash
streamlit run app.py
```

The dashboard visualises:

- attention masks
- generated layouts
- feature importance
- generated images
- evaluation metrics

---

# AG-T2I Layout Strategies

| Layout | Description |
|---------|-------------|
| **StepRow** | One row per TabNet decision step; features sorted by importance within each step. |
| **StepSparse** | Fixed-width version of StepRow (10 columns per step, wrapping within the band rather than truncating). |
| **PackedRow** | Packs retained features row-major by *global* importance, ignoring step boundaries. |
| **PackedCol** | Column-major transpose of PackedRow. |
| **AttentionMap** | Complete attention matrix (decision steps × features), attention-weighted and normalised. |

The first four layouts discard features whose global importance is below **θ = 0.005** (configurable via the `IMPORTANCE_CUTOFF` environment variable — see `run_e4_threshold_sensitivity.py` under [Ablation Studies](#ablation-studies)).

AttentionMap retains every feature.

---

# Benchmarks

Implemented models:

| Model | Category | `--model` value |
|-------|----------|------------------|
| XGBoost | Gradient Boosting | `XGBoost` |
| LightGBM | Gradient Boosting | `LightGBM` |
| CatBoost | Gradient Boosting | `CatBoost` |
| Random Forest | Gradient Boosting (bagged trees) | `Random Forest` |
| TabNet | Deep Learning | `TabNet` |
| MLP | Deep Learning | `MLP` |
| FTT-lite | Transformer (reduced reimplementation, not the published FT-Transformer — results should not be read as characterising that architecture) | `FT-Transformer (lite)` |
| IGTD | Image-based | `IGTD` |
| MDS-Layout | Image-based | `MDS-layout` |
| DeepInsight | Image-based | `DeepInsight` |
| Naive Reshape | Image-based | `Naive Reshape` |
| AG-T2I StepRow | Proposed | `AG-T2I-step_row` |
| AG-T2I StepSparse | Proposed | `AG-T2I-step_sparse` |
| AG-T2I PackedRow | Proposed | `AG-T2I-packed` |
| AG-T2I PackedCol | Proposed | `AG-T2I-packed_T` |
| AG-T2I AttentionMap | Proposed | `AG-T2I-attention_map` |

Run the benchmark

```bash
python running_all_models/hyperparameter_search.py --dataset Cancer
```

Run specific models

```bash
python running_all_models/hyperparameter_search.py --dataset Cancer --model DeepInsight

python running_all_models/hyperparameter_search.py --dataset Cancer --model DeepInsight IGTD "Naive Reshape"

python running_all_models/hyperparameter_search.py --dataset Cancer --model XGBoost "AG-T2I-step_row"
```

Run all five AG-T2I layouts for one dataset in a single call:

```bash
python running_all_models/hyperparameter_search.py --dataset Cancer --model "AG-T2I-step_row" "AG-T2I-step_sparse" "AG-T2I-packed" "AG-T2I-packed_T" "AG-T2I-attention_map"
```

Results are saved under

```text
running_all_models/results/
```

`running_all_models/benchmark_parallel.py` is the lower-level orchestrator that `hyperparameter_search.py` calls internally for the final evaluation step; it's rarely invoked directly. `running_all_models/benchmark.py` is an earlier, sequential predecessor — see the warning in its own docstring before using it, since its seed count doesn't match the rest of this codebase.

---

# Hyperparameter Optimisation

Tune one model

```bash
python running_all_models/hyperparameter_search.py --dataset Cancer --model DeepInsight --fresh
```

Tune multiple models

```bash
python running_all_models/hyperparameter_search.py --dataset Cancer --model DeepInsight IGTD "Naive Reshape" --fresh
```

Tune every model, across the full dataset suite

```bash
python running_all_models/hyperparameter_search.py --fresh
```

`--fresh` deletes existing tuned parameters, Optuna studies, and the TabNet cache before starting — back up `running_all_models/best_params/` first if you want to keep what's there.

Best parameters are automatically stored in

```text
running_all_models/best_params/<dataset>.json
```

and reused during benchmarking.

---

# Ablation Studies

Five experiments isolate specific factors in the framework, beyond the main benchmark above. Each has a dedicated orchestrator script at the project root; all are resumable — re-running skips combinations already completed.

| # | Name | What it isolates | Script |
|---|------|-------------------|--------|
| E1 | Layout transfer across architectures | Whether a layout's relative merit survives a change of downstream CNN architecture | manual loop, see script docstrings in `cnn/cnn_architectures.py` and `analyse_e1.py`/`analyse_ablations.py --only e1` |
| E2 | Permutation control | Whether spatial *arrangement* specifically (not feature set or image size) drives performance | `run_e2_permutation_control.py` |
| E3 | AGT2I-AM decomposition | Which of AGT2I-AM's three non-geometric factors (attention weighting, row replication, normalisation) drive its behaviour | `run_e3_am_decomposition.py` |
| E4 | Importance-threshold sensitivity | How the retention threshold θ trades off retained feature count against downstream performance | `run_e4_threshold_sensitivity.py` |
| E6 | Shared attention backbone | Whether differences between layouts reflect geometry, or incidental variation between independently-tuned TabNet backbones | `run_e6_shared_backbone.py` |

(There is no E5 — the numbering is historical, not a gap in what's implemented.)

E3, E4, and E6 each train one TabNet backbone per dataset — reusing already-tuned hyperparameters from the main benchmark — and vary only the one factor under test downstream, rather than retraining independently per condition. This removes a possible confound between the factor being tested and ordinary training-run variation, and is significantly cheaper besides.

Aggregate all five into thesis-ready tables:

```bash
python analyse_ablations.py
python analyse_ablations.py --only e2   # or e1 / e3 / e4 / e6
```

Each orchestrator requires the corresponding AG-T2I layout(s) already tuned via `hyperparameter_search.py` for its target datasets — running an ablation script against an untuned dataset prints exactly which command to run first, rather than falling back to untuned defaults silently.

---

# Outputs

Each experiment generates:

- Trained TabNet model
- Frozen attention masks
- Aggregated attention statistics
- Layout coordinates
- Image tensors
- CNN checkpoint
- Evaluation metrics (CSV and JSON)
- Diagnostic visualisations

---

# Reproducibility

Reproducibility here has two distinct levels, and it matters not to conflate them:

- **Layout generation, given a fixed, already-trained TabNet, is deterministic.** The coordinate-assignment rule involves no randomness of its own.
- **The hyperparameter search that selects that TabNet is a stochastic process** (Randomized Search for baselines, Bayesian Optimisation via Optuna for AG-T2I). It is *reproducible* given the same random seed for the sampler, but it is not deterministic in the sense of exploring the same fixed sequence regardless of seed — a different seed explores a different subset of the search space.

Within the final benchmark:

- The outer 5-fold split is generated **independently for each of the three training seeds** (`StratifiedKFold` with `random_state` set to that seed), not from a single global seed shared across all three. Within one seed, every method is compared on identical folds.
- CUDA/cuDNN deterministic mode is enabled for TabNet and CNN training.
- Internal 80/20 early-stopping splits (TabNet, CNN) use a fixed `random_state=42`, independent of the outer training seed.

Running the same experiment with the same dataset, layout, and full set of hyperparameters reproduces identical images and, given deterministic CUDA execution, identical trained weights. Running the full HPO-to-benchmark pipeline from scratch reproduces the same *distribution* of outcomes, not necessarily the identical numbers on a re-run with a different sampler seed.

---

# Roadmap

- [ ] Complete the ablation suite across the full dataset range (currently run on representative subsets — see each orchestrator's default `--datasets` list)
- [ ] Release pretrained models
- [ ] Add more benchmark datasets
- [ ] Publish accompanying paper
- [ ] Docker support

---

# Citation

```bibtex
@article{bourgingoncalves2026agt2i,
  title={Attention-Guided Tabular-to-Image Representations for Deep Learning},
  author={Bourgin Gonçalves, Miguel and Dutra, Inês},
  year={2026},
  note={Available at https://github.com/MiguelGoncalvesWork2003/Attention_Guided-Table2Image}
}
```

---

# Acknowledgements

Developed as part of the MSc in Data Science at the University of Porto under the supervision of Prof. Inês Dutra.

---

# License

This project is licensed under the MIT License.

See the `LICENSE` file for details.