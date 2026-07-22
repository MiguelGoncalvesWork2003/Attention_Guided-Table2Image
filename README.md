# Attention-Guided Tabular-to-Image Representations

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AG-T2I** transforms tabular data into CNN-compatible images using **supervised feature attention** learned by TabNet.

The layout is derived deterministically from the model's own attention masks—no iterative optimisation, no stochastic search.

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
 Deterministic Layout Generation
                  │
        Image Construction
                  │
     Lightweight CNN Training
                  │
      Evaluation & Metrics
```

See the accompanying paper for the complete mathematical formulation.

---

# Features

- Deterministic image generation – the same input always produces the same image.
- Supervised layouts directly derived from TabNet attention.
- Five layout strategies:
  - StepRow
  - StepSparse
  - PackedRow
  - PackedCol
  - AttentionMap
- End-to-end benchmarking pipeline with 5-fold cross-validation.
- Automatic preprocessing (imputation, encoding, scaling).
- Hyperparameter optimisation (Random Search + Bayesian Optimisation).
- Statistical significance testing (Wilcoxon, Friedman, Nemenyi).
- Interactive Streamlit dashboard.

---

# Repository Structure

The repository is organised around the four stages of the AG-T2I pipeline.

```text
.
├── api.py                         # Command-line API
├── app.py                         # Streamlit dashboard
├── preprocessing/                 # Data preprocessing
├── tabnet_fs/                     # TabNet training & attention extraction
├── image_builder/                 # Layouts & image construction
├── cnn/                           # CNN architecture, training & evaluation
├── running_all_models/
│   ├── benchmark_parallel.py
│   ├── benchmark.py
│   ├── hyperparameter_search.py
│   ├── models_factory.py
│   ├── metrics.py
│   ├── utils.py
│   └── statistical_tests.py
├── experiments/                   # Output graphs & results
├── data/
│   ├── raw/
│   └── processed/
├── cache/
├── figures/
├── requirements.txt
└── README.md
```

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

---

# Quick Start

## Run the complete AG-T2I pipeline

```bash
python api.py run Cancer --target Class --layout step_row --seed 42
```

This executes:

1. Preprocessing
2. TabNet training
3. Attention extraction
4. Layout generation
5. Image construction
6. CNN training
7. Evaluation

## Launch the interactive dashboard

```bash
streamlit run app.py
```

The dashboard allows you to inspect:

- attention masks
- generated layouts
- feature importance
- generated images
- evaluation metrics

---

# Layout Strategies

| Layout | Description |
|---------|-------------|
| **StepRow** | One row per TabNet decision step; features sorted by importance. |
| **StepSparse** | Fixed-width version of StepRow (10 columns per step). |
| **PackedRow** | Packs retained features row-major according to global importance. |
| **PackedCol** | Column-major version of PackedRow. |
| **AttentionMap** | Complete attention matrix (steps × features); no feature removal. |

The first four layouts discard features whose average attention is below **θ = 0.005**.

The **AttentionMap** keeps every feature.

---

# Benchmarks & Statistical Analysis

The repository includes implementations of:

- XGBoost
- LightGBM
- CatBoost
- Random Forest
- MLP
- TabNet
- FT-Transformer (light)
- Naive Reshape
- IGTD-inspired baseline
- All five AG-T2I layouts

Run the complete benchmark:

```bash
python running_all_models/benchmark_parallel.py --dataset Cancer --workers 8
```

Run statistical significance tests:

```bash
python running_all_models/statistical_tests.py
```

Results are stored in

```
running_all_models/results/
```

---

# Hyperparameter Optimisation

Search the best parameters using a 3-fold inner cross-validation.

```bash
python running_all_models/hyperparameter_search.py Cancer --target Class --agt2i_trials 20
```

The AG-T2I variants optimise both TabNet and CNN parameters.

Results are stored in

```
running_all_models/results_hyperparameter/
```

---

# Outputs

Each experiment produces:

- Trained TabNet model
- Frozen attention masks
- Aggregated attention statistics
- Layout coordinates
- Generated image tensors
- Trained CNN checkpoint
- Evaluation metrics (CSV + JSON)
- Diagnostic visualisations

---

# Reproducibility

All experiments are fully deterministic.

- Fixed global seed (42)
- CUDA deterministic mode
- Frozen deterministic layouts
- Fixed stratified splits

Running the same experiment with the same dataset and seed always produces identical results.

---

# Roadmap

- [ ] Release pretrained TabNet and CNN models
- [ ] Add additional benchmark datasets
- [ ] Publish accompanying paper
- [ ] Docker support

---

# Citation

If you use this repository, please cite

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