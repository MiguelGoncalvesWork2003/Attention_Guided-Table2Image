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
- Five AG-T2I layout strategies:
  - StepRow
  - StepSparse
  - PackedRow
  - PackedCol
  - AttentionMap
- **Extended baseline suite** now includes:
  - Real **IGTD** (Iterative Global Tabular Data)
  - IGTD-inspired (MDS-based)
  - **DeepInsight** (t-SNE-based feature embedding)
  - Naive Reshape (random feature ordering)
- All baselines can be tuned individually or alongside AG-T2I layouts.
- End-to-end benchmarking pipeline with 5-fold cross-validation.
- Automatic preprocessing (imputation, encoding, scaling).
- Hyperparameter optimisation via Random Search (baselines) and Bayesian Optimisation (Optuna) (AG-T2I).
- Statistical significance testing (Wilcoxon, Friedman, Nemenyi).
- Interactive Streamlit dashboard.

---

# Repository Structure

```text
.
├── api.py
├── app.py
├── preprocessing/
├── tabnet_fs/
├── image_builder/
├── cnn/
├── running_all_models/
│   ├── benchmark_parallel.py
│   ├── benchmark.py
│   ├── hyperparameter_search.py
│   ├── models_factory.py
│   ├── metrics.py
│   ├── utils.py
│   └── statistical_tests.py
├── external/
│   ├── IGTD/
│   └── DeepInsight/
├── experiments/
├── data/
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
| **StepRow** | One row per TabNet decision step; features sorted by importance. |
| **StepSparse** | Fixed-width version of StepRow (10 columns per step). |
| **PackedRow** | Packs retained features row-major according to global importance. |
| **PackedCol** | Column-major version of PackedRow. |
| **AttentionMap** | Complete attention matrix (steps × features). |

The first four layouts discard features whose average attention is below **θ = 0.005**.

AttentionMap retains every feature.

---

# Benchmarks

Implemented models:

| Model | Category |
|-------|----------|
| XGBoost | Gradient Boosting |
| LightGBM | Gradient Boosting |
| CatBoost | Gradient Boosting |
| TabNet | Deep Learning |
| FT-Transformer (Lite) | Transformer |
| IGTD | CNN-based |
| IGTD-inspired | CNN-based |
| DeepInsight | CNN-based |
| Naive Reshape | CNN-based |
| AG-T2I StepRow | Proposed |
| AG-T2I PackedRow | Proposed |
| AG-T2I PackedCol | Proposed |
| AG-T2I StepSparse | Proposed |
| AG-T2I AttentionMap | Proposed |

Run the benchmark

```bash
python running_all_models/benchmark_parallel.py --dataset Cancer --workers 8
```

Run specific models

```bash
python running_all_models/benchmark_parallel.py --dataset Iris --model DeepInsight

python running_all_models/benchmark_parallel.py --dataset Iris --model DeepInsight IGTD "Naive Reshape"

python running_all_models/benchmark_parallel.py --dataset Iris --model XGBoost AG-T2I-step_row
```

Results are saved under

```text
running_all_models/results/
```

---

# Hyperparameter Optimisation

Tune one model

```bash
python running_all_models/hyperparameter_search.py --dataset Iris --model DeepInsight --fresh
```

Tune multiple models

```bash
python running_all_models/hyperparameter_search.py --dataset Iris --model DeepInsight IGTD "Naive Reshape" --fresh
```

Tune every model

```bash
python running_all_models/hyperparameter_search.py --dataset Iris --fresh
```

Best parameters are automatically stored in

```text
running_all_models/best_params/<dataset>.json
```

and reused during benchmarking.

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

All experiments are deterministic.

- Fixed random seed (42)
- CUDA deterministic mode
- Frozen layouts
- Fixed stratified splits

Running the same experiment with the same dataset and seed reproduces identical results.

---

# Roadmap

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