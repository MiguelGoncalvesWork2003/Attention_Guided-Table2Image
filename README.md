# Attention-Guided Tabular-to-Image Representations

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Transform tabular data into CNN-compatible images using **supervised feature attention** extracted from **TabNet**.

The pipeline is fully decoupled:

1. Train a TabNet model to learn feature attention.
2. Freeze the learned attention masks.
3. Generate deterministic image layouts from the extracted attention.
4. Train a lightweight CNN on the generated images.

This repository contains the complete implementation used in the paper:

> **Attention-Guided Tabular-to-Image Representations for Deep Learning**

---

## Overview

Unlike traditional tabular-to-image methods that rely on iterative optimization or unsupervised feature similarity, **AG-T2I** constructs spatial layouts directly from supervised feature attention learned by TabNet.

The proposed framework includes five attention-guided layouts:

- StepRow
- StepSparse
- PackedRow
- PackedCol
- AttentionMap

as well as two comparison baselines:

- Naive Reshape
- IGTD-inspired (MDS approximation)

The complete pipeline includes:

- preprocessing
- TabNet attention extraction
- image generation
- CNN training
- benchmark evaluation
- statistical analysis
- interactive visualization

---

## Features

- Supervised attention-guided image generation
- Five deterministic AG-T2I layouts
- Naive and IGTD-inspired baselines
- Complete preprocessing pipeline
- Lightweight CNN for image classification
- Parallel benchmark execution
- Cached preprocessing and TabNet models
- Hyperparameter search
- Streamlit visualization dashboard
- Reproducible experiments with fixed random seeds

---

## Repository Structure

```text
.
├── api.py                         # Command-line API
├── app.py                         # Streamlit dashboard
├── preprocessing/                 # Data preprocessing
├── tabnet_fs/                     # TabNet training and attention extraction
├── image_builder/                 # Image generation layouts
├── cnn/                           # CNN architecture and training
├── running_all_models/
│   ├── benchmark_parallel.py      # Parallel benchmark
│   ├── hyperparameter_search.py   # Hyperparameter optimization
│   ├── models_factory.py
│   ├── metrics.py
│   ├── utils.py
│   └── statistical_tests.py
├── experiments/
├── data/
│   ├── raw/
│   └── processed/
├── cache/
├── figures/
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/MiguelGoncalvesWork2003/Attention_Guided-Table2Image.git

cd Attention_Guided-Table2Image

pip install -r requirements.txt
```

---

# Quick Start

## Run a single AG-T2I pipeline

```bash
python api.py run Cancer --target Class --layout step_row --seed 42
```

This executes:

- preprocessing
- TabNet training
- attention extraction
- image generation
- CNN training
- evaluation

---

## Launch the dashboard

```bash
streamlit run app.py
```

The dashboard allows visual inspection of:

- learned attention masks
- generated layouts
- feature importance
- generated images

---

## Parallel benchmark

Run every model using the optimized parallel benchmark:

```bash
python running_all_models/benchmark_parallel.py --dataset Cancer --workers 8
```

Features:

- parallel execution
- preprocessing cache
- cached TabNet models
- automatic GPU usage (if available)

Results are stored in:

```text
running_all_models/results/
```

---

## Hyperparameter search

Search the best parameters for all models:

```bash
python running_all_models/hyperparameter_search.py Cancer --target Class --agt2i_trials 20
```

Output:

```text
running_all_models/hyperparameter_results/
```

---

## AG-T2I pipeline search

Random Search

```bash
python api.py random Cancer --target Class --trials 50 --jobs 4
```

Bayesian Optimization

```bash
python api.py bayesian Cancer --target Class --trials 50
```

---

# Datasets

Place CSV files inside

```text
data/raw/
```

Example datasets:

- Cancer.csv
- Diabetes.csv
- Glass.csv
- Thyroid.csv
- Card.csv

Each dataset must contain the target column (default: `Class`).

Alternatively specify:

```bash
--target <column_name>
```

---

# Reproducing the Paper

Run all experiments:

```bash
cd running_all_models

python benchmark_parallel.py
```

Statistical analysis:

```bash
python statistical_tests.py
```

The benchmark computes:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

using stratified 5-fold cross-validation.

---

# Citation

If you use this repository, please cite:

```bibtex
@article{bourgingoncalves2026agt2i,
  title={Attention-Guided Tabular-to-Image Representations for Deep Learning},
  author={Bourgin Gonçalves, Miguel and Dutra, Inês},
  journal={Preprint (under review)},
  year={2026},
  note={Available at https://github.com/MiguelGoncalvesWork2003/Attention_Guided-Table2Image}
}
```

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.