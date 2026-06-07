# Attention-Guided Tabular-to-Image Representations

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Transform tabular data into CNN‑compatible images using **supervised feature attention** from [TabNet](https://arxiv.org/abs/2004.13621).  
The pipeline is fully decoupled: TabNet learns which features matter, then a deterministic layout maps them to 2D grids, and finally a lightweight CNN classifies the resulting images.

![Pipeline](experiments/mol_visualization/Cancer/step_row/train_average_per_class.png)   <!-- optional, if you have an image -->

---

## Key Features

- **Task‑aware layouts** – attention masks are frozen *after* TabNet training, so the spatial arrangement reflects true predictive importance.
- **Multiple layout strategies** – StepRow, StepSparse, PackedRow, PackedCol, AttentionMap, plus naive and IGTD‑inspired baselines.
- **Complete pipeline** – preprocessing, TabNet training, image building, CNN training & evaluation.
- **Interactive dashboard** – a Streamlit app (`app.py`) for visual inspection of attention, layouts, and generated images.
- **Reproducible** – fixed seeds, predefined splits, environment variables for all hyperparameters.
- **Benchmark suite** – compare against XGBoost, LightGBM, CatBoost, FT‑Transformer, etc., with extended metrics (balanced accuracy, precision, recall, F1, ROC‑AUC).

---

## Installation

```bash
git clone https://github.com/yourusername/attention-guided-t2i.git
cd attention-guided-t2i
pip install -r requirements.txt