# Multi-Omics Breast Cancer Classification using DC-CRO and Explainable AI

> **Thesis Project** — Classifying breast cancer subtypes from integrated multi-omics data using Diversity-Controlled Chemical Reaction Optimization (DC-CRO) for feature selection and Explainable AI (xAI) for model interpretability.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Baseline Results](#baseline-results)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

High-dimensional multi-omics data holds immense promise for precision oncology, yet the sheer number of features (**1,800+** molecular markers) introduces noise, redundancy, and the curse of dimensionality. This thesis addresses the problem in three stages:

1. **Baseline Establishment** — Benchmark standard classifiers on the full feature set with rigorous cross-validation to quantify the performance ceiling without feature selection.
2. **DC-CRO Feature Selection** — Apply a Diversity-Controlled Chemical Reaction Optimization metaheuristic to discover compact, high-performance feature subsets.
3. **Explainable AI** — Use SHAP-based explanations to interpret the selected features and provide biologically meaningful insights for clinical decision-making.

---

## Dataset

| Property | Value |
|---|---|
| **Source** | TCGA Breast Cancer (BRCA) cohort |
| **Samples** | ~1,000 patients |
| **Omics Layers** | RNA-Seq gene expression, Copy Number Variation (CNV), Somatic Mutations, Protein expression (RPPA) |
| **Raw Features** | 1,936 molecular features |
| **After Deduplication** | 1,837 unique features (99 content-duplicate CNV columns removed) |
| **Clinical Targets** | ER Status, HER2 Final Status, Histological Type |

The raw data file (`data/brca_data_w_subtypes.csv`) integrates all four omics layers with clinical annotations.

---

## Methodology

### Preprocessing Pipeline

```
Raw CSV ──► Content-Based Deduplication ──► Target Encoding ──► Train/Test Split (80/20, Stratified)
                                                                        │
                                                                        ▼
                                                               StandardScaler (fit on train only)
                                                                        │
                                                                        ▼
                                                               SMOTE Oversampling (train only)
```

**Key design decisions for scientific rigor:**

- **Content-based deduplication** — Removes columns with identical values (not just identical names), eliminating 99 redundant CNV features arising from shared chromosomal loci.
- **No data leakage** — Scaling is fit exclusively on training data; SMOTE is applied inside cross-validation folds via `imblearn.pipeline.Pipeline`.
- **Stratified splitting** — Class proportions are preserved in both train and test sets.
- **Feature name preservation** — Column names are saved as `.npy` arrays for downstream SHAP explainability plots.

### Baseline Evaluation

Three classifiers are evaluated using **Stratified 5-Fold Cross-Validation** with SMOTE applied per-fold:

| Classifier | Configuration |
|---|---|
| **Random Forest** | 100 estimators, default hyperparameters |
| **SVM** | RBF kernel, probability estimates enabled |
| **XGBoost** | Default hyperparameters, logloss evaluation metric |

Metrics reported: Accuracy, F1 (Weighted & Macro), Precision, Recall, ROC-AUC, MCC.

---

## Project Structure

```
Multi Omics Cancer - CRO xAI/
│
├── data/
│   └── brca_data_w_subtypes.csv        # Raw TCGA BRCA multi-omics dataset
│
├── src/
│   ├── __init__.py                     # Package initializer
│   ├── preprocess.py                   # Data loading, deduplication, encoding, SMOTE
│   ├── baseline.py                     # Stratified CV evaluation with ImbPipeline
│   └── utils.py                        # Reproducibility (seed locking)
│
├── outputs/
│   ├── baseline_metrics/               # Per-target CSV files with pivoted metrics
│   │   ├── ER.Status_baseline.csv
│   │   ├── HER2.Final.Status_baseline.csv
│   │   └── histological.type_baseline.csv
│   ├── preprocessed/                   # Saved .npy arrays (train/test splits, feature names)
│   └── figures/                        # Plots (populated in later weeks)
│
├── notebooks/                          # Jupyter notebooks for EDA (future)
├── reports/                            # Generated reports and figures for thesis
│
├── run_week1.py                        # Master script — runs full Week 1 pipeline
├── requirements.txt                    # Python dependencies
├── .gitignore
└── README.md
```

---

## Installation

**Prerequisites:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/peashdasrudra/Multi-Omics-Cancer-Classification---CRO-xAI.git
cd Multi-Omics-Cancer-Classification---CRO-xAI

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `numpy`, `pandas` | Data manipulation |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `imbalanced-learn` | SMOTE oversampling & ImbPipeline |
| `xgboost` | Gradient boosted classifier |
| `matplotlib`, `seaborn` | Visualization |
| `joblib` | Model serialization |

---

## Usage

### Run the Full Week 1 Pipeline

```bash
python run_week1.py
```

This will:
1. Load and deduplicate the raw multi-omics dataset
2. For each clinical target (ER Status, HER2 Status, Histological Type):
   - Encode labels, split data (80/20 stratified), scale features
   - Save pre-SMOTE data for leakage-free CV evaluation
   - Apply SMOTE and save resampled data for future DC-CRO use
   - Run baseline evaluation (RF, SVM, XGBoost) with Stratified 5-Fold CV
   - Export metrics to CSV

All outputs are saved to the `outputs/` directory.

---

## Baseline Results

> Results from Stratified 5-Fold Cross-Validation with per-fold SMOTE. Values reported as **mean ± std**.

### ER Status (Binary: Positive / Negative)

| Model | Accuracy | F1 (Weighted) | F1 (Macro) | ROC-AUC | MCC |
|---|---|---|---|---|---|
| **XGBoost** | **0.9316 ± 0.0298** | **0.9323 ± 0.0285** | **0.9097 ± 0.0364** | 0.9547 ± 0.0282 | **0.8220 ± 0.0698** |
| Random Forest | 0.9294 ± 0.0243 | 0.9283 ± 0.0242 | 0.9022 ± 0.0320 | **0.9569 ± 0.0318** | 0.8078 ± 0.0649 |
| SVM | 0.9157 ± 0.0154 | 0.9121 ± 0.0178 | 0.8776 ± 0.0260 | 0.9523 ± 0.0351 | 0.7690 ± 0.0431 |

### HER2 Final Status (Binary: Positive / Negative)

| Model | Accuracy | F1 (Weighted) | F1 (Macro) | ROC-AUC | MCC |
|---|---|---|---|---|---|
| **XGBoost** | **0.9286 ± 0.0233** | **0.9247 ± 0.0273** | **0.8532 ± 0.0565** | **0.9247 ± 0.0567** | **0.7168 ± 0.1057** |
| Random Forest | 0.9056 ± 0.0293 | 0.8895 ± 0.0416 | 0.7679 ± 0.0950 | 0.8904 ± 0.0493 | 0.5921 ± 0.1516 |
| SVM | 0.8687 ± 0.0277 | 0.8253 ± 0.0465 | 0.6057 ± 0.1145 | 0.8889 ± 0.0533 | 0.3303 ± 0.2167 |

### Histological Type (Multi-class)

| Model | Accuracy | F1 (Weighted) | F1 (Macro) | ROC-AUC | MCC |
|---|---|---|---|---|---|
| **XGBoost** | **0.9167 ± 0.0163** | **0.9164 ± 0.0146** | **0.8616 ± 0.0214** | **0.9377 ± 0.0249** | **0.7255 ± 0.0428** |
| SVM | 0.8936 ± 0.0330 | 0.8852 ± 0.0358 | 0.7978 ± 0.0635 | 0.9056 ± 0.0509 | 0.6148 ± 0.1290 |
| Random Forest | 0.8830 ± 0.0218 | 0.8744 ± 0.0229 | 0.7799 ± 0.0399 | 0.9140 ± 0.0424 | 0.5775 ± 0.0834 |

**Key takeaway:** XGBoost consistently outperforms RF and SVM across all targets. The HER2 target shows the most room for improvement (lower MCC), making it a prime candidate for DC-CRO feature selection.

---

## Roadmap

| Week | Milestone | Status |
|---|---|---|
| **Week 1** | Data preprocessing, deduplication, baseline evaluation | ✅ Complete |
| **Week 2** | DC-CRO metaheuristic skeleton & feature selection | 🔲 Planned |
| **Week 3** | DC-CRO integration with classifiers, hyperparameter tuning | 🔲 Planned |
| **Week 4** | SHAP-based explainability analysis | 🔲 Planned |
| **Week 5** | Final evaluation, thesis write-up, and figures | 🔲 Planned |

---

## Reproducibility

All experiments use a fixed random seed (`42`) locked across NumPy, Python's `random` module, and `PYTHONHASHSEED`. Results are fully reproducible given the same environment and dataset.

---

## License

This project is part of an academic thesis. All rights reserved.

---

<p align="center">
  <i>Built for the Department of Computer Science, University of Dhaka</i>
</p>
