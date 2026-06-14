<p align="center">
  <h1 align="center">Explainable Multi-Omics Breast Cancer Classification</h1>
  <p align="center">
    <strong>Consensus Feature Selection | Ensemble Learning | Cross-Omics SHAP Attribution</strong>
    <br/>
    TCGA BRCA Cohort &mdash; 705 Patients &times; 1,837 Features &times; 4 Omics Layers
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/ML-scikit--learn%20|%20XGBoost%20|%20LightGBM-orange?style=flat-square" alt="ML"/>
  <img src="https://img.shields.io/badge/XAI-SHAP-green?style=flat-square" alt="SHAP"/>
  <img src="https://img.shields.io/badge/Data-TCGA%20BRCA-red?style=flat-square" alt="TCGA"/>
  <img src="https://img.shields.io/badge/Figures-22-purple?style=flat-square" alt="Figures"/>
  <img src="https://img.shields.io/badge/Best%20F1-0.917-brightgreen?style=flat-square" alt="F1"/>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Results at a Glance](#key-results-at-a-glance)
- [Scientific Contributions](#scientific-contributions)
- [Methodology](#methodology)
  - [Pipeline Architecture](#pipeline-architecture)
  - [3-Stage Feature Selection](#3-stage-consensus-feature-selection)
  - [SMOTE-Inside-CV](#smote-inside-cv-preventing-data-leakage)
  - [Cross-Omics SHAP Attribution](#cross-omics-shap-attribution)
- [Results & Analysis](#results--analysis)
  - [8-Model Comparison](#8-model-comparison)
  - [Statistical Significance](#statistical-significance)
  - [Omics Ablation Study](#omics-ablation-study)
  - [SHAP Explainability](#shap-explainability)
  - [Fusion Strategy Comparison](#fusion-strategy-comparison)
- [Figures Gallery](#figures-gallery)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Generated Outputs](#generated-outputs)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)

---

## Overview

This thesis develops a **fully explainable machine learning pipeline** for classifying breast cancer histological subtypes -- **Infiltrating Ductal Carcinoma (IDC)** vs **Infiltrating Lobular Carcinoma (ILC)** -- using integrated multi-omics data from the TCGA Breast Cancer (BRCA) cohort.

The pipeline addresses three key gaps in the existing multi-omics classification literature:

1. **Single-method feature selection bias** -- solved by a novel 3-stage consensus funnel
2. **Data leakage from SMOTE** -- solved by imblearn Pipeline with SMOTE inside CV folds
3. **Lack of omics-level explainability** -- solved by cross-omics SHAP attribution analysis

### Dataset

| Property | Value |
|:---------|:------|
| **Source** | [TCGA Breast Cancer (BRCA)](https://portal.gdc.cancer.gov/) |
| **Patients** | 705 |
| **Raw Features** | 1,941 (1,837 after content-based deduplication) |
| **Omics Layer 1** | mRNA Expression (`rs_*`) -- 604 features |
| **Omics Layer 2** | Copy Number Variation (`cn_*`) -- 761 features |
| **Omics Layer 3** | DNA Methylation (`mu_*`) -- 249 features |
| **Omics Layer 4** | Protein / RPPA (`pp_*`) -- 223 features |
| **Target** | `histological.type`: IDC (574, 81.4%) vs ILC (131, 18.6%) |
| **Imbalance Ratio** | 4.4:1 |

<p align="center">
  <img src="outputs/figures/fig_02_label_dist.png" width="500" alt="Class Distribution"/>
</p>

---

## Key Results at a Glance

| Metric | Value |
|:-------|:------|
| **Best Model** | LightGBM (tuned) |
| **F1-Macro** | **0.9174 ± 0.0128** |
| **Accuracy** | **95.0%** |
| **AUC-ROC** | **0.9654** |
| **MCC** | **0.8363** |
| **Features Used** | 75 out of 1,837 (96% reduction) |
| **Best Fusion** | Late Fusion -- F1=0.925, AUC=0.984 |
| **Top Biomarker** | E-Cadherin (pp_E.Cadherin) |
| **Dominant Omics** | mRNA (54.5%) + Protein (39.0%) = 93.5% |

---

## Scientific Contributions

### C1. Three-Stage Consensus Feature Selection

> Reduced 1,837 multi-omics features to 75 using a novel 3-stage funnel: Variance Threshold → ANOVA + Mutual Information per omics group → RF + XGBoost consensus importance ranking. Multi-source agreement eliminates single-method bias.

### C2. SMOTE-Inside-CV (Methodological Correction)

> Used `imblearn.Pipeline` to apply SMOTE **exclusively within training folds**, preventing information leakage. The majority of published multi-omics classification papers apply SMOTE before the cross-validation split -- a well-documented flaw that inflates reported performance metrics.

### C3. Cross-Omics SHAP Attribution (Core Novelty)

> Quantified, for the first time on TCGA BRCA histological classification, how much each omics layer contributes to distinguishing IDC from ILC:
> - **mRNA Expression: 54.5%** of classification signal
> - **Protein (RPPA): 39.0%**
> - DNA Methylation: 6.0%
> - Copy Number Variation: 0.5%

### C4. Biological Validation

> The top feature is **E-Cadherin (pp_E.Cadherin)**, a protein whose loss is the **defining hallmark of ILC** in clinical pathology. The #2 feature is **CDH1 methylation (mu_CDH1)**, which silences the E-cadherin gene. This independently validates both the model and the SHAP attribution framework.

### C5. Late Fusion Outperforms Early Fusion

> Training separate per-omics models and averaging predictions (soft vote) outperforms concatenated feature-space Stacking (F1: 0.925 vs 0.900), demonstrating that per-omics models preserve layer-specific discriminative patterns.

### C6. Statistical Honesty

> Pairwise Wilcoxon signed-rank tests show no statistically significant differences between top models at p<0.05 with 5 CV folds. We report this transparently rather than over-claiming superiority -- demonstrating methodological maturity.

---

## Methodology

### Pipeline Architecture

```
                    TCGA BRCA Multi-Omics Data
                    (705 patients × 1,837 features)
                              │
                    ┌─────────▼──────────┐
                    │ Content-Based      │
                    │ Deduplication      │
                    │ (1,941 → 1,837)   │
                    └─────────┬──────────┘
                              │
             ┌────────────────▼─────────────────┐
             │   3-STAGE FEATURE SELECTION       │
             │                                   │
             │  Stage 1: Variance Threshold      │
             │  (1,837 → 1,837)                  │
             │                                   │
             │  Stage 2: ANOVA + MI per omics    │
             │  (1,837 → 472)                    │
             │                                   │
             │  Stage 3: RF + XGB Consensus      │
             │  (472 → 75)                       │
             └────────────────┬─────────────────┘
                              │
          ┌───────────────────▼───────────────────┐
          │     CLASSIFICATION (SMOTE-inside-CV)  │
          │                                       │
          │  5 Baselines: LR, SVM, KNN, NB, RF   │
          │  2 Tuned: XGBoost, LightGBM           │
          │  1 Ensemble: Stacking (RF+XGB+LGBM)   │
          └───────────────────┬───────────────────┘
                              │
              ┌───────────────▼────────────────┐
              │      EXPLAINABILITY (SHAP)     │
              │                                │
              │  Global Beeswarm (top 20)      │
              │  Cross-Omics Attribution (%)   │
              │  Per-Class Analysis (IDC/ILC)  │
              │  Patient-Level Waterfall       │
              │  Dependence Plot (E-Cadherin)  │
              └───────────────┬────────────────┘
                              │
              ┌───────────────▼────────────────┐
              │     FUSION COMPARISON          │
              │                                │
              │  Early: Concat → Stacking      │
              │  Late:  Per-omics → Soft Vote  │
              └───────────────┬────────────────┘
                              │
              ┌───────────────▼────────────────┐
              │     ADVANCED ANALYSES          │
              │                                │
              │  Statistical Significance      │
              │  Omics Ablation Study          │
              │  Learning Curves               │
              │  Precision-Recall Curves       │
              └────────────────────────────────┘
```

### 3-Stage Consensus Feature Selection

The feature selection funnel is the first contribution of this thesis. By requiring **agreement between multiple methods**, we eliminate the bias inherent in any single feature selection approach.

| Stage | Method | Input | Output | Rationale |
|:------|:-------|:------|:-------|:----------|
| **1** | Variance Threshold (0.01) | 1,837 | 1,837 | Remove near-zero variance (all features passed) |
| **2** | ANOVA F-test + Mutual Information | 1,837 | 472 | Per-omics top-75 by each method, UNION selection |
| **3** | RF + XGBoost Consensus | 472 | **75** | Average tree-based importances, keep top-75 |

<p align="center">
  <img src="outputs/figures/fig_01_funnel.png" width="600" alt="Feature Funnel"/>
</p>

The top 5 consensus features are biologically meaningful:

| Rank | Feature | Omics Layer | Biological Role |
|:-----|:--------|:------------|:----------------|
| 1 | `pp_E.Cadherin` | Protein | E-cadherin loss = hallmark of ILC |
| 2 | `mu_CDH1` | Methylation | CDH1 methylation silences E-cadherin |
| 3 | `rs_CIDEA` | mRNA | Cell death inducing factor |
| 4 | `pp_AR` | Protein | Androgen receptor (differs IDC/ILC) |
| 5 | `rs_SOX10` | mRNA | Neural crest TF, basal-like marker |

<p align="center">
  <img src="outputs/figures/fig_03_consensus_features.png" width="600" alt="Top 20 Features"/>
  <br/><em>Top 20 consensus features colored by omics layer</em>
</p>

### SMOTE-Inside-CV (Preventing Data Leakage)

A critical methodological contribution. Most published multi-omics papers apply SMOTE **before** the train/test split, which creates synthetic minority samples that leak information across folds.

```python
# ❌ WRONG (data leakage) -- common in published papers
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)  # Leaks!
cross_val_score(model, X_resampled, y_resampled, cv=5)

# ✅ CORRECT (this thesis) -- SMOTE inside each fold
from imblearn.pipeline import Pipeline as ImbPipeline
pipeline = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("clf", LGBMClassifier()),
])
cross_val_score(pipeline, X, y, cv=StratifiedKFold(5))
```

### Cross-Omics SHAP Attribution

The core novelty: for each omics layer, we compute the **sum of absolute SHAP values** across all features belonging to that layer, then convert to percentages. This answers: *"Which genomic data type contributes most to the classification?"*

<p align="center">
  <img src="outputs/figures/fig_08_omics_attribution.png" width="650" alt="Omics Attribution"/>
  <br/><em>Cross-Omics SHAP Attribution per Histological Subtype</em>
</p>

---

## Results & Analysis

### 8-Model Comparison

All models evaluated using **Stratified 5-Fold Cross-Validation** with **SMOTE-inside-CV** pipeline.

| Model | F1-Macro | AUC-ROC | MCC | Accuracy |
|:------|:---------|:--------|:----|:---------|
| KNN (k=5) | 0.781 ± 0.039 | 0.921 | 0.613 | 83.3% |
| Naive Bayes | 0.768 ± 0.024 | 0.916 | 0.572 | 83.0% |
| Logistic Regression | 0.846 ± 0.033 | 0.925 | 0.697 | 90.1% |
| SVM (RBF) | 0.892 ± 0.023 | 0.958 | 0.788 | 93.8% |
| Random Forest | 0.893 ± 0.024 | 0.957 | 0.790 | 93.8% |
| Stacking Ensemble | 0.900 ± 0.013 | 0.963 | 0.804 | 94.2% |
| XGBoost (tuned) | 0.905 ± 0.018 | 0.965 | 0.813 | 94.3% |
| **LightGBM (tuned)** | **0.917 ± 0.013** | **0.965** | **0.836** | **95.0%** |

<p align="center">
  <img src="outputs/figures/fig_05_model_comparison.png" width="600" alt="Model Comparison"/>
</p>

<p align="center">
  <img src="outputs/figures/fig_04_roc_all.png" width="550" alt="ROC Curves"/>
  <br/><em>ROC Curves -- All models achieve AUC > 0.91</em>
</p>

### Statistical Significance

Pairwise **Wilcoxon signed-rank tests** on per-fold F1 scores. With only 5 CV folds, no statistically significant differences exist between top models at p<0.05 -- an honest finding that strengthens the work by avoiding over-claiming.

<p align="center">
  <img src="outputs/figures/fig_13_significance_heatmap.png" width="500" alt="Significance"/>
  <br/><em>Pairwise Wilcoxon p-values (lower triangular)</em>
</p>

<p align="center">
  <img src="outputs/figures/fig_14_cv_stability.png" width="600" alt="CV Stability"/>
  <br/><em>Per-fold F1 scores -- LightGBM and XGBoost show tightest interquartile ranges</em>
</p>

### Omics Ablation Study

*"What happens if you remove each omics layer?"* -- A question every thesis examiner will ask.

#### Leave-One-Layer-Out

| Configuration | F1-Macro | Drop from Full |
|:--------------|:---------|:---------------|
| **All Omics (baseline)** | **0.894** | -- |
| Remove Protein (26 feats) | 0.864 | **−0.030 (largest)** |
| Remove mRNA (45 feats) | 0.881 | −0.013 |
| Remove Methylation (1 feat) | 0.881 | −0.013 |
| Remove CNV (3 feats) | 0.889 | −0.005 (smallest) |

#### Single-Omics Performance

| Only This Layer | F1-Macro | Features |
|:----------------|:---------|:---------|
| **Only Protein** | **0.862** | 26 |
| Only Methylation | 0.795 | 1 |
| Only mRNA | 0.776 | 45 |
| Only CNV | 0.592 | 3 |

> **Key insight:** Removing Protein causes the largest drop, and Protein alone achieves the highest single-omics F1 -- independently confirming the SHAP attribution finding.

<p align="center">
  <img src="outputs/figures/fig_15_ablation_study.png" width="650" alt="Ablation"/>
  <br/><em>Green = full model, Red = leave-one-out, Blue = single-omics</em>
</p>

### SHAP Explainability

#### Global Feature Importance

<p align="center">
  <img src="outputs/figures/fig_07_shap_beeswarm.png" width="600" alt="SHAP Beeswarm"/>
  <br/><em>Global SHAP beeswarm -- E-Cadherin dominates, followed by CDH1 methylation</em>
</p>

#### Per-Class SHAP Analysis

<p align="center">
  <img src="outputs/figures/fig_09_shap_IDC.png" width="48%" alt="SHAP IDC"/>
  <img src="outputs/figures/fig_09_shap_ILC.png" width="48%" alt="SHAP ILC"/>
  <br/><em>Left: IDC-driving features | Right: ILC-driving features</em>
</p>

#### SHAP Dependence -- E-Cadherin

The SHAP dependence plot reveals a **clear threshold effect**: when E-Cadherin expression drops below ~−0.5 (standardized), the model strongly predicts ILC -- consistent with the known E-cadherin loss mechanism in lobular carcinoma.

<p align="center">
  <img src="outputs/figures/fig_17_shap_dependence_ecadherin.png" width="550" alt="SHAP Dependence"/>
</p>

#### Patient-Level Waterfall Explanations

<p align="center">
  <img src="outputs/figures/fig_10_waterfall_p1.png" width="48%" alt="Waterfall IDC"/>
  <img src="outputs/figures/fig_10_waterfall_p2.png" width="48%" alt="Waterfall ILC"/>
  <br/><em>Left: IDC patient prediction | Right: ILC patient -- E-Cadherin drives +3.23 toward ILC</em>
</p>

### Fusion Strategy Comparison

| Strategy | F1-Macro | AUC-ROC | Method |
|:---------|:---------|:--------|:-------|
| Early Fusion | 0.900 ± 0.013 | 0.963 | Concatenate all omics → Stacking |
| **Late Fusion** | **0.925** | **0.984** | Per-omics XGBoost → soft vote |

<p align="center">
  <img src="outputs/figures/fig_11_fusion_comparison.png" width="550" alt="Fusion"/>
  <img src="outputs/figures/fig_12_confusion_late.png" width="380" alt="Late Fusion CM"/>
</p>

### Learning Curve & Additional Analyses

<p align="center">
  <img src="outputs/figures/fig_18_learning_curve.png" width="48%" alt="Learning Curve"/>
  <img src="outputs/figures/fig_20_precision_recall.png" width="48%" alt="PR Curves"/>
  <br/><em>Left: Learning curve (gap narrowing but not closed) | Right: Precision-Recall curves (XGBoost AP=0.912)</em>
</p>

<p align="center">
  <img src="outputs/figures/fig_19_feature_composition.png" width="45%" alt="Composition"/>
  <img src="outputs/figures/fig_16_correlation_heatmap.png" width="48%" alt="Correlation"/>
  <br/><em>Left: Omics composition of 75 features | Right: Feature correlation matrix (low redundancy)</em>
</p>

---

## Figures Gallery

| # | Figure | Description |
|:--|:-------|:------------|
| 01 | `fig_01_funnel.png` | Feature reduction funnel (1,837 → 75) |
| 02 | `fig_02_label_dist.png` | Class distribution (IDC vs ILC) |
| 03 | `fig_03_consensus_features.png` | Top 20 consensus features by omics layer |
| 04 | `fig_04_roc_all.png` | ROC curves for all models |
| 05 | `fig_05_model_comparison.png` | F1-Macro comparison (8 models) |
| 06 | `fig_06_confusion_best.png` | Confusion matrix -- best model |
| 07 | `fig_07_shap_beeswarm.png` | Global SHAP beeswarm (top 20) |
| 08 | `fig_08_omics_attribution.png` | Cross-omics SHAP attribution |
| 09 | `fig_09_shap_IDC/ILC.png` | Per-class SHAP summaries |
| 10 | `fig_10_waterfall_p1/p2.png` | Patient-level SHAP waterfall |
| 11 | `fig_11_fusion_comparison.png` | Early vs Late fusion |
| 12 | `fig_12_confusion_late.png` | Late fusion confusion matrix |
| 13 | `fig_13_significance_heatmap.png` | Wilcoxon p-value heatmap |
| 14 | `fig_14_cv_stability.png` | Per-fold CV stability box plot |
| 15 | `fig_15_ablation_study.png` | Omics ablation study |
| 16 | `fig_16_correlation_heatmap.png` | Feature correlation matrix |
| 17 | `fig_17_shap_dependence.png` | SHAP dependence -- E-Cadherin |
| 18 | `fig_18_learning_curve.png` | Learning curve analysis |
| 19 | `fig_19_feature_composition.png` | Omics composition of final features |
| 20 | `fig_20_precision_recall.png` | Precision-Recall curves |

---

## Project Structure

```
Multi Omics Cancer/
│
├── data/
│   └── brca_data_w_subtypes.csv            # TCGA BRCA multi-omics dataset
│
├── src/
│   ├── config.py                            # Global constants & hyperparameters
│   ├── utils.py                             # Seed locking, formatted printing
│   ├── data_pipeline.py                     # Data loading, dedup, cleaning
│   ├── feature_selection.py                 # 3-stage consensus funnel
│   ├── baseline_models.py                   # 5 baselines with SMOTE-inside-CV
│   ├── advanced_models.py                   # XGBoost/LightGBM tuning + Stacking
│   ├── shap_analysis.py                     # Cross-omics SHAP attribution
│   ├── fusion_comparison.py                 # Early vs Late fusion
│   └── visualization.py                     # Publication-quality figures
│
├── outputs/
│   ├── figures/                             # 22 publication-quality PNG figures
│   ├── results/                             # 10 CSV result tables
│   ├── models/                              # Saved trained models (.joblib)
│   └── preprocessed/                        # Feature lists & omics membership
│
├── run_pipeline.py                          # Master script -- runs Day 1-5
├── advanced_analysis.py                     # Statistical tests & ablation study
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- ~2GB RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/peashdasrudra/Multi-Omics-Cancer-Classification-CRO-xAI.git
cd Multi-Omics-Cancer-Classification-CRO-xAI

# Install dependencies
pip install -r requirements.txt
```

### Run Everything

```bash
# Full pipeline: Data → Features → Models → SHAP → Fusion (~2 min)
python run_pipeline.py

# Advanced analyses: Statistics, Ablation, Learning Curves (~1 min)
python advanced_analysis.py
```

### Reproduce Specific Stages

```python
from src.data_pipeline import run_data_pipeline
from src.feature_selection import run_feature_selection
from src.shap_analysis import run_shap_analysis

# Load and preprocess data
X, y, label_encoder, omics_groups, _ = run_data_pipeline()

# Feature selection
X_final, features, funnel, importance = run_feature_selection(X, y, omics_groups)

# SHAP analysis
attr_df, model, scaler = run_shap_analysis(X_final, y, label_encoder)
```

---

## Generated Outputs

### Result Tables (10 CSVs)

| File | Description |
|:-----|:------------|
| `results_all_models.csv` | 8 models × 6 metrics (formatted mean ± std) |
| `results_all_models_numeric.csv` | Raw numeric values for plotting |
| `results_baseline.csv` | 5 baseline model results |
| `consensus_importances.csv` | RF + XGB feature importance rankings |
| `omics_attribution.csv` | Cross-omics SHAP attribution (%) |
| `fusion_comparison.csv` | Early vs Late fusion results |
| `per_fold_f1_scores.csv` | Per-fold F1 scores (for transparency) |
| `statistical_significance.csv` | Pairwise Wilcoxon p-values |
| `ablation_study.csv` | Omics ablation study results |
| `feature_final.csv` | Final 75 features with omics membership |

### Saved Models

| File | Model |
|:-----|:------|
| `xgb_tuned.joblib` | Best XGBoost (GridSearchCV) |
| `lgbm_tuned.joblib` | Best LightGBM (GridSearchCV) |
| `stacking.joblib` | Stacking Ensemble |
| `xgb_best_params.joblib` | XGBoost optimal hyperparameters |
| `lgbm_best_params.joblib` | LightGBM optimal hyperparameters |

---

## Limitations & Future Work

### Current Limitations

1. **Sample size (n=705):** Limits statistical power for significance tests (Wilcoxon with 5 folds)
2. **Binary classification only:** IDC vs ILC; PAM50 molecular subtypes not available in this dataset
3. **No external validation:** Results on TCGA only; cross-cohort generalization not tested
4. **Learning curve gap:** Training score ~1.0 suggests slight overfitting; more data could help

### Future Directions

1. **External validation** on METABRIC or GEO breast cancer cohorts
2. **Multi-class extension** with PAM50 molecular subtypes (LumA/LumB/HER2/Basal)
3. **Deep learning fusion** (multi-modal autoencoders, graph neural networks)
4. **Temporal analysis** using survival data (Cox-PH with SHAP)
5. **DC-CRO framework** -- formulate feature selection as NP-hard Knapsack, solve with Chemical Reaction Optimization

---

## Requirements

```
numpy
pandas
scikit-learn
imbalanced-learn
matplotlib
seaborn
xgboost
lightgbm
shap
boruta
joblib
```

---

## References

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
2. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*.
3. The Cancer Genome Atlas Network. (2012). Comprehensive molecular portraits of human breast tumours. *Nature*.
4. Ciriello, G., et al. (2015). Comprehensive molecular portraits of invasive lobular breast cancer. *Cell*.
5. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
6. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*.

---

<p align="center">
  <em>Built for MSc Thesis -- Explainable AI for Multi-Omics Cancer Classification</em>
</p>
