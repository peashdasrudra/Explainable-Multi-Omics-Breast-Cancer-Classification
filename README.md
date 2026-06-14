<div align="center">

# 🧬 Explainable Multi-Omics Breast Cancer Classification

### Consensus Feature Selection × Ensemble Learning × Cross-Omics SHAP Attribution

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006400?style=for-the-badge)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.44+-B041FF?style=for-the-badge)](https://shap.readthedocs.io)
[![TCGA](https://img.shields.io/badge/TCGA-BRCA-DC143C?style=for-the-badge)](https://portal.gdc.cancer.gov/)

<br/>

**705 Patients** · **1,837 Features** · **4 Omics Layers** · **8 Models** · **22 Figures**

*An end-to-end explainable ML pipeline for breast cancer histological subtype classification*
*using TCGA multi-omics data with novel cross-omics SHAP attribution analysis*

<br/>



</div>

---

<br/>

## 📋 Table of Contents

<details open>
<summary><b>Click to expand</b></summary>

- [🎯 Overview](#-overview)
- [⚡ Key Results](#-key-results)
- [🔬 Scientific Contributions](#-scientific-contributions)
- [🏗️ Pipeline Architecture](#️-pipeline-architecture)
- [📊 Methodology](#-methodology)
- [📈 Results & Analysis](#-results--analysis)
- [🧪 Advanced Analyses](#-advanced-analyses)
- [🖼️ Complete Figure Gallery](#️-complete-figure-gallery)
- [📂 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [📦 Generated Outputs](#-generated-outputs)
- [⚠️ Limitations & Future Work](#️-limitations--future-work)
- [📚 References](#-references)

</details>

<br/>

---

## 🎯 Overview

This thesis develops a **fully explainable machine learning pipeline** for classifying breast cancer histological subtypes — **Infiltrating Ductal Carcinoma (IDC)** vs **Infiltrating Lobular Carcinoma (ILC)** — using integrated multi-omics data from The Cancer Genome Atlas (TCGA) Breast Cancer cohort.

<br/>

<div align="center">

### The Problem

> *Multi-omics cancer classification studies suffer from three critical gaps:*
> *single-method feature selection bias, SMOTE data leakage, and lack of omics-level explainability.*

### Our Solution

> *A reproducible, modular pipeline that selects features by consensus, prevents information leakage,*
> *and quantifies — for the first time — how much each omics layer contributes to the classification.*

</div>

<br/>

### 📊 Dataset at a Glance

<div align="center">

| | Property | Value |
|:--|:---------|:------|
| 🏥 | **Source** | TCGA Breast Cancer (BRCA) Cohort |
| 👥 | **Patients** | 705 |
| 🧬 | **Raw Features** | 1,941 → 1,837 (after content-based deduplication) |
| 📊 | **Omics Layer 1** | mRNA Expression (`rs_*`) — 604 features |
| 🔢 | **Omics Layer 2** | Copy Number Variation (`cn_*`) — 761 features |
| 🧪 | **Omics Layer 3** | DNA Methylation (`mu_*`) — 249 features |
| 🔬 | **Omics Layer 4** | Protein / RPPA (`pp_*`) — 223 features |
| 🎯 | **Target** | IDC (574, 81.4%) vs ILC (131, 18.6%) |
| ⚖️ | **Imbalance** | 4.4 : 1 |

</div>

<br/>

<div align="center">
<img src="outputs/figures/fig_02_label_dist.png" width="500" alt="Class Distribution"/>
</div>

<br/>

---

## ⚡ Key Results

<div align="center">

| | Metric | Value |
|:--|:-------|:------|
| 🏆 | **Best Model** | LightGBM (tuned) |
| 📏 | **F1-Macro** | **0.9174 ± 0.0128** |
| 🎯 | **Accuracy** | **95.0%** |
| 📉 | **AUC-ROC** | **0.9654** |
| 📐 | **MCC** | **0.8363** |
| ✂️ | **Features Used** | 75 / 1,837 (**96% reduction**) |
| 🔗 | **Best Fusion** | Late Fusion — F1 = 0.925, AUC = 0.984 |
| 🧬 | **#1 Biomarker** | E-Cadherin (`pp_E.Cadherin`) |
| 🔍 | **Dominant Omics** | mRNA (54.5%) + Protein (39.0%) = **93.5%** |

</div>

<br/>

<div align="center">
<img src="outputs/figures/fig_06_confusion_best.png" width="420" alt="Best Model Confusion Matrix"/>
<br/>
<sub>Confusion Matrix — LightGBM (tuned): 95% accuracy on IDC/ILC classification</sub>
</div>

<br/>

---

## 🔬 Scientific Contributions

<br/>

### C1 · Three-Stage Consensus Feature Selection
> 🧩 Reduced **1,837 → 75 features** using a novel 3-stage funnel: Variance Threshold → ANOVA + Mutual Information per omics → RF + XGBoost consensus ranking. Multi-source agreement eliminates single-method selection bias.

### C2 · SMOTE-Inside-CV *(Methodological Correction)*
> 🛡️ Applied SMOTE **exclusively within training folds** using `imblearn.Pipeline`, preventing synthetic-sample information leakage. Most published multi-omics papers apply SMOTE before the CV split — a well-documented flaw that inflates metrics.

### C3 · Cross-Omics SHAP Attribution *(Core Novelty)*
> 🔬 Quantified, for the first time on TCGA BRCA histological classification, each omics layer's contribution:
> **mRNA: 54.5%** · **Protein: 39.0%** · Methylation: 6.0% · CNV: 0.5%

### C4 · Biological Validation
> 🧬 Top feature is **E-Cadherin** (`pp_E.Cadherin`) — whose loss is the **defining hallmark of ILC** in clinical pathology. \#2 is **CDH1 methylation** (`mu_CDH1`), which silences the E-cadherin gene. This validates both the model and the SHAP framework.

### C5 · Late Fusion Outperforms Early Fusion
> 🔗 Per-omics models with soft voting (**F1 = 0.925**) outperform concatenated Stacking (**F1 = 0.900**), showing that separate models preserve layer-specific discriminative patterns.

### C6 · Statistical Honesty
> 📊 Pairwise Wilcoxon tests show no significant differences between top models at p < 0.05 with 5 folds. Reported transparently — demonstrating methodological maturity over over-claiming.

<br/>

---

## 🏗️ Pipeline Architecture

```
                         ┌─────────────────────────┐
                         │   TCGA BRCA Multi-Omics  │
                         │  705 × 1,837 features    │
                         └────────────┬────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │  Content Deduplication   │
                         │    1,941 → 1,837         │
                         └────────────┬────────────┘
                                      │
              ┌───────────────────────▼───────────────────────┐
              │         3-STAGE FEATURE SELECTION              │
              │                                                │
              │   Stage 1 ─ Variance Threshold (0.01)          │
              │   1,837 → 1,837                                │
              │                                                │
              │   Stage 2 ─ ANOVA + MI (per-omics, k=75)      │
              │   1,837 → 472                                  │
              │                                                │
              │   Stage 3 ─ RF + XGBoost Consensus             │
              │   472 → 75                                     │
              └───────────────────────┬───────────────────────┘
                                      │
              ┌───────────────────────▼───────────────────────┐
              │        CLASSIFICATION (SMOTE-inside-CV)        │
              │                                                │
              │   5 Baselines · LR, SVM, KNN, NB, RF          │
              │   2 Tuned    · XGBoost, LightGBM               │
              │   1 Ensemble · Stacking (RF + XGB + LGBM)      │
              └───────────────────────┬───────────────────────┘
                                      │
              ┌───────────────────────▼───────────────────────┐
              │           EXPLAINABILITY (SHAP)                │
              │                                                │
              │   Global Beeswarm · Cross-Omics Attribution    │
              │   Per-Class IDC/ILC · Dependence Plots         │
              │   Patient-Level Waterfall                      │
              └───────────────────────┬───────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
┌────────▼─────────┐   ┌─────────────▼──────────────┐   ┌─────────▼────────┐
│  FUSION COMPARE  │   │   STATISTICAL RIGOUR       │   │  ABLATION STUDY  │
│  Early vs Late   │   │  Wilcoxon · CV Stability   │   │  Leave-1-out     │
│  Fusion          │   │  Learning · PR Curves      │   │  Single-omics    │
└──────────────────┘   └────────────────────────────┘   └──────────────────┘
```

<br/>

---

## 📊 Methodology

### Feature Selection Funnel

<div align="center">
<img src="outputs/figures/fig_01_funnel.png" width="600" alt="Feature Funnel"/>
<br/>
<sub>Three-stage consensus funnel: 1,837 → 1,837 → 472 → 75 features</sub>
</div>

<br/>

| Stage | Method | In → Out | Rationale |
|:-----:|:-------|:--------:|:----------|
| **1** | Variance Threshold (0.01) | 1,837 → 1,837 | Remove near-zero variance features |
| **2** | ANOVA + Mutual Information | 1,837 → 472 | Per-omics top-75 by each method (union) |
| **3** | RF + XGB Consensus | 472 → **75** | Average tree-based importances |

<br/>

#### 🧬 Top 5 Discovered Biomarkers

| Rank | Feature | Omics | Biological Significance |
|:----:|:--------|:------|:------------------------|
| 🥇 | `pp_E.Cadherin` | Protein | E-cadherin loss = **hallmark of ILC** |
| 🥈 | `mu_CDH1` | Methylation | CDH1 methylation silences E-cadherin |
| 🥉 | `rs_CIDEA` | mRNA | Cell death-inducing factor |
| 4 | `pp_AR` | Protein | Androgen receptor — differs in IDC vs ILC |
| 5 | `rs_SOX10` | mRNA | Neural crest transcription factor |

<div align="center">
<img src="outputs/figures/fig_03_consensus_features.png" width="600" alt="Top 20 Features"/>
<br/>
<sub>Top 20 consensus features colored by omics layer origin</sub>
</div>

<br/>

### SMOTE-Inside-CV: Preventing Data Leakage

```python
# ❌ WRONG (data leakage) — common in published papers
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)  # Leaks across folds!
cross_val_score(model, X_resampled, y_resampled, cv=5)

# ✅ CORRECT (this thesis) — SMOTE exclusively inside training folds
from imblearn.pipeline import Pipeline as ImbPipeline
pipeline = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),     # Applied only on train fold
    ("clf", LGBMClassifier()),
])
cross_val_score(pipeline, X, y, cv=StratifiedKFold(5))
```

<br/>

---

## 📈 Results & Analysis

### 8-Model Comparison

All models evaluated using **Stratified 5-Fold CV** with **SMOTE-inside-CV** pipeline.

<div align="center">

| Model | F1-Macro | AUC-ROC | MCC | Accuracy |
|:------|:--------:|:-------:|:---:|:--------:|
| KNN (k=5) | 0.781 ± 0.039 | 0.921 | 0.613 | 83.3% |
| Naive Bayes | 0.768 ± 0.024 | 0.916 | 0.572 | 83.0% |
| Logistic Regression | 0.846 ± 0.033 | 0.925 | 0.697 | 90.1% |
| SVM (RBF) | 0.892 ± 0.023 | 0.958 | 0.788 | 93.8% |
| Random Forest | 0.893 ± 0.024 | 0.957 | 0.790 | 93.8% |
| Stacking Ensemble | 0.900 ± 0.013 | 0.963 | 0.804 | 94.2% |
| XGBoost (tuned) | 0.905 ± 0.018 | 0.965 | 0.813 | 94.3% |
| **LightGBM (tuned)** | **0.917 ± 0.013** | **0.965** | **0.836** | **95.0%** |

</div>

<br/>

<div align="center">
<img src="outputs/figures/fig_05_model_comparison.png" width="600" alt="Model Comparison"/>
<br/><br/>
<img src="outputs/figures/fig_04_roc_all.png" width="550" alt="ROC Curves"/>
<br/>
<sub>ROC Curves — All models achieve AUC > 0.91; LightGBM and XGBoost lead</sub>
</div>

<br/>

### 🔬 Cross-Omics SHAP Attribution *(Core Finding)*

<div align="center">

| Omics Layer | Attribution | Status |
|:------------|:----------:|:------:|
| **mRNA Expression** | **54.5%** | 🟢 Dominant |
| **Protein (RPPA)** | **39.0%** | 🟢 Dominant |
| DNA Methylation | 6.0% | 🟡 Moderate |
| Copy Number Variation | 0.5% | 🔴 Negligible |

</div>

> **Key Finding:** mRNA + Protein together account for **93.5%** of the SHAP attribution. This is independently confirmed by the ablation study (removing Protein causes the largest performance drop). E-Cadherin loss — the defining hallmark of ILC — emerges as the #1 feature.

<div align="center">
  <img src="outputs/figures/fig_07_shap_beeswarm.png" width="700" alt="SHAP Beeswarm"/>
<br/>
<sub>Global SHAP Feature Importance — E-Cadherin protein dominates IDC vs ILC classification</sub>
<img src="outputs/figures/fig_08_omics_attribution.png" width="650" alt="Omics Attribution"/>
<br/>
<sub>Cross-Omics SHAP Attribution — mRNA and Protein dominate IDC vs ILC classification</sub>
</div>

<br/>

### Per-Class SHAP Analysis

<div align="center">
<img src="outputs/figures/fig_09_shap_IDC.png" width="48%" alt="SHAP IDC"/>
<img src="outputs/figures/fig_09_shap_ILC.png" width="48%" alt="SHAP ILC"/>
<br/>
<sub>Left: Features driving IDC classification | Right: Features driving ILC classification</sub>
</div>

<br/>

### SHAP Dependence — E-Cadherin Threshold Effect

When E-Cadherin expression drops below ~−0.5 (standardized), the model strongly predicts ILC — consistent with the known E-cadherin loss mechanism in lobular carcinoma.

<div align="center">
<img src="outputs/figures/fig_17_shap_dependence_ecadherin.png" width="550" alt="SHAP Dependence"/>
</div>

<br/>

### Patient-Level Waterfall Explanations

<div align="center">
<img src="outputs/figures/fig_10_waterfall_p1.png" width="48%" alt="Waterfall IDC"/>
<img src="outputs/figures/fig_10_waterfall_p2.png" width="48%" alt="Waterfall ILC"/>
<br/>
<sub>Left: IDC patient prediction | Right: ILC patient — E-Cadherin drives +3.23 toward ILC</sub>
</div>

<br/>

### Fusion Strategy Comparison

<div align="center">

| Strategy | F1-Macro | AUC-ROC | Architecture |
|:---------|:--------:|:-------:|:-------------|
| Early Fusion | 0.900 ± 0.013 | 0.963 | Concatenate → Stacking Ensemble |
| **Late Fusion** | **0.925** | **0.984** | Per-omics XGBoost → Soft Vote |

</div>

<div align="center">
<img src="outputs/figures/fig_11_fusion_comparison.png" width="550" alt="Fusion"/>
<img src="outputs/figures/fig_12_confusion_late.png" width="380" alt="Late Fusion CM"/>
</div>

<br/>

---

## 🧪 Advanced Analyses

### Statistical Significance (Wilcoxon Signed-Rank)

<div align="center">
<img src="outputs/figures/fig_13_significance_heatmap.png" width="480" alt="Significance"/>
<img src="outputs/figures/fig_14_cv_stability.png" width="480" alt="CV Stability"/>
<br/>
<sub>Left: Pairwise p-values | Right: Per-fold F1 stability — LGBM and XGB show tightest IQR</sub>
</div>

<br/>

### Omics Ablation Study

*"What happens if you remove each omics layer?"*

<div align="center">

**Leave-One-Layer-Out:**

| Configuration | F1-Macro | Δ from Full |
|:--------------|:--------:|:-----------:|
| ✅ All Omics (baseline) | **0.894** | — |
| ❌ Remove Protein (26) | 0.864 | **−0.030** |
| ❌ Remove mRNA (45) | 0.881 | −0.013 |
| ❌ Remove Methylation (1) | 0.881 | −0.013 |
| ❌ Remove CNV (3) | 0.889 | −0.005 |

**Single-Omics:**

| Configuration | F1-Macro | Features |
|:--------------|:--------:|:--------:|
| Only Protein | **0.862** | 26 |
| Only Methylation | 0.795 | 1 |
| Only mRNA | 0.776 | 45 |
| Only CNV | 0.592 | 3 |

</div>

<div align="center">
<img src="outputs/figures/fig_15_ablation_study.png" width="650" alt="Ablation"/>
<br/>
<sub>Green = full model | Red = leave-one-out | Blue = single-omics</sub>
</div>

<br/>

### Learning Curve & Precision-Recall

<div align="center">
<img src="outputs/figures/fig_18_learning_curve.png" width="48%" alt="Learning Curve"/>
<img src="outputs/figures/fig_20_precision_recall.png" width="48%" alt="PR Curves"/>
<br/>
<sub>Left: Validation still improving — more data could help | Right: XGBoost AP = 0.912</sub>
</div>

<br/>

### Feature Analysis

<div align="center">
<img src="outputs/figures/fig_19_feature_composition.png" width="42%" alt="Composition"/>
<img src="outputs/figures/fig_16_correlation_heatmap.png" width="52%" alt="Correlation"/>
<br/>
<sub>Left: Omics composition of 75 consensus features | Right: Low inter-feature correlation confirms independence</sub>
</div>

<br/>

---

## 🖼️ Complete Figure Gallery

<details>
<summary><b>📁 All 22 Figures (click to expand)</b></summary>

<br/>

| # | Filename | Description |
|:-:|:---------|:------------|
| 01 | `fig_01_funnel.png` | Feature reduction funnel (1,837 → 75) |
| 02 | `fig_02_label_dist.png` | Class distribution (IDC vs ILC) |
| 03 | `fig_03_consensus_features.png` | Top 20 consensus features by omics |
| 04 | `fig_04_roc_all.png` | ROC curves — all 8 models |
| 05 | `fig_05_model_comparison.png` | F1-Macro bar chart comparison |
| 06 | `fig_06_confusion_best.png` | Confusion matrix — LightGBM |
| 07 | `fig_07_shap_beeswarm.png` | Global SHAP beeswarm (top 20) |
| 08 | `fig_08_omics_attribution.png` | Cross-omics SHAP attribution bars |
| 09a | `fig_09_shap_IDC.png` | Per-class SHAP — IDC features |
| 09b | `fig_09_shap_ILC.png` | Per-class SHAP — ILC features |
| 10a | `fig_10_waterfall_p1.png` | Patient waterfall — IDC case |
| 10b | `fig_10_waterfall_p2.png` | Patient waterfall — ILC case |
| 11 | `fig_11_fusion_comparison.png` | Early vs Late fusion |
| 12 | `fig_12_confusion_late.png` | Confusion matrix — Late fusion |
| 13 | `fig_13_significance_heatmap.png` | Wilcoxon p-value heatmap |
| 14 | `fig_14_cv_stability.png` | CV stability box plots |
| 15 | `fig_15_ablation_study.png` | Omics ablation study |
| 16 | `fig_16_correlation_heatmap.png` | Feature correlation matrix |
| 17 | `fig_17_shap_dependence.png` | SHAP dependence — E-Cadherin |
| 18 | `fig_18_learning_curve.png` | Learning curve analysis |
| 19 | `fig_19_feature_composition.png` | Omics composition pie chart |
| 20 | `fig_20_precision_recall.png` | Precision-Recall curves |

</details>

<br/>

---

## 📂 Project Structure

```
📦 Multi-Omics-Cancer-Classification/
│
├── 📁 data/
│   └── brca_data_w_subtypes.csv              # TCGA BRCA multi-omics dataset
│
├── 📁 src/
│   ├── config.py                              # Global constants & hyperparameters
│   ├── utils.py                               # Seed locking, formatted printing
│   ├── data_pipeline.py                       # Data loading, dedup, cleaning
│   ├── feature_selection.py                   # 3-stage consensus funnel
│   ├── baseline_models.py                     # 5 baselines (SMOTE-inside-CV)
│   ├── advanced_models.py                     # XGBoost/LightGBM tuning + Stacking
│   ├── shap_analysis.py                       # Cross-omics SHAP attribution
│   ├── fusion_comparison.py                   # Early vs Late fusion
│   └── visualization.py                       # Publication-quality figures
│
├── 📁 outputs/
│   ├── 📁 figures/                            # 22 publication-quality PNGs
│   ├── 📁 results/                            # 10 CSV result tables
│   ├── 📁 models/                             # Saved .joblib models
│   └── 📁 preprocessed/                       # Feature lists & omics mapping
│
├── run_pipeline.py                            # 🚀 Master script (Day 1–5)
├── advanced_analysis.py                       # 📊 Stats, ablation, learning curves
├── requirements.txt                           # Dependencies
└── README.md                                  # You are here
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone
git clone https://github.com/peashdasrudra/Multi-Omics-Cancer-Classification-CRO-xAI.git
cd Multi-Omics-Cancer-Classification-CRO-xAI

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# Complete pipeline: Data → Features → Models → SHAP → Fusion (~2 min)
python run_pipeline.py

# Advanced analyses: Statistical tests, Ablation, Learning Curves (~1 min)
python advanced_analysis.py
```

### Programmatic Access

```python
from src.data_pipeline import run_data_pipeline
from src.feature_selection import run_feature_selection
from src.shap_analysis import run_shap_analysis

# Load & preprocess
X, y, label_encoder, omics_groups, _ = run_data_pipeline()

# Feature selection (1,837 → 75)
X_final, features, funnel, importance = run_feature_selection(X, y, omics_groups)

# SHAP analysis
attr_df, model, scaler = run_shap_analysis(X_final, y, label_encoder)
print(attr_df)  # Cross-omics attribution table
```

---

## 📦 Generated Outputs

<details>
<summary><b>📊 Result Tables (10 CSVs)</b></summary>

| File | Description |
|:-----|:------------|
| `results_all_models.csv` | 8 models × 6 metrics (mean ± std) |
| `results_all_models_numeric.csv` | Raw numeric values |
| `results_baseline.csv` | 5 baseline results |
| `consensus_importances.csv` | RF + XGB importance rankings |
| `omics_attribution.csv` | Cross-omics SHAP % |
| `fusion_comparison.csv` | Early vs Late fusion |
| `per_fold_f1_scores.csv` | Per-fold F1 (transparency) |
| `statistical_significance.csv` | Pairwise Wilcoxon p-values |
| `ablation_study.csv` | Omics ablation results |
| `feature_final.csv` | 75 final features + omics labels |

</details>

<details>
<summary><b>💾 Saved Models (5 files)</b></summary>

| File | Description |
|:-----|:------------|
| `xgb_tuned.joblib` | Best XGBoost (GridSearchCV) |
| `lgbm_tuned.joblib` | Best LightGBM (GridSearchCV) |
| `stacking.joblib` | Stacking Ensemble |
| `xgb_best_params.joblib` | XGBoost hyperparameters |
| `lgbm_best_params.joblib` | LightGBM hyperparameters |

</details>

---

## ⚠️ Limitations & Future Work

### Current Limitations

| # | Limitation | Impact |
|:-:|:-----------|:-------|
| 1 | Sample size (n=705) | Low power for Wilcoxon tests with 5 folds |
| 2 | Binary classification only | IDC/ILC; no PAM50 molecular subtypes |
| 3 | No external validation | TCGA only; cross-cohort generalization untested |
| 4 | Slight overfitting | Training ≈ 1.0; learning curve gap not fully closed |

### Future Directions

- 🌐 **External validation** on METABRIC or GEO cohorts
- 🔢 **Multi-class extension** with PAM50 subtypes (LumA/LumB/HER2/Basal)
- 🧠 **Deep learning fusion** — multi-modal autoencoders, graph neural networks
- ⏱️ **Survival analysis** — Cox-PH regression with SHAP
- 🧪 **DC-CRO framework** — feature selection as NP-hard Knapsack + Chemical Reaction Optimization

---

## 📚 References

1. Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions.* NeurIPS.
2. Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.
3. TCGA Network. (2012). *Comprehensive molecular portraits of human breast tumours.* Nature.
4. Ciriello, G., et al. (2015). *Comprehensive molecular portraits of invasive lobular breast cancer.* Cell.
5. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* KDD.
6. Ke, G., et al. (2017). *LightGBM: A highly efficient gradient boosting decision tree.* NeurIPS.

---

<div align="center">

**Built for MSc Thesis — Explainable AI for Multi-Omics Cancer Classification**

<sub>Made with 🧬 science and ☕ coffee</sub>

</div>
