# Explainable Multi-Omics Breast Cancer Classification

**Consensus Feature Selection, Ensemble Learning, and Cross-Omics SHAP Attribution on TCGA BRCA**

---

## Abstract

Breast cancer is the most prevalent malignancy worldwide, with histological subtype classification — particularly distinguishing Infiltrating Ductal Carcinoma (IDC) from Infiltrating Lobular Carcinoma (ILC) — remaining critical for treatment planning. While multi-omics data integration promises improved classification accuracy, existing approaches suffer from three key limitations: (1) reliance on single-method feature selection introducing bias, (2) incorrect application of SMOTE oversampling outside cross-validation folds causing data leakage, and (3) lack of omics-layer-level explainability.

This work introduces a complete, reproducible pipeline that addresses all three limitations. We propose a **three-stage consensus feature selection funnel** that combines variance filtering, ANOVA/Mutual Information union selection, and RF/XGBoost consensus ranking to reduce 1,837 features to 75 without single-method bias. We implement **SMOTE-inside-CV** using `imblearn.Pipeline` to guarantee leak-free evaluation under 4.4:1 class imbalance. Most importantly, we introduce **Cross-Omics SHAP Attribution** — aggregating SHAP values by omics layer to quantify each layer's percentage contribution to subtype classification, revealing that Protein features (especially E-Cadherin) drive 40.26% of subtype classification signal, consistent with the known CDH1 loss mechanism in lobular carcinoma.

Evaluated on the TCGA BRCA cohort (705 patients, 4 omics layers), LightGBM (tuned) achieves the top individual score (**F1-Macro = 0.905 ± 0.020**, **AUC-ROC = 0.960 ± 0.034**), while Late Fusion (per-omics soft vote) achieves **F1-Macro = 0.925** and **AUC-ROC = 0.984**. Leak-free nested cross-validation confirms negligible feature selection leakage ($\Delta \text{F1} = 0.021 < 0.03$).

---

## Key Results

### Model Performance (8-Model Comparison)

| Model | F1-Macro | AUC-ROC | MCC |
|:---|:---:|:---:|:---:|
| LightGBM (tuned) | **0.9054 ± 0.0195** | 0.9602 ± 0.0336 | **0.8140 ± 0.0373** |
| XGBoost (tuned) | 0.8986 ± 0.0260 | 0.9634 ± 0.0314 | 0.7992 ± 0.0516 |
| Stacking Ensemble | 0.8959 ± 0.0188 | **0.9655 ± 0.0269** | 0.7955 ± 0.0369 |
| SVM (RBF) | 0.8887 ± 0.0495 | 0.9524 ± 0.0249 | 0.7852 ± 0.0901 |
| Random Forest | 0.8801 ± 0.0377 | 0.9608 ± 0.0273 | 0.7675 ± 0.0698 |
| Logistic Regression | 0.8409 ± 0.0226 | 0.9231 ± 0.0368 | 0.6888 ± 0.0387 |
| Naive Bayes | 0.7922 ± 0.0289 | 0.9155 ± 0.0321 | 0.6117 ± 0.0547 |
| KNN (k=5) | 0.7870 ± 0.0519 | 0.9260 ± 0.0297 | 0.6246 ± 0.0821 |

### Key Biomarker Finding

**E-Cadherin (`pp_E.Cadherin`)** is consistently the #1 most important feature across all SHAP analyses and all random splits (ranked #1 in 5 out of 5 splits), confirming its role as the dominant biomarker for IDC vs ILC classification — a biologically validated result (loss of E-Cadherin is the defining hallmark of lobular carcinoma).

---

## Scientific Contributions

| # | Contribution | Evidence |
|:--|:---|:---|
| C1 | Multi-omics integration (mRNA + CNV + Methylation + Protein) on TCGA BRCA for IDC/ILC classification | 705 patients × 4 omics layers |
| C2 | Three-stage consensus feature selection funnel (Variance → ANOVA+MI union → RF+XGB consensus) | Reduces 1,837 → 75 features without single-method bias |
| C3 | SMOTE-inside-CV using `imblearn.Pipeline` for leak-free evaluation | Nested CV confirms leak-free performance ($F1 = 0.884$, $\Delta F1 < 0.03$) |
| C4 | Cross-Omics SHAP Attribution — omics-layer-level explainability | Protein features drive 40.26% of classification signal |
| C5 | SHAP ranking stability analysis (Jaccard + Spearman across 5 splits) | E-Cadherin = #1 feature in 5/5 splits |

---

## Methodology

### Pipeline Architecture

```
TCGA BRCA Dataset (705 patients × 1,837 features × 4 omics layers)
    │
    ├── Phase 1: Data Pipeline + 3-Stage Consensus Feature Selection
    │       ├── Stage 1: Variance Threshold (1,837 → 1,837)
    │       ├── Stage 2: ANOVA + MI per-omics union (1,837 → 472)
    │       └── Stage 3: RF + XGB consensus ranking (472 → 75)
    │
    ├── Phase 2: Baseline Models (5 classifiers × SMOTE-inside-CV)
    │       ├── Logistic Regression, SVM (RBF), KNN (k=5)
    │       └── Naive Bayes, Random Forest
    │
    ├── Phase 3: Advanced Models + Stacking Ensemble
    │       ├── XGBoost tuning (RandomizedSearchCV, 50 iter)
    │       ├── LightGBM tuning (RandomizedSearchCV, 50 iter)
    │       └── Stacking: RF + XGB + LightGBM → LR meta-learner
    │
    ├── Phase 4: SHAP Explainability (Core Novelty)
    │       ├── Global SHAP beeswarm (top 20 features)
    │       ├── Cross-Omics SHAP Attribution (% per layer per class)
    │       ├── Per-class SHAP summary (IDC vs ILC)
    │       └── Patient-level waterfall explanations
    │
    └── Phase 5: Fusion Comparison
            ├── Early Fusion (concatenated Stacking: F1 = 0.896, AUC = 0.966)
            └── Late Fusion (per-omics XGBoost → soft vote: F1 = 0.925, AUC = 0.984)
```

### Omics Layers

| Omics Layer | Prefix | Features Selected | Description |
|:---|:---:|:---:|:---|
| mRNA Expression | `rs_` | 47 | Gene expression levels from RNA-seq |
| Protein (RPPA) | `pp_` | 24 | Reverse Phase Protein Array measurements |
| Copy Number Variation | `cn_` | 2 | Chromosomal copy number alterations |
| DNA Methylation | `mu_` | 2 | CpG site methylation levels |

---

## Repository Structure

```
Explainable-Multi-Omics-Breast-Cancer-Classification/
│
├── 01_run_core_pipeline.py         # Master script — runs the full 5-phase pipeline
├── 02_run_supplementary_analysis.py # All supplementary analyses for thesis defense
│
├── src/                            # Core pipeline modules
│   ├── __init__.py                 # Package metadata and module listing
│   ├── config.py                   # Global configuration and hyperparameters
│   ├── utils.py                    # Seed locking and console output utilities
│   ├── data_pipeline.py            # TCGA BRCA data loading and preprocessing
│   ├── feature_selection.py        # 3-stage consensus feature selection funnel
│   ├── baseline_models.py          # 5 baseline classifiers with SMOTE-inside-CV
│   ├── advanced_models.py          # XGBoost/LightGBM tuning + Stacking Ensemble
│   ├── shap_analysis.py            # Cross-Omics SHAP Attribution (core novelty)
│   ├── shap_stability.py           # SHAP ranking stability analysis
│   ├── nested_cv_validation.py     # Leak-free nested cross-validation
│   ├── fusion_comparison.py        # Early vs Late fusion comparison
│   └── visualization.py            # Publication-quality figure generation (300 DPI)
│
├── data/
│   └── brca_data_w_subtypes.csv    # TCGA BRCA multi-omics dataset (705 × 1,936)
│
├── outputs/
│   ├── figures/                    # All generated 300 DPI figures (fig_01 through fig_26)
│   ├── results/                    # CSV result tables and statistical tests
│   ├── models/                     # Saved model objects (.joblib)
│   └── preprocessed/               # Intermediate data files
│
├── docs/
│   └── Thesis_Complete_Roadmap_Rudra.pdf  # Thesis planning document
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Installation & Execution

### Prerequisites

- Python 3.9+
- 8 GB RAM recommended (for XGBoost/LightGBM tuning)

### Setup

```bash
# Clone the repository
git clone https://github.com/peashdasrudra/Explainable-Multi-Omics-Breast-Cancer-Classification.git
cd Explainable-Multi-Omics-Breast-Cancer-Classification

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Execution

The pipeline is split into two scripts that should be run sequentially:

```bash
# Step 1: Run the core pipeline (Phases 1-5)
# Generates: 14 figures (300 DPI), model comparison tables, SHAP attribution
# Expected runtime: ~3-5 minutes
python 01_run_core_pipeline.py

# Step 2: Run supplementary analyses (Sections A-D)
# Generates: 14 additional figures, statistical tests, ablation study, nested CV
# Expected runtime: ~3-5 minutes
python 02_run_supplementary_analysis.py
```

---

## Generated Outputs

### Figures (28 total, 300 DPI)

#### Core Pipeline (`01_run_core_pipeline.py`)

| Figure | Filename | Description |
|:---:|:---|:---|
| 1 | `fig_01_funnel.png` | 3-Stage Feature Selection Funnel |
| 2 | `fig_02_label_dist.png` | Class Distribution (IDC vs ILC) |
| 3 | `fig_03_consensus_features.png` | Top 20 Consensus Features |
| 4 | `fig_04_roc_all.png` | ROC Curves — All Models |
| 5 | `fig_05_model_comparison.png` | Model Comparison (F1-Macro) |
| 6 | `fig_06_confusion_best.png` | Confusion Matrix — Best Model |
| 7 | `fig_07_shap_beeswarm.png` | Global SHAP Beeswarm |
| 8 | `fig_08_omics_attribution.png` | **Cross-Omics SHAP Attribution** |
| 9 | `fig_09_shap_IDC.png` | Per-Class SHAP — IDC |
| 9b | `fig_09_shap_ILC.png` | Per-Class SHAP — ILC |
| 10 | `fig_10_waterfall_p1.png` | Patient Waterfall — IDC |
| 10b | `fig_10_waterfall_p2.png` | Patient Waterfall — ILC |
| 11 | `fig_11_fusion_comparison.png` | Early vs Late Fusion |
| 12 | `fig_12_confusion_late.png` | Confusion Matrix — Late Fusion |

#### Supplementary Analyses (`02_run_supplementary_analysis.py`)

| Figure | Filename | Description |
|:---:|:---|:---|
| 13 | `fig_13_significance_heatmap.png` | Wilcoxon p-value Heatmap (5-fold) |
| 14 | `fig_14_cv_stability.png` | Cross-Validation Stability Box Plot |
| 15 | `fig_15_ablation_study.png` | Omics Ablation Study |
| 16 | `fig_16_correlation_heatmap.png` | Feature Correlation Matrix |
| 17 | `fig_17_shap_dependence_ecadherin.png` | SHAP Dependence — E-Cadherin |
| 18 | `fig_18_learning_curve.png` | Learning Curve (LightGBM) |
| 19 | `fig_19_feature_composition.png` | Feature Composition Pie Chart |
| 20 | `fig_20_precision_recall.png` | Precision-Recall Curves |
| 21 | `fig_21_nested_cv_comparison.png` | Nested CV vs Original |
| 22 | `fig_22_feature_stability_nested.png` | Feature Selection Stability |
| 23 | `fig_23_shap_stability.png` | SHAP Ranking Stability |
| 24 | `fig_24_fusion_cv_comparison.png` | Fusion CV Comparison |
| 25 | `fig_25_significance_30fold.png` | Wilcoxon Heatmap (30-fold) |
| 26 | `fig_26_pathway_analysis.png` | Pathway Enrichment Analysis |

### Result Tables (18 CSVs)

| File | Content |
|:---|:---|
| `results_baseline.csv` | 5 baseline model metrics |
| `results_all_models.csv` | Full 8-model comparison |
| `results_all_models_numeric.csv` | Numeric version for plotting |
| `consensus_importances.csv` | RF+XGB consensus feature importances |
| `omics_attribution.csv` | Cross-omics SHAP attribution (%) |
| `fusion_comparison.csv` | Early vs Late fusion results |
| `per_fold_f1_scores.csv` | Per-fold F1 scores (5-fold CV) |
| `per_fold_f1_scores_30fold.csv` | Per-fold F1 scores (30-fold CV) |
| `statistical_significance.csv` | Pairwise Wilcoxon p-values (5-fold) |
| `statistical_significance_30fold.csv` | Pairwise Wilcoxon p-values (30-fold) |
| `ablation_study.csv` | Omics ablation results |
| `nested_cv_results.csv` | Nested CV per-fold results |
| `late_fusion_cv_results.csv` | Late fusion CV per-fold results |
| `shap_stability_results.csv` | SHAP stability per-split rankings |
| `shap_stability_summary.csv` | SHAP stability summary metrics |
| `gene_symbols_75features.csv` | Gene symbol mapping |
| `pathway_enrichment.csv` | GO/KEGG enrichment results |

---

## Dependencies

| Package | Version | Purpose |
|:---|:---:|:---|
| numpy | ≥1.24 | Numerical computation |
| pandas | ≥2.0 | Data manipulation |
| scikit-learn | ≥1.3 | ML models and evaluation |
| imbalanced-learn | ≥0.11 | SMOTE-inside-CV pipeline |
| xgboost | ≥2.0 | Gradient boosted trees |
| lightgbm | ≥4.0 | Gradient boosted trees |
| shap | ≥0.42 | SHAP explainability |
| matplotlib | ≥3.7 | Figure generation |
| seaborn | ≥0.12 | Statistical visualization |
| scipy | ≥1.11 | Statistical tests |
| joblib | ≥1.3 | Model serialization |
| gseapy | ≥1.0 | Pathway enrichment (optional) |

---

## Reproducibility

All experiments use `RANDOM_STATE = 42` with seeds locked for Python, NumPy, and all model random states. Running the pipeline on the same data will produce identical results.

---

## References

1. TCGA Network (2012). Comprehensive molecular portraits of human breast tumours. *Nature*, 490(7418), 61-70.
2. Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR*, 16, 321-357.
3. Lundberg, S. M. & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
4. Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
5. Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*.
6. Bergstra, J. & Bengio, Y. (2012). Random search for hyper-parameter optimization. *JMLR*, 13, 281-305.
7. Ciriello, G. et al. (2015). Comprehensive molecular portraits of invasive lobular breast cancer. *Cell*, 163(2), 506-519.
8. Nadeau, C. & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52, 239-281.

---

## License

This project is for academic research purposes. Please cite this work if you use it in your research.
