"""
src/ — Core Pipeline Modules
==============================
Explainable Multi-Omics Breast Cancer Classification Using
Consensus Feature Selection, Ensemble Learning, and
Cross-Omics SHAP Attribution on TCGA BRCA.

This package implements the complete machine learning pipeline for
classifying breast cancer histological subtypes (IDC vs ILC) from
integrated multi-omics data. The pipeline spans data preprocessing,
feature selection, classification, and explainability analysis.

Modules
-------
config
    Global constants, file paths, hyperparameter grids, and visualization settings.
utils
    Reproducibility utilities (seed locking) and formatted console output.
data_pipeline
    TCGA BRCA data loading, content-based deduplication, missing value handling,
    and label encoding.
feature_selection
    Three-stage consensus feature selection funnel:
    Stage 1 — Variance Threshold, Stage 2 — ANOVA + MI per-omics union,
    Stage 3 — RF + XGBoost consensus importance ranking.
baseline_models
    Five baseline classifiers (LR, SVM, KNN, NB, RF) evaluated with
    Stratified K-Fold CV using imblearn.Pipeline for SMOTE-inside-CV.
advanced_models
    XGBoost and LightGBM hyperparameter tuning via RandomizedSearchCV,
    plus a Stacking Ensemble (RF + XGB + LightGBM → LR meta-learner).
shap_analysis
    Cross-omics SHAP attribution analysis — the core scientific novelty.
    Quantifies each omics layer's contribution to classification decisions.
shap_stability
    SHAP ranking stability analysis across multiple random train/test splits.
    Measures Jaccard similarity and Spearman rank correlation.
nested_cv_validation
    Leak-free nested cross-validation that wraps feature selection inside
    each outer CV fold to eliminate information leakage.
fusion_comparison
    Early Fusion (concatenated Stacking) vs Late Fusion (per-omics soft vote)
    comparison.
visualization
    Publication-quality figure generation for all pipeline outputs.
"""
