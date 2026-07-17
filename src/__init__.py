"""
src/ -- Core Pipeline Modules.

Explainable Multi-Omics Breast Cancer Subtype Classification Using
Consensus Feature Selection, Ensemble Learning & Cross-Omics SHAP Attribution
on TCGA BRCA.

Modules:
    config              - Global constants, paths, hyperparameters
    utils               - Seed locking, formatted printing utilities
    data_pipeline       - Data loading, deduplication, cleaning
    feature_selection   - 3-stage consensus feature selection funnel
    baseline_models     - 5 baseline classifiers (SMOTE-inside-CV)
    advanced_models     - XGBoost/LightGBM tuning + Stacking Ensemble
    shap_analysis       - Cross-omics SHAP attribution analysis
    shap_stability      - SHAP ranking stability across random splits
    fusion_comparison   - Early vs Late multi-omics fusion
    nested_cv_validation- Leak-free nested cross-validation
    visualization       - Publication-quality figure generation
"""
