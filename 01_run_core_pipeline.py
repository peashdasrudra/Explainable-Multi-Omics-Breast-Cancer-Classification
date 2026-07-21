#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_run_core_pipeline.py
========================
Master Execution Script — Core Thesis Pipeline.

Thesis:
    Explainable Multi-Omics Breast Cancer Classification Using
    Consensus Feature Selection, Ensemble Learning, and
    Cross-Omics SHAP Attribution on TCGA BRCA

Pipeline Phases:
    Phase 1: Data Pipeline + 3-Stage Consensus Feature Selection
        - Load TCGA BRCA multi-omics dataset (705 patients × 1,837 features)
        - Content-based column deduplication
        - 3-stage consensus feature selection funnel (1,837 → 75)

    Phase 2: Baseline Classification (SMOTE-inside-CV)
        - 5 baseline classifiers: LR, SVM, KNN, Naive Bayes, Random Forest
        - Evaluated with Stratified 5-Fold CV using imblearn.Pipeline
          to guarantee SMOTE is applied exclusively within training folds

    Phase 3: Advanced Models + Stacking Ensemble
        - XGBoost and LightGBM hyperparameter tuning (RandomizedSearchCV)
        - Stacking Ensemble: RF + XGBoost + LightGBM → LR meta-learner
        - Full 8-model comparison table

    Phase 4: SHAP Explainability Analysis (Core Novelty)
        - Global SHAP beeswarm (top 20 features)
        - Cross-omics SHAP attribution (% contribution per omics layer)
        - Per-class SHAP summary (IDC vs ILC driving features)
        - Patient-level waterfall explanations

    Phase 5: Fusion Comparison + Final Compilation
        - Early Fusion (concatenated Stacking) vs Late Fusion (per-omics soft vote)
        - Final results compilation

Generated Outputs:
    Figures (14):  fig_01 through fig_12 saved to outputs/figures/
    Result Tables: Saved to outputs/results/
    Saved Models:  Saved to outputs/models/

Usage:
    python 01_run_core_pipeline.py

References:
    [1] TCGA Network (2012). Comprehensive molecular portraits of human
        breast tumours. Nature, 490(7418), 61-70.
    [2] Chawla et al. (2002). SMOTE: Synthetic minority over-sampling
        technique. JAIR, 16, 321-357.
    [3] Lundberg & Lee (2017). A unified approach to interpreting model
        predictions. NeurIPS.
    [4] Chen & Guestrin (2016). XGBoost: A scalable tree boosting system. KDD.
    [5] Ke et al. (2017). LightGBM: A highly efficient gradient boosting
        decision tree. NeurIPS.
"""

import time
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows to prevent encoding errors in console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to Python path for src.* imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import set_all_seeds, print_section
from src.data_pipeline import run_data_pipeline
from src.feature_selection import run_feature_selection
from src.baseline_models import (
    run_baselines, results_to_dataframe, results_to_numeric_dataframe,
    save_baseline_results
)
from src.advanced_models import run_advanced_models
from src.shap_analysis import run_shap_analysis
from src.fusion_comparison import run_fusion_comparison
from src.visualization import (
    plot_feature_funnel, plot_label_distribution,
    plot_consensus_features, plot_roc_curves,
    plot_model_comparison, plot_confusion_matrix,
    plot_confusion_late_fusion, plot_fusion_comparison
)


def main():
    """
    Execute the complete thesis pipeline end-to-end.

    This is the primary entry point that reproduces all core results
    reported in the thesis. All random seeds are locked to 42 for
    full reproducibility.
    """
    total_start = time.time()

    print("\n" + "=" * 70)
    print("  EXPLAINABLE MULTI-OMICS BREAST CANCER CLASSIFICATION")
    print("  Consensus Feature Selection × Ensemble Learning × Cross-Omics SHAP")
    print("  TCGA BRCA — 705 Patients × 1,837 Features × 4 Omics Layers")
    print("=" * 70)

    # Lock all random seeds for reproducibility
    set_all_seeds(42)

    # =================================================================
    # PHASE 1: Data Pipeline + 3-Stage Consensus Feature Selection
    # =================================================================
    # Loads the TCGA BRCA multi-omics CSV (mRNA + CNV + Methylation + Protein),
    # removes content-duplicate columns, encodes IDC/ILC labels, and runs
    # the 3-stage consensus feature selection funnel:
    #   Stage 1: Variance Threshold (remove near-constant features)
    #   Stage 2: ANOVA F-test + Mutual Information (per-omics union)
    #   Stage 3: RF + XGBoost consensus importance (top-75)
    print_section("PHASE 1 — Data Pipeline + Feature Selection")
    phase1_start = time.time()

    # Run data pipeline: load, deduplicate, encode
    X, y, label_encoder, omics_groups, class_dist = run_data_pipeline()

    # Generate label distribution figure (fig_02)
    plot_label_distribution(y, label_encoder)

    # Run 3-stage consensus feature selection (1,837 → 75 features)
    X_final, final_features, funnel, importance_df = run_feature_selection(X, y, omics_groups)

    # Generate feature funnel figure (fig_01)
    plot_feature_funnel(funnel)

    # Generate consensus features figure (fig_03)
    plot_consensus_features(importance_df)

    phase1_time = time.time() - phase1_start
    print(f"\n  [TIME] Phase 1 complete in {phase1_time:.1f}s")
    print(f"  [OK] X_final shape: {X_final.shape}")

    # =================================================================
    # PHASE 2: Baseline Models with SMOTE-inside-CV
    # =================================================================
    # CRITICAL: Uses imblearn.Pipeline (NOT sklearn.pipeline.Pipeline)
    # to ensure SMOTE is applied ONLY inside training folds, preventing
    # synthetic sample leakage into test folds. This corrects a widespread
    # methodological flaw in published multi-omics classification papers.
    print_section("PHASE 2 — Baseline Models (SMOTE-inside-CV)")
    phase2_start = time.time()

    baseline_results = run_baselines(X_final.values, y)

    # Save baseline results to CSV
    baseline_df = results_to_dataframe(baseline_results)
    baseline_numeric = results_to_numeric_dataframe(baseline_results)
    save_baseline_results(baseline_df, baseline_numeric)

    phase2_time = time.time() - phase2_start
    print(f"\n  [TIME] Phase 2 complete in {phase2_time:.1f}s")

    # =================================================================
    # PHASE 3: XGBoost + LightGBM Tuning + Stacking Ensemble
    # =================================================================
    # Hyperparameter optimization via RandomizedSearchCV (50 iterations)
    # with SMOTE-inside-CV pipeline, followed by a Stacking Ensemble
    # combining RF + tuned XGBoost + tuned LightGBM with a Logistic
    # Regression meta-learner.
    print_section("PHASE 3 — Advanced Models + Stacking Ensemble")
    phase3_start = time.time()

    all_results, best_model_name, xgb_params, lgbm_params = \
        run_advanced_models(X_final.values, y, baseline_results)

    # Generate model comparison figure (fig_05)
    plot_model_comparison(all_results)

    # Generate ROC curves for all models (fig_04)
    plot_roc_curves(X_final.values, y, all_results, label_encoder)

    # Generate confusion matrix for the best model (fig_06)
    plot_confusion_matrix(X_final.values, y, best_model_name, label_encoder)

    phase3_time = time.time() - phase3_start
    print(f"\n  [TIME] Phase 3 complete in {phase3_time:.1f}s")

    # =================================================================
    # PHASE 4: SHAP Analysis — The Core Novelty
    # =================================================================
    # Cross-omics SHAP attribution analysis: for the first time on
    # TCGA BRCA IDC/ILC classification, quantifies how much each
    # omics layer (mRNA, Protein, Methylation, CNV) contributes to
    # the AI's classification decision. Uses TreeExplainer for
    # exact SHAP value computation on the tuned XGBoost model.
    print_section("PHASE 4 — SHAP Analysis (Core Novelty)")
    phase4_start = time.time()

    attr_df, shap_model, shap_scaler = run_shap_analysis(
        X_final, y, label_encoder, xgb_params
    )

    phase4_time = time.time() - phase4_start
    print(f"\n  [TIME] Phase 4 complete in {phase4_time:.1f}s")

    # =================================================================
    # PHASE 5: Fusion Comparison + Final Compilation
    # =================================================================
    # Compares Early Fusion (concatenate all omics → Stacking) with
    # Late Fusion (train per-omics XGBoost → soft-vote). Late fusion
    # probes whether combining specialized per-omics models outperforms
    # the monolithic early-fusion approach.
    print_section("PHASE 5 — Fusion Comparison + Final Compilation")
    phase5_start = time.time()

    fusion_df, late_metrics, late_pred, y_test, late_cm = \
        run_fusion_comparison(X_final, y, all_results, label_encoder)

    # Generate fusion comparison figure (fig_11)
    early_results = all_results.get("Stacking Ensemble", None)
    plot_fusion_comparison(fusion_df, early_results, late_metrics)

    # Generate late fusion confusion matrix (fig_12)
    plot_confusion_late_fusion(y_test, late_pred, label_encoder)

    phase5_time = time.time() - phase5_start
    print(f"\n  [TIME] Phase 5 complete in {phase5_time:.1f}s")

    # =================================================================
    # FINAL SUMMARY
    # =================================================================
    total_time = time.time() - total_start
    print_section("PIPELINE COMPLETE — FINAL SUMMARY")

    print(f"  Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    print(f"\n  Feature Selection Funnel:")
    for stage, count in funnel.items():
        print(f"    {stage}: {count}")

    print(f"\n  Best Model: {best_model_name}")
    best_f1 = all_results[best_model_name]["f1_macro"]
    best_auc = all_results[best_model_name]["roc_auc"]
    print(f"    F1-Macro: {best_f1[0]:.4f} ± {best_f1[1]:.4f}")
    print(f"    AUC-ROC:  {best_auc[0]:.4f} ± {best_auc[1]:.4f}")

    print(f"\n  All 8 Models:")
    for model_name, metrics in all_results.items():
        f1 = metrics["f1_macro"]
        marker = " ◄ BEST" if model_name == best_model_name else ""
        print(f"    {model_name}: F1={f1[0]:.4f} ± {f1[1]:.4f}{marker}")

    print(f"\n  Cross-Omics SHAP Attribution:")
    print(f"  {attr_df.to_string()}")

    print(f"\n  Figures generated in: outputs/figures/")
    print(f"  Results tables in:    outputs/results/")
    print(f"  Saved models in:      outputs/models/")

    # List all generated figures
    from src.config import FIGURES_DIR
    figures = sorted([f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")])
    print(f"\n  Generated Figures ({len(figures)}):")
    for fig in figures:
        print(f"    [OK] {fig}")

    print("\n" + "=" * 70)
    print("  CORE PIPELINE COMPLETE — Run 02_run_supplementary_analysis.py next")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
