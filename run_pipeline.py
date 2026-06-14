"""
run_pipeline.py -- Master Execution Script.

Runs the complete thesis pipeline end-to-end:
  Day 1: Data Pipeline + 3-Stage Feature Selection
  Day 2: 5 Baseline Models with SMOTE-inside-CV
  Day 3: XGBoost/LightGBM Tuning + Stacking Ensemble
  Day 4: SHAP Analysis (Core Novelty)
  Day 5: Fusion Comparison + Final Compilation

Explainable Multi-Omics Breast Cancer Classification Using
Consensus Feature Selection, Ensemble Learning & Cross-Omics SHAP Attribution
on TCGA BRCA.

Usage:
    python run_pipeline.py
"""
import time
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
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
    """Execute the complete thesis pipeline."""
    total_start = time.time()

    print("\n" + "=" * 70)
    print("  EXPLAINABLE MULTI-OMICS BREAST CANCER CLASSIFICATION")
    print("  Consensus Feature Selection x Ensemble Learning x Cross-Omics SHAP")
    print("  TCGA BRCA -- 705 Patients x 1,837 Features x 4 Omics Layers")
    print("=" * 70)

    # --- Lock seeds ---
    set_all_seeds(42)

    # =================================================================
    # DAY 1: Data Pipeline + Feature Selection (Stage 1-2)
    # =================================================================
    print_section("DAY 1 -- Data Pipeline + Feature Selection")
    day1_start = time.time()

    # Run data pipeline
    X, y, label_encoder, omics_groups, class_dist = run_data_pipeline()

    # Generate label distribution figure (fig_02)
    plot_label_distribution(y, label_encoder)

    # Run 3-stage feature selection
    X_final, final_features, funnel, importance_df = run_feature_selection(X, y, omics_groups)

    # Generate feature funnel figure (fig_01)
    plot_feature_funnel(funnel)

    # Generate consensus features figure (fig_03)
    plot_consensus_features(importance_df)

    day1_time = time.time() - day1_start
    print(f"\n  [TIME] Day 1 complete in {day1_time:.1f}s")
    print(f"  [OK] X_final shape: {X_final.shape}")

    # =================================================================
    # DAY 2: Baseline Models with SMOTE-inside-CV
    # =================================================================
    print_section("DAY 2 -- Baseline Models (SMOTE-inside-CV)")
    day2_start = time.time()

    baseline_results = run_baselines(X_final.values, y)

    # Save baseline results
    baseline_df = results_to_dataframe(baseline_results)
    baseline_numeric = results_to_numeric_dataframe(baseline_results)
    save_baseline_results(baseline_df, baseline_numeric)

    day2_time = time.time() - day2_start
    print(f"\n  [TIME] Day 2 complete in {day2_time:.1f}s")

    # =================================================================
    # DAY 3: XGBoost + LightGBM Tuning + Stacking Ensemble
    # =================================================================
    print_section("DAY 3 -- Advanced Models + Stacking Ensemble")
    day3_start = time.time()

    all_results, best_model_name, xgb_params, lgbm_params = \
        run_advanced_models(X_final.values, y, baseline_results)

    # Generate model comparison figure (fig_05)
    plot_model_comparison(all_results)

    # Generate ROC curves (fig_04)
    plot_roc_curves(X_final.values, y, all_results, label_encoder)

    # Generate confusion matrix for best model (fig_06)
    plot_confusion_matrix(X_final.values, y, best_model_name, label_encoder)

    day3_time = time.time() - day3_start
    print(f"\n  [TIME] Day 3 complete in {day3_time:.1f}s")

    # =================================================================
    # DAY 4: SHAP Analysis -- The Core Novelty
    # =================================================================
    print_section("DAY 4 -- SHAP Analysis (Core Novelty)")
    day4_start = time.time()

    attr_df, shap_model, shap_scaler = run_shap_analysis(
        X_final, y, label_encoder, xgb_params
    )

    day4_time = time.time() - day4_start
    print(f"\n  [TIME] Day 4 complete in {day4_time:.1f}s")

    # =================================================================
    # DAY 5: Fusion Comparison + Final Compilation
    # =================================================================
    print_section("DAY 5 -- Fusion Comparison + Final Compilation")
    day5_start = time.time()

    fusion_df, late_metrics, late_pred, y_test, late_cm = \
        run_fusion_comparison(X_final, y, all_results, label_encoder)

    # Generate fusion comparison figure (fig_11)
    early_results = all_results.get("Stacking Ensemble", None)
    plot_fusion_comparison(fusion_df, early_results, late_metrics)

    # Generate late fusion confusion matrix (fig_12)
    plot_confusion_late_fusion(y_test, late_pred, label_encoder)

    day5_time = time.time() - day5_start
    print(f"\n  [TIME] Day 5 complete in {day5_time:.1f}s")

    # =================================================================
    # FINAL SUMMARY
    # =================================================================
    total_time = time.time() - total_start
    print_section("PIPELINE COMPLETE -- FINAL SUMMARY")

    print(f"  Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"\n  Feature Selection Funnel:")
    for stage, count in funnel.items():
        print(f"    {stage}: {count}")

    print(f"\n  Best Model: {best_model_name}")
    best_f1 = all_results[best_model_name]["f1_macro"]
    best_auc = all_results[best_model_name]["roc_auc"]
    print(f"    F1-Macro: {best_f1[0]:.4f} +/- {best_f1[1]:.4f}")
    print(f"    AUC-ROC:  {best_auc[0]:.4f} +/- {best_auc[1]:.4f}")

    print(f"\n  All 8 Models:")
    for model_name, metrics in all_results.items():
        f1 = metrics["f1_macro"]
        marker = " <-- BEST" if model_name == best_model_name else ""
        print(f"    {model_name}: F1={f1[0]:.4f}+/-{f1[1]:.4f}{marker}")

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
    print("  EXPERIMENTS COMPLETE -- Ready for thesis writing")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
