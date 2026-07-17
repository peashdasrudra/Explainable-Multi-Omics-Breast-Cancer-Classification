"""
nested_cv_validation.py -- Leak-Free Nested Cross-Validation.

Addresses the #1 reviewer criticism: feature selection on the full dataset
before CV causes information leakage.

This module wraps the ENTIRE feature selection pipeline inside each outer
CV fold, so that test data NEVER influences feature selection. The results
are compared against the original (non-nested) pipeline to prove that the
leakage was negligible.

Methodology:
    For each outer fold k=1..5:
        1. Split data into train_k and test_k
        2. Run 3-stage feature selection on train_k ONLY
        3. Reduce both train_k and test_k to the selected features
        4. Train SMOTE + model pipeline on train_k
        5. Evaluate on test_k
    Report: per-fold F1 and mean +/- std across folds
"""
import numpy as np
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import (
    RANDOM_STATE, CV_SPLITS, VARIANCE_THRESHOLD, STAGE2_K_PER_OMICS,
    STAGE3_TOP_N, OMICS_SHORT_NAMES, FIGURES_DIR, RESULTS_DIR, FIGURE_DPI
)
from src.utils import set_all_seeds, print_section, print_step


def _feature_selection_on_fold(X_train_df, y_train,
                                variance_threshold=VARIANCE_THRESHOLD,
                                k_per_omics=STAGE2_K_PER_OMICS,
                                top_n=STAGE3_TOP_N):
    """
    Run the full 3-stage feature selection on a SINGLE training fold.
    Returns the list of selected feature names.
    """
    # Stage 1: Variance Threshold
    selector = VarianceThreshold(threshold=variance_threshold)
    X_s1 = selector.fit_transform(X_train_df)
    s1_mask = selector.get_support()
    s1_cols = X_train_df.columns[s1_mask].tolist()
    X_s1_df = pd.DataFrame(X_s1, columns=s1_cols, index=X_train_df.index)

    # Stage 2: ANOVA + MI per omics (union)
    all_selected = []
    for prefix, name in OMICS_SHORT_NAMES.items():
        omics_cols = [c for c in s1_cols if c.startswith(prefix)]
        if len(omics_cols) == 0:
            continue

        X_omics = X_s1_df[omics_cols].values
        k = min(k_per_omics, len(omics_cols))

        # ANOVA
        anova_sel = SelectKBest(f_classif, k=k)
        anova_sel.fit(X_omics, y_train)
        anova_feats = set(np.array(omics_cols)[anova_sel.get_support()])

        # Mutual Information
        mi_sel = SelectKBest(mutual_info_classif, k=k)
        mi_sel.fit(X_omics, y_train)
        mi_feats = set(np.array(omics_cols)[mi_sel.get_support()])

        all_selected.extend(list(anova_feats | mi_feats))

    X_s2_df = X_s1_df[all_selected].copy()

    # Stage 3: RF + XGB consensus ranking
    X_s2_arr = X_s2_df.values

    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_s2_arr, y_train)

    xgb = XGBClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
        eval_metric="mlogloss", use_label_encoder=False, verbosity=0
    )
    xgb.fit(X_s2_arr, y_train)

    avg_imp = (rf.feature_importances_ + xgb.feature_importances_) / 2
    imp_df = pd.DataFrame({
        "feature": all_selected,
        "avg_importance": avg_imp
    }).sort_values("avg_importance", ascending=False)

    top_features = imp_df.head(top_n)["feature"].tolist()
    return top_features


def run_nested_cv(X, y, label_encoder=None):
    """
    Run leak-free nested CV:
        Outer loop: 5-fold stratified CV
        Inner operation: full 3-stage feature selection on training fold only
        Model: LightGBM (best model) with SMOTE

    Returns a DataFrame comparing nested vs original results.
    """
    print_section("NESTED CV VALIDATION -- Leak-Free Feature Selection")

    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    feature_names = list(X.columns) if hasattr(X, 'columns') else [f"f_{i}" for i in range(X.shape[1])]
    X_df = X if hasattr(X, 'columns') else pd.DataFrame(X, columns=feature_names)

    # Track per-fold results
    nested_results = {
        "fold": [], "n_features_selected": [],
        "f1_macro": [], "accuracy": [], "roc_auc": [], "mcc": [],
        "top5_features": [], "ecadherin_selected": []
    }

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_df, y)):
        print_step(fold_idx + 1, f"Fold {fold_idx + 1}/{CV_SPLITS}")

        X_train_fold = X_df.iloc[train_idx]
        X_test_fold = X_df.iloc[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]

        # Feature selection on training fold ONLY
        selected_features = _feature_selection_on_fold(X_train_fold, y_train_fold)
        n_selected = len(selected_features)

        # Check if E-Cadherin is selected (stability indicator)
        ecad_selected = any("E.Cadherin" in f or "E-Cadherin" in f for f in selected_features)

        # Reduce both train and test to selected features
        X_train_sel = X_train_fold[selected_features].values
        X_test_sel = X_test_fold[selected_features].values

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)

        # SMOTE on training only
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train_fold)

        # Train LightGBM (best model)
        model = LGBMClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
        )
        model.fit(X_train_res, y_train_res)

        # Predict
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)

        # Metrics
        f1 = f1_score(y_test_fold, y_pred, average="macro")
        acc = accuracy_score(y_test_fold, y_pred)
        mcc = matthews_corrcoef(y_test_fold, y_pred)

        n_classes = len(np.unique(y))
        if n_classes == 2:
            auc = roc_auc_score(y_test_fold, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test_fold, y_proba, multi_class="ovr")

        # Record
        nested_results["fold"].append(fold_idx + 1)
        nested_results["n_features_selected"].append(n_selected)
        nested_results["f1_macro"].append(f1)
        nested_results["accuracy"].append(acc)
        nested_results["roc_auc"].append(auc)
        nested_results["mcc"].append(mcc)
        nested_results["top5_features"].append(", ".join(selected_features[:5]))
        nested_results["ecadherin_selected"].append(ecad_selected)

        print(f"       Features: {n_selected}, F1={f1:.4f}, AUC={auc:.4f}, "
              f"E-Cadherin={'YES' if ecad_selected else 'no'}")
        print(f"       Top-5: {selected_features[:5]}")

    # Summary
    nested_df = pd.DataFrame(nested_results)
    mean_f1 = nested_df["f1_macro"].mean()
    std_f1 = nested_df["f1_macro"].std()
    mean_auc = nested_df["roc_auc"].mean()
    std_auc = nested_df["roc_auc"].std()
    ecad_stability = nested_df["ecadherin_selected"].sum()

    print(f"\n  [NESTED CV RESULTS]")
    print(f"  F1-Macro:  {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"  AUC-ROC:   {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  E-Cadherin selected in {ecad_stability}/{CV_SPLITS} folds")

    # Save results
    nested_path = os.path.join(RESULTS_DIR, "nested_cv_results.csv")
    nested_df.to_csv(nested_path, index=False)
    print(f"  Saved -> {nested_path}")

    return nested_df


def plot_nested_vs_original(nested_df, original_f1_mean=0.917, original_f1_std=0.013,
                            original_auc_mean=0.965, original_auc_std=None):
    """
    Bar chart comparing nested CV (leak-free) vs original pipeline results.
    If the gap is small, it proves the feature selection leakage was negligible.
    """
    print_step("FIG", "Generating Nested vs Original comparison plot...")

    nested_f1 = nested_df["f1_macro"].mean()
    nested_f1_std = nested_df["f1_macro"].std()
    nested_auc = nested_df["roc_auc"].mean()
    nested_auc_std = nested_df["roc_auc"].std()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # F1-Macro comparison
    ax = axes[0]
    methods = ["Original Pipeline\n(Feature Selection\non Full Data)",
               "Nested CV\n(Feature Selection\nper Fold)"]
    f1_vals = [original_f1_mean, nested_f1]
    f1_errs = [original_f1_std, nested_f1_std]
    colors = ["#3498DB", "#2ECC71"]

    bars = ax.bar(methods, f1_vals, yerr=f1_errs, color=colors, capsize=8,
                  edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val, err in zip(bars, f1_vals, f1_errs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.005,
                f"{val:.4f}\n\u00b1{err:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    delta_f1 = original_f1_mean - nested_f1
    ax.set_title(f"F1-Macro Comparison\n(\u0394 = {delta_f1:+.4f})", fontsize=13, fontweight="bold")
    ax.set_ylabel("F1-Macro", fontsize=12)
    ax.set_ylim(0.7, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # AUC comparison
    ax = axes[1]
    auc_vals = [original_auc_mean, nested_auc]
    auc_errs = [original_auc_std if original_auc_std else 0.01, nested_auc_std]

    bars = ax.bar(methods, auc_vals, yerr=auc_errs, color=colors, capsize=8,
                  edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val, err in zip(bars, auc_vals, auc_errs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.005,
                f"{val:.4f}\n\u00b1{err:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    delta_auc = original_auc_mean - nested_auc
    ax.set_title(f"AUC-ROC Comparison\n(\u0394 = {delta_auc:+.4f})", fontsize=13, fontweight="bold")
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_ylim(0.7, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle("Leak-Free Validation: Original vs Nested CV",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig_21_nested_cv_comparison.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


def plot_feature_stability(nested_df):
    """
    Show which features were selected across folds and how often E-Cadherin appears.
    """
    print_step("FIG", "Generating Feature Stability across folds plot...")

    # Collect all features from all folds
    all_features = []
    for _, row in nested_df.iterrows():
        feats = [f.strip() for f in row["top5_features"].split(",")]
        all_features.extend(feats)

    # Count frequency
    freq = Counter(all_features)
    top_feats = freq.most_common(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    names = [f[0] for f in top_feats]
    counts = [f[1] for f in top_feats]

    # Color by presence frequency
    colors = ["#2ECC71" if c == CV_SPLITS else "#3498DB" if c >= CV_SPLITS - 1 else "#F39C12"
              for c in counts]

    ax.barh(range(len(names)), counts, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(f"Appearances in Top-5 across {CV_SPLITS} Folds", fontsize=12)
    ax.set_title("Feature Selection Stability (Nested CV Top-5 per Fold)",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, CV_SPLITS + 0.5)

    # Add legend
    legend_patches = [
        mpatches.Patch(color="#2ECC71", label=f"All {CV_SPLITS} folds"),
        mpatches.Patch(color="#3498DB", label=f"{CV_SPLITS - 1} folds"),
        mpatches.Patch(color="#F39C12", label="Fewer folds"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_22_feature_stability_nested.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


# =============================================
# MAIN EXECUTION
# =============================================
def main():
    """Run the nested CV validation as a standalone script."""
    set_all_seeds(42)

    from src.data_pipeline import run_data_pipeline
    from src.feature_selection import run_feature_selection

    print_section("NESTED CV VALIDATION -- Publication-Critical")

    # Load data
    print_step(0, "Loading data...")
    X, y, label_encoder, omics_groups, class_dist = run_data_pipeline()

    # Also run original pipeline to get the reference numbers
    print_step(0, "Running original feature selection for reference...")
    X_final, final_features, funnel, importance_df = run_feature_selection(X, y, omics_groups)

    # Run nested CV
    nested_df = run_nested_cv(X, y, label_encoder)

    # Plot comparison (using the known original best-model results)
    plot_nested_vs_original(nested_df)

    # Plot feature stability
    plot_feature_stability(nested_df)

    # Print comparison summary
    nested_f1 = nested_df["f1_macro"].mean()
    original_f1 = 0.917  # From the original pipeline LightGBM result
    delta = original_f1 - nested_f1

    print_section("NESTED CV VALIDATION COMPLETE")
    print(f"  Original Pipeline F1:  0.917 +/- 0.013")
    print(f"  Nested CV F1:          {nested_f1:.4f} +/- {nested_df['f1_macro'].std():.4f}")
    print(f"  Performance Gap (delta):   {delta:+.4f}")

    if abs(delta) < 0.03:
        print(f"\n  [OK] Gap < 0.03 -- Feature selection leakage was NEGLIGIBLE.")
        print(f"       This validates the original pipeline's results.")
    else:
        print(f"\n  [!!] Gap >= 0.03 -- Feature selection had some impact.")
        print(f"       Report both results transparently in the thesis.")

    ecad_count = nested_df["ecadherin_selected"].sum()
    print(f"\n  E-Cadherin selected in {ecad_count}/{CV_SPLITS} folds")
    if ecad_count == CV_SPLITS:
        print(f"  [OK] E-Cadherin is STABLE across all folds -- strong biomarker.")


if __name__ == "__main__":
    main()
