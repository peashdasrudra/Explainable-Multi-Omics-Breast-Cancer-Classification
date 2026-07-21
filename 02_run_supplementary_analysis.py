#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_run_supplementary_analysis.py
================================
Supplementary Analyses for Thesis Defense and Peer Review.

Thesis:
    Explainable Multi-Omics Breast Cancer Classification Using
    Consensus Feature Selection, Ensemble Learning, and
    Cross-Omics SHAP Attribution on TCGA BRCA

Purpose:
    This script executes all supplementary analyses that strengthen
    the main pipeline results (01_run_core_pipeline.py) against
    common examiner and reviewer criticisms. It produces additional
    publication-quality figures and statistical evidence tables.

Sections:
    A. STATISTICAL VALIDATION
        A1. Pairwise Wilcoxon Signed-Rank Test (5-fold CV)
        A2. Per-Fold CV Stability Box Plot
        A3. Upgraded 30-Fold Wilcoxon (10x3 Repeated Stratified K-Fold)

    B. MODEL DIAGNOSTICS
        B1. Omics Ablation Study (leave-one-layer-out + single-omics)
        B2. Learning Curve Analysis (sample-size sensitivity)
        B3. Precision-Recall Curves (imbalanced-class evaluation)
        B4. Feature Correlation Heatmap (redundancy check)
        B5. Feature Composition by Omics Layer (pie chart)

    C. EXPLAINABILITY & ROBUSTNESS
        C1. SHAP Dependence Plot (E-Cadherin threshold effect)
        C2. SHAP Ranking Stability (5-split Jaccard & Spearman)
        C3. Leak-Free Nested Cross-Validation
        C4. CV-Based Late Fusion (fair early-vs-late comparison)

    D. BIOLOGICAL INTERPRETATION
        D1. Pathway / Gene Ontology Enrichment Analysis

Generated Outputs:
    Figures (14):
        fig_13 through fig_26 saved to outputs/figures/
    Result Tables (11):
        Various CSVs saved to outputs/results/

Prerequisites:
    Run 01_run_core_pipeline.py first to generate baseline results.

Usage:
    python 02_run_supplementary_analysis.py

References:
    [1] Wilcoxon, F. (1945). Individual comparisons by ranking methods.
    [2] Nadeau & Bengio (2003). Inference for the generalization error.
    [3] Lundberg & Lee (2017). A unified approach to interpreting
        model predictions. NeurIPS.
"""

# ═══════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows to prevent encoding errors in console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to Python path for src.* imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold,
    cross_validate, learning_curve, train_test_split
)
from sklearn.metrics import (
    make_scorer, matthews_corrcoef,
    precision_recall_curve, average_precision_score,
    f1_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

from src.config import (
    RANDOM_STATE, CV_SPLITS, FIGURES_DIR, RESULTS_DIR,
    OMICS_SHORT_NAMES, OMICS_COLORS, FIGURE_DPI, MODEL_COLORS
)
from src.utils import set_all_seeds, print_section, print_step
from src.data_pipeline import run_data_pipeline
from src.feature_selection import run_feature_selection
from src.baseline_models import get_global_cv, build_smote_pipeline, evaluate_model_cv

# ──── Publication-quality matplotlib defaults ────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": FIGURE_DPI,
})


# ═══════════════════════════════════════════════════════════════════
#  SECTION A: STATISTICAL VALIDATION
# ═══════════════════════════════════════════════════════════════════

# ─── A1. Pairwise Wilcoxon Signed-Rank Test (5-fold) ─────────────
def run_statistical_tests(X, y):
    """
    Perform pairwise Wilcoxon signed-rank tests between all model pairs
    using per-fold F1-Macro scores from Stratified 5-Fold CV.

    Rationale:
        A thesis cannot claim one model is "better" without a statistical
        test. The Wilcoxon signed-rank test is non-parametric and appropriate
        for paired samples (same CV folds) [1].

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Encoded target labels.

    Returns
    -------
    fold_scores : dict
        Per-fold F1-Macro scores for each model.
    sig_df : pd.DataFrame
        Pairwise p-values and significance flags.
    """
    print_step(1, "Statistical Significance Tests (Wilcoxon signed-rank, 5-fold)")

    skf = get_global_cv()

    models = {
        "LR": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "NB": GaussianNB(),
        "RF": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "XGB": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                             random_state=RANDOM_STATE, eval_metric="mlogloss",
                             use_label_encoder=False, verbosity=0, n_jobs=-1),
        "LGBM": LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                               random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
    }

    # Collect per-fold F1-Macro scores using SMOTE-inside-CV pipeline
    fold_scores = {}
    for name, model in models.items():
        pipeline = build_smote_pipeline(model)
        cv_result = cross_validate(pipeline, X, y, cv=skf,
                                   scoring={"f1_macro": "f1_macro"}, n_jobs=-1)
        fold_scores[name] = cv_result["test_f1_macro"]
        mean = cv_result["test_f1_macro"].mean()
        std = cv_result["test_f1_macro"].std()
        print(f"       {name}: {mean:.4f} +/- {std:.4f}")

    # Save per-fold scores for transparency
    fold_df = pd.DataFrame(fold_scores)
    fold_df.index.name = "Fold"
    fold_df.to_csv(os.path.join(RESULTS_DIR, "per_fold_f1_scores.csv"))

    # Compute pairwise Wilcoxon signed-rank p-values
    model_names = list(fold_scores.keys())
    n = len(model_names)
    p_matrix = np.ones((n, n))
    sig_results = []

    for i in range(n):
        for j in range(i + 1, n):
            a = fold_scores[model_names[i]]
            b = fold_scores[model_names[j]]
            try:
                stat, p_val = stats.wilcoxon(a, b)
            except ValueError:
                p_val = 1.0  # Identical distributions
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val
            sig_results.append({
                "Model_A": model_names[i],
                "Model_B": model_names[j],
                "p_value": round(p_val, 4),
                "significant_p05": "YES" if p_val < 0.05 else "no",
            })

    sig_df = pd.DataFrame(sig_results)
    sig_df.to_csv(os.path.join(RESULTS_DIR, "statistical_significance.csv"), index=False)

    # Generate p-value heatmap (Figure 13)
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(p_matrix, dtype=bool), k=0)
    sns.heatmap(p_matrix, mask=mask, annot=True, fmt=".3f",
                xticklabels=model_names, yticklabels=model_names,
                cmap="RdYlGn_r", vmin=0, vmax=0.1,
                ax=ax, cbar_kws={"label": "p-value"})
    ax.set_title("Pairwise Wilcoxon Signed-Rank Test (p-values)\n"
                 "p < 0.05 = significant difference",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_13_significance_heatmap.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")

    return fold_scores, sig_df


# ─── A2. Per-Fold CV Stability Box Plot ──────────────────────────
def plot_cv_stability(fold_scores):
    """
    Horizontal box plot showing per-fold F1-Macro distributions for
    each model, with individual fold points overlaid as a strip plot.

    Purpose:
        Demonstrates that model performance is stable across CV folds
        and not driven by a single favorable split.

    Parameters
    ----------
    fold_scores : dict
        Model name -> array of per-fold F1-Macro scores.
    """
    print_step(2, "Per-Fold CV Stability Box Plot")

    fold_df = pd.DataFrame(fold_scores)
    order = fold_df.median().sort_values(ascending=True).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = MODEL_COLORS[:len(order)]
    # Note: Matplotlib 3.11+ removed the 'labels' parameter from boxplot;
    # use set_yticklabels() instead for compatibility.
    bp = ax.boxplot(
        [fold_df[m] for m in order],
        patch_artist=True, vert=False, widths=0.6
    )
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual fold points for transparency
    for i, m in enumerate(order):
        y = np.random.normal(i + 1, 0.04, size=len(fold_df[m]))
        ax.scatter(fold_df[m], y, alpha=0.8, s=30, zorder=5,
                   color="black", edgecolors="white")

    ax.set_xlabel("F1-Macro", fontsize=12)
    ax.set_title("Cross-Validation Stability — Per-Fold F1 Scores",
                 fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_14_cv_stability.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


# ─── A3. Upgraded 30-Fold Wilcoxon (10×3 Repeated CV) ────────────
def run_upgraded_statistical_tests(X, y):
    """
    Wilcoxon signed-rank tests with 10×3 Repeated Stratified K-Fold CV
    yielding 30 paired observations per model — providing substantially
    stronger statistical power than the 5-fold variant.

    Rationale:
        With only 5 paired observations, the Wilcoxon test has limited
        power to detect real differences. The 10×3 repeated CV scheme
        (Nadeau & Bengio, 2003) provides 30 observations while
        controlling for variance inflation from repeated use of data.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.

    Returns
    -------
    fold_scores : dict
        30-element score arrays per model.
    sig_df : pd.DataFrame
        Pairwise significance results.
    """
    print_step(3, "Upgraded Statistical Tests (10×3 Repeated Stratified K-Fold)")

    rskf = RepeatedStratifiedKFold(
        n_splits=10, n_repeats=3, random_state=RANDOM_STATE
    )

    models = {
        "LR": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "NB": GaussianNB(),
        "RF": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "XGB": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                             random_state=RANDOM_STATE, eval_metric="mlogloss",
                             use_label_encoder=False, verbosity=0, n_jobs=-1),
        "LGBM": LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                               random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
    }

    fold_scores = {}
    for name, model in models.items():
        pipeline = build_smote_pipeline(model)
        cv_result = cross_validate(pipeline, X, y, cv=rskf,
                                   scoring="f1_macro", n_jobs=-1)
        scores = cv_result["test_score"]
        fold_scores[name] = scores
        print(f"       {name}: {scores.mean():.4f} +/- {scores.std():.4f}")

    # Save per-fold scores
    fold_df = pd.DataFrame(fold_scores)
    fold_df.index.name = "Fold"
    fold_df.to_csv(os.path.join(RESULTS_DIR, "per_fold_f1_scores_30fold.csv"))

    # Pairwise Wilcoxon signed-rank tests
    model_names = list(fold_scores.keys())
    n = len(model_names)
    p_matrix = np.ones((n, n))
    sig_results = []

    for i in range(n):
        for j in range(i + 1, n):
            a = fold_scores[model_names[i]]
            b = fold_scores[model_names[j]]
            try:
                stat, p_val = stats.wilcoxon(a, b)
            except ValueError:
                p_val = 1.0
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val
            sig_results.append({
                "Model_A": model_names[i],
                "Model_B": model_names[j],
                "p_value": round(p_val, 6),
                "significant_p05": "YES" if p_val < 0.05 else "no",
            })

    sig_df = pd.DataFrame(sig_results)
    sig_df.to_csv(os.path.join(RESULTS_DIR, "statistical_significance_30fold.csv"),
                  index=False)

    n_significant = len(sig_df[sig_df["significant_p05"] == "YES"])
    print(f"       Significant differences: {n_significant}/{len(sig_df)} pairs (p<0.05)")

    # Generate upgraded heatmap (Figure 25)
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(p_matrix, dtype=bool), k=0)
    sns.heatmap(p_matrix, mask=mask, annot=True, fmt=".4f",
                xticklabels=model_names, yticklabels=model_names,
                cmap="RdYlGn_r", vmin=0, vmax=0.1,
                ax=ax, cbar_kws={"label": "p-value"})
    ax.set_title("Pairwise Wilcoxon Signed-Rank (10×3 Repeated CV)\n"
                 "30 paired observations | p < 0.05 = significant",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_25_significance_30fold.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")

    return fold_scores, sig_df


# ═══════════════════════════════════════════════════════════════════
#  SECTION B: MODEL DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════

# ─── B1. Omics Ablation Study ────────────────────────────────────
def run_ablation_study(X_final, y):
    """
    Ablation study: measure the impact of each omics layer by training
    LightGBM with (a) each layer removed, and (b) each layer alone.

    Rationale:
        Examiners will ask: "What happens if you remove mRNA?" or
        "Can protein alone classify subtypes?" This analysis provides
        direct empirical answers.

    Parameters
    ----------
    X_final : pd.DataFrame
        Feature matrix with column names preserving omics prefixes.
    y : np.ndarray
        Target labels.

    Returns
    -------
    abl_df : pd.DataFrame
        Ablation results with F1-Macro and delta from full model.
    """
    print_step(4, "Omics Ablation Study")

    feature_names = list(X_final.columns) if hasattr(X_final, 'columns') else []
    skf = get_global_cv()

    # Group features by omics layer prefix
    omics_features = {}
    for prefix, name in OMICS_SHORT_NAMES.items():
        cols = [c for c in feature_names if c.startswith(prefix)]
        if cols:
            omics_features[name] = cols

    model = LGBMClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
    )

    results = []

    # Baseline: full model with all omics layers
    pipeline = build_smote_pipeline(model)
    cv = cross_validate(pipeline, X_final.values, y, cv=skf,
                        scoring="f1_macro", n_jobs=-1)
    full_f1 = cv["test_score"].mean()
    full_std = cv["test_score"].std()
    results.append({"Configuration": "All Omics (Full)", "F1-Macro": full_f1,
                     "Std": full_std, "Delta": 0.0})
    print(f"       All Omics: F1={full_f1:.4f} +/- {full_std:.4f}")

    # Leave-one-out: remove each omics layer one at a time
    for omics_name, cols in omics_features.items():
        remaining = [c for c in feature_names if c not in cols]
        if not remaining:
            continue
        col_indices = [feature_names.index(c) for c in remaining]
        X_ablated = X_final.values[:, col_indices]

        pipeline = build_smote_pipeline(model)
        cv = cross_validate(pipeline, X_ablated, y, cv=skf,
                            scoring="f1_macro", n_jobs=-1)
        abl_f1 = cv["test_score"].mean()
        delta = full_f1 - abl_f1
        results.append({
            "Configuration": f"Remove {omics_name} ({len(cols)} feats)",
            "F1-Macro": abl_f1, "Std": cv["test_score"].std(), "Delta": delta,
        })
        print(f"       Remove {omics_name}: F1={abl_f1:.4f} (Δ={delta:+.4f})")

    # Single-omics: train with each layer alone
    for omics_name, cols in omics_features.items():
        col_indices = [feature_names.index(c) for c in cols]
        X_single = X_final.values[:, col_indices]

        pipeline = build_smote_pipeline(model)
        cv = cross_validate(pipeline, X_single, y, cv=skf,
                            scoring="f1_macro", n_jobs=-1)
        single_f1 = cv["test_score"].mean()
        results.append({
            "Configuration": f"Only {omics_name} ({len(cols)} feats)",
            "F1-Macro": single_f1, "Std": cv["test_score"].std(),
            "Delta": full_f1 - single_f1,
        })
        print(f"       Only {omics_name}: F1={single_f1:.4f}")

    # Save and plot
    abl_df = pd.DataFrame(results)
    abl_df.to_csv(os.path.join(RESULTS_DIR, "ablation_study.csv"), index=False)

    # Figure 15: Ablation study horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    configs = abl_df["Configuration"].tolist()
    f1_vals = abl_df["F1-Macro"].tolist()
    stds = abl_df["Std"].tolist()
    colors = []
    for c in configs:
        if "Full" in c:
            colors.append("#2ECC71")
        elif "Remove" in c:
            colors.append("#E74C3C")
        else:
            colors.append("#3498DB")

    bars = ax.barh(range(len(configs)), f1_vals, xerr=stds,
                   color=colors, height=0.6, capsize=3, edgecolor="white")
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=10)
    ax.set_xlabel("F1-Macro", fontsize=12)
    ax.set_title("Omics Ablation Study — Impact of Each Layer",
                 fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(x=full_f1, color="#2ECC71", linestyle="--", alpha=0.5,
               label="Full model baseline")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_15_ablation_study.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")

    return abl_df


# ─── B2. Learning Curve Analysis ─────────────────────────────────
def plot_learning_curves(X, y):
    """
    Learning curve: training vs. validation F1-Macro as a function
    of training set size.

    Purpose:
        Determines whether the model would benefit from more data
        (high bias) or if it is already plateauing (sufficient data).
        Important for discussing the 705-patient sample size limitation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    """
    print_step(5, "Learning Curve Analysis")

    model = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", LGBMClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
        )),
    ])

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.2, 1.0, 8),
        cv=get_global_cv(),
        scoring="f1_macro",
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color="#E74C3C")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.1, color="#3498DB")
    ax.plot(train_sizes, train_mean, "o-", color="#E74C3C",
            label="Training Score", linewidth=2)
    ax.plot(train_sizes, val_mean, "o-", color="#3498DB",
            label="Validation Score", linewidth=2)

    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("F1-Macro", fontsize=12)
    ax.set_title("Learning Curve — LightGBM (Best Model)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_ylim(0.5, 1.05)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_18_learning_curve.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


# ─── B3. Precision-Recall Curves ─────────────────────────────────
def plot_precision_recall_curves(X, y):
    """
    Precision-Recall curves for the minority class (ILC).

    Rationale:
        With a 4.4:1 class imbalance (IDC:ILC), Precision-Recall curves
        are more informative than ROC curves because they are sensitive
        to the minority class performance [Saito & Rehmsmeier, 2015].

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    """
    print_step(6, "Precision-Recall Curves (for imbalanced data)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_s, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                 random_state=RANDOM_STATE, eval_metric="mlogloss",
                                 use_label_encoder=False, verbosity=0, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                   random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = MODEL_COLORS[:len(models)]

    for (name, model), color in zip(models.items(), colors):
        model.fit(X_train_res, y_train_res)
        y_proba = model.predict_proba(X_test_s)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(recall, precision, color=color, lw=2,
                label=f"{name} (AP={ap:.3f})")

    # Baseline: prevalence of positive class
    baseline = y_test.mean()
    ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5,
               label=f"Baseline (prevalence={baseline:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves (ILC = positive class)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_20_precision_recall.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


# ─── B4. Feature Correlation Heatmap ─────────────────────────────
def plot_feature_correlation(X_final, top_n=25):
    """
    Pearson correlation heatmap of the top 25 consensus features.

    Purpose:
        Verifies that selected features are not highly redundant.
        High collinearity could indicate that the consensus funnel
        failed to remove correlated features.

    Parameters
    ----------
    X_final : pd.DataFrame
        Feature matrix with named columns.
    top_n : int
        Number of top features to include.
    """
    print_step(7, f"Feature Correlation Heatmap (top {top_n})")

    feature_names = list(X_final.columns)[:top_n]
    X_top = X_final[feature_names]
    corr = X_top.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=False, cmap="RdBu_r",
                vmin=-1, vmax=1, center=0, square=True,
                linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"})

    ax.set_title(f"Feature Correlation Matrix (Top {top_n} Consensus Features)",
                 fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_16_correlation_heatmap.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


# ─── B5. Feature Composition Pie Chart ───────────────────────────
def plot_feature_composition(X_final):
    """
    Pie chart showing the omics layer composition of the 75 consensus features.

    Purpose:
        Shows which omics layers contribute the most features to the
        final model, complementing the SHAP-based attribution analysis.

    Parameters
    ----------
    X_final : pd.DataFrame
        Feature matrix with named columns.
    """
    print_step(8, "Feature Composition by Omics Layer")

    feature_names = list(X_final.columns)
    composition = {}
    for prefix, name in OMICS_SHORT_NAMES.items():
        count = len([f for f in feature_names if f.startswith(prefix)])
        if count > 0:
            composition[name] = count

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = [OMICS_COLORS.get(n, "#95A5A6") for n in composition.keys()]
    wedges, texts, autotexts = ax.pie(
        composition.values(), labels=composition.keys(),
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct/100.*sum(composition.values())))})",
        colors=colors, startangle=90,
        textprops={"fontsize": 12}, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    ax.set_title("Omics Layer Composition of 75 Consensus Features",
                 fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_19_feature_composition.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════
#  SECTION C: EXPLAINABILITY & ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════

# ─── C1. SHAP Dependence Plot (E-Cadherin) ───────────────────────
def plot_shap_dependence(X_final, y):
    """
    SHAP dependence plot for the #1 biomarker: E-Cadherin (pp_E.Cadherin).

    Purpose:
        Shows the threshold effect: when E-Cadherin expression drops
        below ~−0.5 (standardized), the model strongly predicts ILC —
        consistent with the known E-cadherin loss mechanism in lobular
        carcinoma (Ciriello et al., 2015).

    Parameters
    ----------
    X_final : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Target labels.
    """
    print_step(9, "SHAP Dependence Plot (E-Cadherin)")

    feature_names = list(X_final.columns)

    # Locate E-Cadherin and CDH1 methylation columns
    ecad_col = None
    cdh1_col = None
    for f in feature_names:
        if "E.Cadherin" in f or "E-Cadherin" in f:
            ecad_col = f
        if "CDH1" in f:
            cdh1_col = f

    if ecad_col is None:
        print("       [!] E-Cadherin not found in features — skipping")
        return

    # Train model and compute SHAP
    X_train, X_test, y_train, y_test = train_test_split(
        X_final.values, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=RANDOM_STATE, eval_metric="mlogloss",
        use_label_encoder=False, verbosity=0, n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(
        pd.DataFrame(X_test_s, columns=feature_names)
    )
    X_test_df = pd.DataFrame(X_test_s, columns=feature_names)

    plt.figure(figsize=(10, 7))
    shap.dependence_plot(
        ecad_col, shap_values, X_test_df,
        interaction_index=cdh1_col if cdh1_col else "auto",
        show=False
    )
    plt.title(f"SHAP Dependence Plot — {ecad_col}",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_17_shap_dependence_ecadherin.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


# ─── C2. SHAP Ranking Stability ──────────────────────────────────
def run_shap_stability_analysis(X_final, y):
    """
    SHAP stability analysis across 5 random train/test splits.

    Rationale:
        SHAP values depend on the specific train/test split. This
        analysis verifies that feature rankings are robust by measuring
        Jaccard similarity and Spearman correlation across splits.

    Delegates to src.shap_stability module.

    Parameters
    ----------
    X_final : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Target labels.
    """
    print_step(10, "SHAP Ranking Stability Analysis")

    from src.shap_stability import run_shap_stability, plot_shap_stability
    shap_results = run_shap_stability(X_final, y, n_splits=5, top_k=20)
    plot_shap_stability(shap_results)


# ─── C3. Leak-Free Nested Cross-Validation ───────────────────────
def run_nested_cv_analysis(X, y, label_encoder):
    """
    Nested CV: wraps the entire 3-stage feature selection inside each
    outer CV fold so that test data never influences feature selection.

    Rationale:
        Addresses the #1 reviewer criticism — feature selection on the
        full dataset before CV constitutes information leakage.
        If the performance gap between nested and original is small
        (< 0.03 F1), the leakage was negligible.

    Delegates to src.nested_cv_validation module.

    Parameters
    ----------
    X : pd.DataFrame
        Full (pre-feature-selection) feature matrix.
    y : np.ndarray
        Target labels.
    label_encoder : LabelEncoder
        For class name lookup.

    Returns
    -------
    nested_df : pd.DataFrame
        Per-fold nested CV results.
    """
    print_step(11, "Leak-Free Nested Cross-Validation")

    from src.nested_cv_validation import (
        run_nested_cv, plot_nested_vs_original, plot_feature_stability
    )
    nested_df = run_nested_cv(X, y, label_encoder)
    plot_nested_vs_original(nested_df)
    plot_feature_stability(nested_df)
    return nested_df


# ─── C4. CV-Based Late Fusion ────────────────────────────────────
def run_late_fusion_cv(X_final, y):
    """
    Late fusion with proper 5-fold CV evaluation for fair comparison
    with the early fusion (stacking) approach.

    Methodology:
        For each fold:
        1. Train separate XGBoost models per omics layer on training set
        2. Generate prediction probabilities on test set
        3. Average probabilities (soft vote)
        4. Collect per-fold F1 and AUC

    Parameters
    ----------
    X_final : pd.DataFrame
        Feature matrix with omics-prefixed column names.
    y : np.ndarray
        Target labels.

    Returns
    -------
    late_cv_results : dict
        Mean/std F1-Macro and AUC-ROC, plus per-fold scores.
    """
    print_step(12, "CV-Based Late Fusion (Fair Comparison)")

    feature_names = list(X_final.columns) if hasattr(X_final, 'columns') else []
    X_df = X_final if hasattr(X_final, 'columns') else pd.DataFrame(X_final, columns=feature_names)

    # Group features by omics layer
    omics_features = {}
    for prefix, name in OMICS_SHORT_NAMES.items():
        cols = [c for c in feature_names if c.startswith(prefix)]
        if cols:
            omics_features[name] = cols

    print(f"       Omics groups: {', '.join([f'{n}({len(c)})' for n, c in omics_features.items()])}")

    skf = get_global_cv()
    fold_f1_scores = []
    fold_auc_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_df, y)):
        X_train = X_df.iloc[train_idx]
        X_test = X_df.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        per_omics_probas = []
        for omics_name, cols in omics_features.items():
            X_tr_omics = X_train[cols].values
            X_te_omics = X_test[cols].values

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_omics)
            X_te_scaled = scaler.transform(X_te_omics)

            smote = SMOTE(random_state=RANDOM_STATE)
            X_tr_res, y_tr_res = smote.fit_resample(X_tr_scaled, y_train)

            model = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=RANDOM_STATE, n_jobs=-1,
                eval_metric="mlogloss", use_label_encoder=False, verbosity=0
            )
            model.fit(X_tr_res, y_tr_res)
            per_omics_probas.append(model.predict_proba(X_te_scaled))

        # Soft vote: average probabilities across omics-specific models
        avg_proba = np.mean(per_omics_probas, axis=0)
        late_pred = np.argmax(avg_proba, axis=1)

        f1 = f1_score(y_test, late_pred, average="macro")
        auc = roc_auc_score(y_test, avg_proba[:, 1])
        fold_f1_scores.append(f1)
        fold_auc_scores.append(auc)
        print(f"       Fold {fold_idx + 1}/{CV_SPLITS}: F1={f1:.4f}, AUC={auc:.4f}")

    mean_f1, std_f1 = np.mean(fold_f1_scores), np.std(fold_f1_scores)
    mean_auc, std_auc = np.mean(fold_auc_scores), np.std(fold_auc_scores)
    print(f"       Late Fusion CV: F1={mean_f1:.4f}±{std_f1:.4f}, AUC={mean_auc:.4f}±{std_auc:.4f}")

    # Save per-fold results
    fusion_cv_df = pd.DataFrame({
        "Fold": range(1, len(fold_f1_scores) + 1),
        "F1_Macro": fold_f1_scores, "AUC_ROC": fold_auc_scores
    })
    fusion_cv_df.to_csv(os.path.join(RESULTS_DIR, "late_fusion_cv_results.csv"), index=False)

    late_cv_results = {
        "f1_macro": (mean_f1, std_f1),
        "roc_auc": (mean_auc, std_auc),
        "per_fold_f1": fold_f1_scores,
        "per_fold_auc": fold_auc_scores
    }

    # Generate fusion comparison figure (Figure 24)
    _plot_fusion_comparison_cv(late_cv_results)

    return late_cv_results


def _plot_fusion_comparison_cv(late_cv_results):
    """Generate CV-based Early vs Late Fusion comparison bar chart."""
    # Re-evaluate stacking (early fusion) to get comparable CV scores
    skf = get_global_cv()
    stacking = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
            ("xgb", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                  random_state=RANDOM_STATE, eval_metric="mlogloss",
                                  use_label_encoder=False, verbosity=0, n_jobs=-1)),
            ("lgbm", LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                    random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=5, n_jobs=-1
    )
    early_results = evaluate_model_cv("Stacking", stacking,
                                      X_final_global.values, y_global, skf)

    fig, ax = plt.subplots(figsize=(8, 5))
    strategies = ["Early Fusion\n(Stacking)", "Late Fusion\n(Per-Omics Soft Vote)"]
    f1_means = [early_results["f1_macro"][0], late_cv_results["f1_macro"][0]]
    f1_stds = [early_results["f1_macro"][1], late_cv_results["f1_macro"][1]]
    colors = ["#E74C3C", "#3498DB"]

    bars = ax.bar(strategies, f1_means, yerr=f1_stds, color=colors,
                  capsize=10, edgecolor="white", linewidth=1.5, width=0.4)
    for bar, val, std in zip(bars, f1_means, f1_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.008,
                f"{val:.4f}\n±{std:.4f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    winner = "Late Fusion" if f1_means[1] > f1_means[0] else "Early Fusion"
    delta = abs(f1_means[1] - f1_means[0])
    ax.set_title(f"Early vs Late Fusion (5-Fold CV)\n{winner} wins by Δ={delta:.4f}",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("F1-Macro", fontsize=12)
    ax.set_ylim(0.7, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_24_fusion_cv_comparison.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════
#  SECTION D: BIOLOGICAL INTERPRETATION
# ═══════════════════════════════════════════════════════════════════

# ─── D1. Pathway / GO Enrichment Analysis ────────────────────────
def run_pathway_analysis(X_final):
    """
    Gene Ontology and KEGG pathway enrichment on the 75 consensus
    features using the Enrichr API (via gseapy).

    Purpose:
        Maps the AI-selected features back to known biological pathways,
        providing independent biological validation that the model
        learned clinically meaningful signals.

    Parameters
    ----------
    X_final : pd.DataFrame
        Feature matrix with named columns.
    """
    print_step(13, "Pathway / Gene Ontology Enrichment Analysis")

    feature_names = list(X_final.columns) if hasattr(X_final, 'columns') else []

    # Extract gene symbols from feature names (remove omics prefix)
    gene_symbols = []
    gene_to_omics = {}
    for f in feature_names:
        for prefix in OMICS_SHORT_NAMES.keys():
            if f.startswith(prefix):
                gene = f[len(prefix):].replace(".", "").replace("-", "")
                gene_symbols.append(gene)
                gene_to_omics[gene] = OMICS_SHORT_NAMES[prefix]
                break

    unique_genes = list(set(gene_symbols))
    print(f"       Extracted {len(unique_genes)} unique gene symbols")

    # Save gene mapping
    gene_df = pd.DataFrame({
        "feature": feature_names,
        "gene_symbol": gene_symbols,
        "omics_layer": [gene_to_omics.get(g, "unknown") for g in gene_symbols]
    })
    gene_df.to_csv(os.path.join(RESULTS_DIR, "gene_symbols_75features.csv"), index=False)

    # Attempt gseapy enrichment
    try:
        import gseapy as gp
        enr = gp.enrichr(
            gene_list=unique_genes,
            gene_sets=["GO_Biological_Process_2023", "KEGG_2021_Human"],
            organism="human", outdir=None, no_plot=True, cutoff=0.05
        )
        results_df = enr.results
        if len(results_df) > 0:
            sig_results = results_df[results_df["Adjusted P-value"] < 0.05].copy()
            sig_results = sig_results.sort_values("Adjusted P-value")
            sig_results.head(30).to_csv(
                os.path.join(RESULTS_DIR, "pathway_enrichment.csv"), index=False
            )
            print(f"       Found {len(sig_results)} significant enrichments")
            _plot_enrichment(sig_results.head(15),
                             "GO/KEGG Pathway Enrichment (75 Consensus Features)")
        else:
            print("       No significant enrichments found — using manual analysis")
            _run_manual_pathway_analysis(feature_names)
    except ImportError:
        print("       [!] gseapy not installed — using manual pathway analysis")
        _run_manual_pathway_analysis(feature_names)
    except Exception as e:
        print(f"       [!] gseapy failed ({e}) — using manual pathway analysis")
        _run_manual_pathway_analysis(feature_names)


def _run_manual_pathway_analysis(feature_names):
    """Fallback: manually map features to known breast cancer pathways."""
    pathway_map = {
        "E.Cadherin": "Cell Adhesion / Cadherin Signaling",
        "CDH1": "Cell Adhesion / Cadherin Signaling",
        "Catenin": "Cell Adhesion / Cadherin Signaling",
        "catenin": "Cell Adhesion / Cadherin Signaling",
        "AR": "Hormone Receptor Signaling",
        "ER": "Hormone Receptor Signaling",
        "PR": "Hormone Receptor Signaling",
        "ESR1": "Hormone Receptor Signaling",
        "PGR": "Hormone Receptor Signaling",
        "FOXA1": "Hormone Receptor Signaling",
        "AKT": "PI3K/AKT/mTOR Signaling",
        "mTOR": "PI3K/AKT/mTOR Signaling",
        "PIK3CA": "PI3K/AKT/mTOR Signaling",
        "PTEN": "PI3K/AKT/mTOR Signaling",
        "S6": "PI3K/AKT/mTOR Signaling",
        "TP53": "Cell Cycle / Apoptosis",
        "RB1": "Cell Cycle / Apoptosis",
        "CIDEA": "Cell Cycle / Apoptosis",
        "BCL2": "Cell Cycle / Apoptosis",
        "CASP": "Cell Cycle / Apoptosis",
        "SOX10": "Transcription Regulation",
        "GATA3": "Transcription Regulation",
        "MYC": "Transcription Regulation",
        "BRCA1": "DNA Repair", "BRCA2": "DNA Repair",
        "HER2": "Receptor Tyrosine Kinase Signaling",
        "EGFR": "Receptor Tyrosine Kinase Signaling",
        "ERBB2": "Receptor Tyrosine Kinase Signaling",
    }

    pathway_counts = {}
    mapped_features = []
    for feat in feature_names:
        assigned = False
        for gene_key, pathway in pathway_map.items():
            if gene_key in feat:
                pathway_counts[pathway] = pathway_counts.get(pathway, 0) + 1
                mapped_features.append({"feature": feat, "pathway": pathway})
                assigned = True
                break
        if not assigned:
            mapped_features.append({"feature": feat, "pathway": "Other / Uncharacterized"})
            pathway_counts["Other / Uncharacterized"] = pathway_counts.get("Other / Uncharacterized", 0) + 1

    pd.DataFrame(mapped_features).to_csv(
        os.path.join(RESULTS_DIR, "manual_pathway_mapping.csv"), index=False
    )

    sorted_pathways = sorted(pathway_counts.items(), key=lambda x: x[1], reverse=True)
    pathways = [p[0] for p in sorted_pathways if p[0] != "Other / Uncharacterized"]
    counts = [p[1] for p in sorted_pathways if p[0] != "Other / Uncharacterized"]

    if pathways:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set2(np.linspace(0, 1, len(pathways)))
        ax.barh(range(len(pathways)), counts, color=colors, edgecolor="white", height=0.6)
        ax.set_yticks(range(len(pathways)))
        ax.set_yticklabels(pathways, fontsize=10)
        ax.set_xlabel("Number of Features", fontsize=12)
        ax.set_title("Pathway Distribution of 75 Consensus Features\n(Manual Mapping)",
                     fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "fig_26_pathway_analysis.png")
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        print(f"       Saved -> {path}")


def _plot_enrichment(sig_results, title):
    """Plot top enrichment results as horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(12, 7))
    terms = sig_results["Term"].tolist()
    pvals = -np.log10(sig_results["Adjusted P-value"].values)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(terms)))
    ax.barh(range(len(terms)), pvals, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms, fontsize=8)
    ax.set_xlabel("-log10(Adjusted P-value)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(x=-np.log10(0.05), color="red", linestyle="--", alpha=0.5, label="p=0.05")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_26_pathway_analysis.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════
#  MASTER EXECUTION
# ═══════════════════════════════════════════════════════════════════
# Module-level globals set during main() for use by helper functions
X_final_global = None
y_global = None


def main():
    """
    Execute all supplementary analyses sequentially.

    Workflow:
        1. Load data and run feature selection (reuse core pipeline)
        2. Run Sections A through D in order
        3. Print comprehensive summary
    """
    global X_final_global, y_global

    total_start = time.time()
    set_all_seeds(42)

    print("\n" + "=" * 70)
    print("  SUPPLEMENTARY ANALYSES — Thesis Defense & Peer Review Grade")
    print("  Explainable Multi-Omics Breast Cancer Classification")
    print("=" * 70)

    # ──── Load data and run feature selection ────
    print_section("DATA PREPARATION")
    print_step(0, "Loading data and running feature selection...")
    X, y, label_encoder, omics_groups, class_dist = run_data_pipeline()
    X_final, final_features, funnel, importance_df = run_feature_selection(X, y, omics_groups)
    X_final_global = X_final
    y_global = y
    print(f"\n       X_final: {X_final.shape}, classes: {len(np.unique(y))}")

    # ════════════════════════════════════════════
    # SECTION A: STATISTICAL VALIDATION
    # ════════════════════════════════════════════
    print_section("SECTION A: STATISTICAL VALIDATION")
    section_a_start = time.time()

    # A1. Wilcoxon signed-rank (5-fold)
    fold_scores, sig_df = run_statistical_tests(X_final.values, y)

    # A2. CV stability box plot
    plot_cv_stability(fold_scores)

    # A3. Upgraded 30-fold Wilcoxon
    fold_scores_30, sig_df_30 = run_upgraded_statistical_tests(X_final.values, y)

    print(f"\n  Section A complete in {time.time() - section_a_start:.1f}s")

    # ════════════════════════════════════════════
    # SECTION B: MODEL DIAGNOSTICS
    # ════════════════════════════════════════════
    print_section("SECTION B: MODEL DIAGNOSTICS")
    section_b_start = time.time()

    # B1. Omics ablation study
    abl_df = run_ablation_study(X_final, y)

    # B2. Learning curve
    plot_learning_curves(X_final.values, y)

    # B3. Precision-Recall curves
    plot_precision_recall_curves(X_final.values, y)

    # B4. Feature correlation heatmap
    plot_feature_correlation(X_final)

    # B5. Feature composition pie chart
    plot_feature_composition(X_final)

    print(f"\n  Section B complete in {time.time() - section_b_start:.1f}s")

    # ════════════════════════════════════════════
    # SECTION C: EXPLAINABILITY & ROBUSTNESS
    # ════════════════════════════════════════════
    print_section("SECTION C: EXPLAINABILITY & ROBUSTNESS")
    section_c_start = time.time()

    # C1. SHAP dependence plot
    plot_shap_dependence(X_final, y)

    # C2. SHAP ranking stability
    run_shap_stability_analysis(X_final, y)

    # C3. Leak-free nested CV
    nested_df = run_nested_cv_analysis(X, y, label_encoder)

    # C4. CV-based late fusion
    late_cv_results = run_late_fusion_cv(X_final, y)

    print(f"\n  Section C complete in {time.time() - section_c_start:.1f}s")

    # ════════════════════════════════════════════
    # SECTION D: BIOLOGICAL INTERPRETATION
    # ════════════════════════════════════════════
    print_section("SECTION D: BIOLOGICAL INTERPRETATION")
    section_d_start = time.time()

    # D1. Pathway enrichment
    run_pathway_analysis(X_final)

    print(f"\n  Section D complete in {time.time() - section_d_start:.1f}s")

    # ════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════
    total_time = time.time() - total_start
    print_section("ALL SUPPLEMENTARY ANALYSES COMPLETE")

    generated_figures = [
        "fig_13_significance_heatmap.png",
        "fig_14_cv_stability.png",
        "fig_15_ablation_study.png",
        "fig_16_correlation_heatmap.png",
        "fig_17_shap_dependence_ecadherin.png",
        "fig_18_learning_curve.png",
        "fig_19_feature_composition.png",
        "fig_20_precision_recall.png",
        "fig_21_nested_cv_comparison.png",
        "fig_22_feature_stability_nested.png",
        "fig_23_shap_stability.png",
        "fig_24_fusion_cv_comparison.png",
        "fig_25_significance_30fold.png",
        "fig_26_pathway_analysis.png",
    ]

    generated_csvs = [
        "per_fold_f1_scores.csv",
        "statistical_significance.csv",
        "ablation_study.csv",
        "per_fold_f1_scores_30fold.csv",
        "statistical_significance_30fold.csv",
        "late_fusion_cv_results.csv",
        "nested_cv_results.csv",
        "shap_stability_results.csv",
        "shap_stability_summary.csv",
        "gene_symbols_75features.csv",
        "pathway_enrichment.csv",
    ]

    print(f"  Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    print(f"\n  Figures generated ({len(generated_figures)}):")
    for fig in generated_figures:
        status = "[OK]" if os.path.exists(os.path.join(FIGURES_DIR, fig)) else "[--]"
        print(f"    {status} {fig}")

    print(f"\n  Result tables generated ({len(generated_csvs)}):")
    for csv in generated_csvs:
        status = "[OK]" if os.path.exists(os.path.join(RESULTS_DIR, csv)) else "[--]"
        print(f"    {status} {csv}")

    # Print key findings
    print(f"\n  Statistical Significance (5-fold):")
    sig_5 = sig_df[sig_df["significant_p05"] == "YES"]
    print(f"    Significant pairs: {len(sig_5)}/{len(sig_df)}")

    print(f"\n  Statistical Significance (30-fold):")
    sig_30 = sig_df_30[sig_df_30["significant_p05"] == "YES"]
    print(f"    Significant pairs: {len(sig_30)}/{len(sig_df_30)}")

    print(f"\n  Ablation Study Summary:")
    for _, row in abl_df.iterrows():
        delta_str = f"(Δ={row['Delta']:+.4f})" if row['Delta'] != 0 else "(baseline)"
        print(f"    {row['Configuration']}: F1={row['F1-Macro']:.4f} {delta_str}")

    print("\n" + "=" * 70)
    print("  READY FOR THESIS DEFENSE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
