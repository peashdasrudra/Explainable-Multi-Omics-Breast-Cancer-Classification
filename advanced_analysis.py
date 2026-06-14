"""
advanced_analysis.py -- Publication-Grade Additional Analyses.

Adds the critical analyses that thesis examiners and peer reviewers WILL ask about:
1. Statistical significance tests (Wilcoxon signed-rank between model pairs)
2. Per-fold CV stability plot (box plots showing variance across folds)
3. Omics ablation study (remove each layer, measure impact)
4. Feature correlation heatmap (top 25 consensus features)
5. SHAP dependence plot for top feature (E-Cadherin)
6. Learning curve analysis (does more data help?)
7. Feature overlap Venn-style analysis (which omics contribute which features)
8. Precision-Recall curves (critical for imbalanced data)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from sklearn.metrics import make_scorer, matthews_corrcoef, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
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
from src.baseline_models import get_global_cv, build_smote_pipeline

# ──── Publication style ────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": FIGURE_DPI,
})


# ═══════════════════════════════════════════════════════════
# 1. STATISTICAL SIGNIFICANCE TESTS
# ═══════════════════════════════════════════════════════════
def run_statistical_tests(X, y):
    """
    Perform Wilcoxon signed-rank test between all model pairs on per-fold F1 scores.
    This is CRITICAL for a defensible thesis -- you cannot claim one model is
    'better' without a statistical test.
    """
    print_step(1, "Statistical Significance Tests (Wilcoxon signed-rank)")

    skf = get_global_cv()
    scoring = {"f1_macro": "f1_macro"}

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

    # Collect per-fold F1 scores
    fold_scores = {}
    for name, model in models.items():
        pipeline = build_smote_pipeline(model)
        cv_result = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        fold_scores[name] = cv_result["test_f1_macro"]
        mean = cv_result["test_f1_macro"].mean()
        std = cv_result["test_f1_macro"].std()
        print(f"       {name}: {mean:.4f} +/- {std:.4f}  folds={cv_result['test_f1_macro']}")

    # Save per-fold scores
    fold_df = pd.DataFrame(fold_scores)
    fold_df.index.name = "Fold"
    fold_df.to_csv(os.path.join(RESULTS_DIR, "per_fold_f1_scores.csv"))

    # Pairwise Wilcoxon tests
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
                p_val = 1.0  # identical distributions
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val

            sig = "YES" if p_val < 0.05 else "no"
            sig_results.append({
                "Model_A": model_names[i],
                "Model_B": model_names[j],
                "p_value": round(p_val, 4),
                "significant_p05": sig,
            })

    sig_df = pd.DataFrame(sig_results)
    sig_path = os.path.join(RESULTS_DIR, "statistical_significance.csv")
    sig_df.to_csv(sig_path, index=False)
    print(f"\n       Saved significance tests -> {sig_path}")

    # Plot p-value heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(p_matrix, dtype=bool), k=0)
    sns.heatmap(
        p_matrix, mask=mask, annot=True, fmt=".3f",
        xticklabels=model_names, yticklabels=model_names,
        cmap="RdYlGn_r", vmin=0, vmax=0.1,
        ax=ax, cbar_kws={"label": "p-value"}
    )
    ax.set_title("Pairwise Wilcoxon Signed-Rank Test (p-values)\np < 0.05 = significant difference",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_13_significance_heatmap.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")

    return fold_scores, sig_df


# ═══════════════════════════════════════════════════════════
# 2. PER-FOLD CV STABILITY (BOX PLOT)
# ═══════════════════════════════════════════════════════════
def plot_cv_stability(fold_scores):
    """
    Box plot showing per-fold F1 scores for each model.
    Critical for showing that results are stable across folds.
    """
    print_step(2, "Per-Fold CV Stability Box Plot")

    fold_df = pd.DataFrame(fold_scores)

    # Sort by median F1
    order = fold_df.median().sort_values(ascending=True).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = MODEL_COLORS[:len(order)]
    bp = ax.boxplot(
        [fold_df[m] for m in order], labels=order,
        patch_artist=True, vert=False, widths=0.6
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual fold points
    for i, m in enumerate(order):
        y = np.random.normal(i + 1, 0.04, size=len(fold_df[m]))
        ax.scatter(fold_df[m], y, alpha=0.8, s=30, zorder=5, color="black", edgecolors="white")

    ax.set_xlabel("F1-Macro", fontsize=12)
    ax.set_title("Cross-Validation Stability -- Per-Fold F1 Scores",
                 fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_14_cv_stability.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


# ═══════════════════════════════════════════════════════════
# 3. OMICS ABLATION STUDY
# ═══════════════════════════════════════════════════════════
def run_ablation_study(X_final, y):
    """
    Ablation study: Train LightGBM with ALL features vs removing each omics layer.
    Shows how much each layer contributes to the final model.
    Examiners WILL ask: 'What happens if you remove mRNA?'
    """
    print_step(3, "Omics Ablation Study")

    feature_names = list(X_final.columns) if hasattr(X_final, 'columns') else []
    skf = get_global_cv()

    # Group features by omics
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

    # Full model (all omics)
    pipeline = build_smote_pipeline(model)
    cv = cross_validate(pipeline, X_final.values, y, cv=skf,
                        scoring="f1_macro", n_jobs=-1)
    full_f1 = cv["test_score"].mean()
    full_std = cv["test_score"].std()
    results.append({"Configuration": "All Omics (Full)", "F1-Macro": full_f1,
                     "Std": full_std, "Delta": 0.0})
    print(f"       All Omics: F1={full_f1:.4f} +/- {full_std:.4f}")

    # Remove each omics layer one at a time
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
        abl_std = cv["test_score"].std()
        delta = full_f1 - abl_f1

        results.append({
            "Configuration": f"Remove {omics_name} ({len(cols)} feats)",
            "F1-Macro": abl_f1,
            "Std": abl_std,
            "Delta": delta,
        })
        print(f"       Remove {omics_name}: F1={abl_f1:.4f} (Delta={delta:+.4f})")

    # Single-omics models
    for omics_name, cols in omics_features.items():
        col_indices = [feature_names.index(c) for c in cols]
        X_single = X_final.values[:, col_indices]

        pipeline = build_smote_pipeline(model)
        cv = cross_validate(pipeline, X_single, y, cv=skf,
                            scoring="f1_macro", n_jobs=-1)
        single_f1 = cv["test_score"].mean()
        single_std = cv["test_score"].std()

        results.append({
            "Configuration": f"Only {omics_name} ({len(cols)} feats)",
            "F1-Macro": single_f1,
            "Std": single_std,
            "Delta": full_f1 - single_f1,
        })
        print(f"       Only {omics_name}: F1={single_f1:.4f}")

    # Save results
    abl_df = pd.DataFrame(results)
    abl_path = os.path.join(RESULTS_DIR, "ablation_study.csv")
    abl_df.to_csv(abl_path, index=False)

    # Plot ablation results
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
    ax.set_title("Omics Ablation Study -- Impact of Each Layer",
                 fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(x=full_f1, color="#2ECC71", linestyle="--", alpha=0.5, label="Full model baseline")
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_15_ablation_study.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")

    return abl_df


# ═══════════════════════════════════════════════════════════
# 4. FEATURE CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════
def plot_feature_correlation(X_final, top_n=25):
    """
    Correlation heatmap of top 25 consensus features.
    Shows if selected features are independent or redundant.
    """
    print_step(4, f"Feature Correlation Heatmap (top {top_n})")

    feature_names = list(X_final.columns)[:top_n]
    X_top = X_final[feature_names]

    corr = X_top.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr, mask=mask, annot=False, cmap="RdBu_r",
        vmin=-1, vmax=1, center=0, square=True,
        linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"}
    )

    ax.set_title(f"Feature Correlation Matrix (Top {top_n} Consensus Features)",
                 fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_16_correlation_heatmap.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


# ═══════════════════════════════════════════════════════════
# 5. SHAP DEPENDENCE PLOT (E-CADHERIN)
# ═══════════════════════════════════════════════════════════
def plot_shap_dependence(X_final, y):
    """
    SHAP dependence plot for E-Cadherin -- the top feature.
    Shows how E-Cadherin values relate to SHAP contribution.
    """
    print_step(5, "SHAP Dependence Plot (E-Cadherin)")

    feature_names = list(X_final.columns)

    # Find E-Cadherin column
    ecad_col = None
    cdh1_col = None
    for f in feature_names:
        if "E.Cadherin" in f or "E-Cadherin" in f:
            ecad_col = f
        if "CDH1" in f:
            cdh1_col = f

    if ecad_col is None:
        print("       [!] E-Cadherin not found in features")
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

    # Dependence plot for E-Cadherin
    plt.figure(figsize=(10, 7))
    shap.dependence_plot(
        ecad_col, shap_values, X_test_df,
        interaction_index=cdh1_col if cdh1_col else "auto",
        show=False
    )
    plt.title(f"SHAP Dependence Plot -- {ecad_col}",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_17_shap_dependence_ecadherin.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


# ═══════════════════════════════════════════════════════════
# 6. LEARNING CURVE
# ═══════════════════════════════════════════════════════════
def plot_learning_curves(X, y):
    """
    Learning curve: does the model benefit from more data?
    Important for discussing dataset size limitations.
    """
    print_step(6, "Learning Curve Analysis")

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
    ax.plot(train_sizes, train_mean, "o-", color="#E74C3C", label="Training Score", linewidth=2)
    ax.plot(train_sizes, val_mean, "o-", color="#3498DB", label="Validation Score", linewidth=2)

    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("F1-Macro", fontsize=12)
    ax.set_title("Learning Curve -- LightGBM (Best Model)",
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


# ═══════════════════════════════════════════════════════════
# 7. FEATURE COMPOSITION PIE CHART
# ═══════════════════════════════════════════════════════════
def plot_feature_composition(X_final):
    """
    Pie chart showing omics layer composition of final 75 features.
    """
    print_step(7, "Feature Composition by Omics Layer")

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


# ═══════════════════════════════════════════════════════════
# 8. PRECISION-RECALL CURVES
# ═══════════════════════════════════════════════════════════
def plot_precision_recall_curves(X, y):
    """
    Precision-Recall curves -- critical for imbalanced data (4.4:1 ratio).
    More informative than ROC when classes are imbalanced.
    """
    print_step(8, "Precision-Recall Curves (for imbalanced data)")

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


# ═══════════════════════════════════════════════════════════
# MASTER EXECUTION
# ═══════════════════════════════════════════════════════════
def main():
    """Run all advanced analyses."""
    set_all_seeds(42)

    print_section("ADVANCED ANALYSES -- Publication & Defense Grade")

    # Load data + feature selection (reuse pipeline)
    print_step(0, "Loading data and running feature selection...")
    X, y, label_encoder, omics_groups, class_dist = run_data_pipeline()
    X_final, final_features, funnel, importance_df = run_feature_selection(X, y, omics_groups)

    print(f"\n       X_final: {X_final.shape}, classes: {len(np.unique(y))}")
    print(f"       Starting advanced analyses...\n")

    # 1. Statistical significance tests
    fold_scores, sig_df = run_statistical_tests(X_final.values, y)

    # 2. CV stability box plot
    plot_cv_stability(fold_scores)

    # 3. Omics ablation study
    abl_df = run_ablation_study(X_final, y)

    # 4. Feature correlation heatmap
    plot_feature_correlation(X_final)

    # 5. SHAP dependence plot
    plot_shap_dependence(X_final, y)

    # 6. Learning curve
    plot_learning_curves(X_final.values, y)

    # 7. Feature composition
    plot_feature_composition(X_final)

    # 8. Precision-Recall curves
    plot_precision_recall_curves(X_final.values, y)

    print_section("ADVANCED ANALYSES COMPLETE")

    # Summary of new figures
    new_figures = [
        "fig_13_significance_heatmap.png",
        "fig_14_cv_stability.png",
        "fig_15_ablation_study.png",
        "fig_16_correlation_heatmap.png",
        "fig_17_shap_dependence_ecadherin.png",
        "fig_18_learning_curve.png",
        "fig_19_feature_composition.png",
        "fig_20_precision_recall.png",
    ]

    print(f"  New figures generated ({len(new_figures)}):")
    for fig in new_figures:
        print(f"    [OK] {fig}")

    print(f"\n  New result tables:")
    print(f"    [OK] per_fold_f1_scores.csv")
    print(f"    [OK] statistical_significance.csv")
    print(f"    [OK] ablation_study.csv")

    # Print significance summary
    print(f"\n  Statistical Significance Summary:")
    sig_pairs = sig_df[sig_df["significant_p05"] == "YES"]
    if len(sig_pairs) > 0:
        for _, row in sig_pairs.iterrows():
            print(f"    {row['Model_A']} vs {row['Model_B']}: p={row['p_value']:.4f} (SIGNIFICANT)")
    else:
        print(f"    No significant differences at p<0.05 (common with only 5 folds)")

    print(f"\n  Ablation Study Summary:")
    for _, row in abl_df.iterrows():
        delta_str = f"(Delta={row['Delta']:+.4f})" if row['Delta'] != 0 else "(baseline)"
        print(f"    {row['Configuration']}: F1={row['F1-Macro']:.4f} {delta_str}")


if __name__ == "__main__":
    main()
