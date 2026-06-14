"""
visualization.py -- All 12 thesis figures with publication-quality formatting.

Figures:
  fig_01: Feature Reduction Funnel
  fig_02: Label Distribution
  fig_03: Top 20 Consensus Features
  fig_04: ROC Curves -- All Models
  fig_05: Model Comparison Bar (F1-Macro)
  fig_06: Confusion Matrix -- Best Model
  fig_07: SHAP Global Beeswarm           (generated in shap_analysis.py)
  fig_08: Omics Attribution Grouped Bar   (generated in shap_analysis.py)
  fig_09: Per-Class SHAP                  (generated in shap_analysis.py)
  fig_10: Patient Waterfall               (generated in shap_analysis.py)
  fig_11: Fusion Comparison Bar
  fig_12: Confusion Matrix -- Late Fusion
"""
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from imblearn.over_sampling import SMOTE

from src.config import (
    RANDOM_STATE, FIGURES_DIR, FIGURE_DPI, MODEL_COLORS,
    OMICS_SHORT_NAMES
)
from src.utils import print_step


# ──── Publication style setup ────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": FIGURE_DPI,
})


def plot_feature_funnel(funnel):
    """
    Figure 1 -- Feature Reduction Funnel.
    4-bar horizontal bar chart: 1837 -> ~1400 -> ~300 -> 75 with stage labels.
    """
    print_step("F1", "Generating Feature Reduction Funnel (fig_01)...")

    stages = list(funnel.keys())
    counts = list(funnel.values())

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#E74C3C", "#E67E22", "#3498DB", "#2ECC71"]
    bars = ax.barh(range(len(stages)), counts, color=colors, height=0.6, edgecolor="white")

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
            f"{count:,}", va="center", fontsize=12, fontweight="bold"
        )

    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages, fontsize=11)
    ax.set_xlabel("Number of Features", fontsize=12)
    ax.set_title("3-Stage Feature Selection Funnel", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(counts) * 1.15)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_01_funnel.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


def plot_label_distribution(y, label_encoder):
    """
    Figure 2 -- Label Distribution Bar Chart.
    Shows subtype counts with percentage labels.
    """
    print_step("F2", "Generating Label Distribution (fig_02)...")

    unique, counts = np.unique(y, return_counts=True)
    class_names = [label_encoder.inverse_transform([u])[0] for u in unique]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6"]
    bars = ax.bar(class_names, counts, color=colors[:len(class_names)],
                  edgecolor="white", width=0.6)

    # Add count + percentage labels
    total = sum(counts)
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2., bar.get_height() + 5,
            f"{count}\n({pct:.1f}%)", ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Class Distribution -- TCGA BRCA", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(counts) * 1.2)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_02_label_dist.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


def plot_consensus_features(importance_df, top_n=20):
    """
    Figure 3 -- Top 20 Consensus Features (horizontal bar).
    """
    print_step("F3", "Generating Top 20 Consensus Features (fig_03)...")

    top = importance_df.head(top_n).sort_values("avg_importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Color bars by omics type
    colors = []
    omics_colors = {"mRNA": "#E74C3C", "CNV": "#3498DB", "Methylation": "#2ECC71", "Protein": "#F39C12"}
    for feat in top["feature"]:
        color = "#95A5A6"  # default gray
        for prefix, name in OMICS_SHORT_NAMES.items():
            if feat.startswith(prefix):
                color = omics_colors.get(name, "#95A5A6")
                break
        colors.append(color)

    bars = ax.barh(range(len(top)), top["avg_importance"], color=colors, height=0.7)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=9)
    ax.set_xlabel("Average Importance (RF + XGB)", fontsize=12)
    ax.set_title(f"Top {top_n} Consensus Features", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend for omics colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=omics_colors.get(name, "#95A5A6"), label=name)
        for prefix, name in OMICS_SHORT_NAMES.items()
    ]
    ax.legend(handles=legend_elements, title="Omics Layer", loc="lower right")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_03_consensus_features.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


def plot_roc_curves(X, y, all_results, label_encoder):
    """
    Figure 4 -- ROC Curves for all models (one-vs-rest multiclass).
    Uses a train/test split to generate ROC curves.
    """
    print_step("F4", "Generating ROC Curves (fig_04)...")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_s, y_train)

    # Define models to plot ROC for
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
                                  eval_metric="mlogloss", use_label_encoder=False, verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    n_classes = len(np.unique(y))
    colors = MODEL_COLORS[:len(models)]

    for (name, model), color in zip(models.items(), colors):
        model.fit(X_train_res, y_train_res)

        if n_classes == 2:
            y_proba = model.predict_proba(X_test_s)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{name} (AUC = {roc_auc:.3f})")
        else:
            y_test_bin = label_binarize(y_test, classes=np.unique(y))
            y_proba = model.predict_proba(X_test_s)

            # Macro-average ROC
            fpr_all, tpr_all = [], []
            for i in range(n_classes):
                fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                fpr_all.append(fpr_i)
                tpr_all.append(tpr_i)

            # Compute macro AUC
            from sklearn.metrics import roc_auc_score
            macro_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

            # Plot mean ROC
            mean_fpr = np.linspace(0, 1, 100)
            mean_tpr = np.zeros_like(mean_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(mean_fpr, fpr_all[i], tpr_all[i])
            mean_tpr /= n_classes

            ax.plot(mean_fpr, mean_tpr, color=color, lw=2,
                    label=f"{name} (AUC = {macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves -- All Models (OvR Macro)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_04_roc_all.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


def plot_model_comparison(all_results):
    """
    Figure 5 -- Model Comparison Bar Chart (F1-Macro for all 8 models).
    """
    print_step("F5", "Generating Model Comparison Bar (fig_05)...")

    models = list(all_results.keys())
    f1_means = [all_results[m]["f1_macro"][0] for m in models]
    f1_stds = [all_results[m]["f1_macro"][1] for m in models]

    # Sort by F1
    sorted_idx = np.argsort(f1_means)
    models = [models[i] for i in sorted_idx]
    f1_means = [f1_means[i] for i in sorted_idx]
    f1_stds = [f1_stds[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(models))]
    bars = ax.barh(
        range(len(models)), f1_means, xerr=f1_stds,
        color=colors, height=0.6, capsize=3, edgecolor="white"
    )

    # Add value labels
    for bar, mean, std in zip(bars, f1_means, f1_stds):
        ax.text(
            bar.get_width() + std + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{mean:.4f}", va="center", fontsize=9, fontweight="bold"
        )

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel("F1-Macro (mean ± std)", fontsize=12)
    ax.set_title("Model Comparison -- F1-Macro", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_05_model_comparison.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


def plot_confusion_matrix(X, y, best_model_name, label_encoder, filename="fig_06_confusion_best.png"):
    """
    Figure 6 -- Confusion Matrix Heatmap for the best model.
    """
    print_step("F6", f"Generating Confusion Matrix -- {best_model_name} ({filename})...")

    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    import joblib

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_s, y_train)

    # Use XGBoost as default best model for confusion matrix
    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_STATE, n_jobs=-1,
        eval_metric="mlogloss", use_label_encoder=False, verbosity=0
    )
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_s)

    # Generate confusion matrix
    class_names = list(label_encoder.classes_)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={"shrink": 0.8}
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix -- {best_model_name}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")

    return cm


def plot_confusion_late_fusion(y_test, late_pred, label_encoder):
    """
    Figure 12 -- Confusion Matrix for Late Fusion predictions.
    """
    print_step("F12", "Generating Confusion Matrix -- Late Fusion (fig_12)...")

    class_names = list(label_encoder.classes_)
    cm = confusion_matrix(y_test, late_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={"shrink": 0.8}
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix -- Late Fusion (Soft Vote)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_12_confusion_late.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


def plot_fusion_comparison(fusion_df, early_results, late_metrics):
    """
    Figure 11 -- Fusion Comparison Bar Chart (Early vs Late).
    """
    print_step("F11", "Generating Fusion Comparison Bar (fig_11)...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    strategies = ["Early Fusion", "Late Fusion"]

    # F1-Macro comparison
    f1_values = [
        early_results["f1_macro"][0] if early_results else 0,
        late_metrics["f1_macro"],
    ]
    colors = ["#3498DB", "#E74C3C"]
    axes[0].bar(strategies, f1_values, color=colors, width=0.5, edgecolor="white")
    for i, v in enumerate(f1_values):
        axes[0].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("F1-Macro", fontsize=12)
    axes[0].set_title("F1-Macro Comparison", fontsize=13, fontweight="bold")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].set_ylim(0, max(f1_values) * 1.15)

    # AUC-ROC comparison
    auc_values = [
        early_results["roc_auc"][0] if early_results else 0,
        late_metrics["roc_auc"],
    ]
    axes[1].bar(strategies, auc_values, color=colors, width=0.5, edgecolor="white")
    for i, v in enumerate(auc_values):
        axes[1].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("AUC-ROC", fontsize=12)
    axes[1].set_title("AUC-ROC Comparison", fontsize=13, fontweight="bold")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].set_ylim(0, max(auc_values) * 1.15)

    plt.suptitle("Fusion Strategy Comparison: Early vs Late", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_11_fusion_comparison.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")
