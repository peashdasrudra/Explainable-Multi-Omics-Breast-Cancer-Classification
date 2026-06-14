"""
shap_analysis.py -- Day 4: SHAP Analysis -- The Core Novelty.

Cross-Omics SHAP Attribution: quantify which omics layer (mRNA/CNV/Methylation/Protein)
drives each class prediction -- clinically meaningful and underexplored.

Generates 4 types of SHAP figures:
1. Global SHAP Beeswarm (fig_07)
2. Omics Attribution Grouped Bar (fig_08) -- KEY SCIENTIFIC FINDING
3. Per-Class SHAP Summary (fig_09)
4. Patient Waterfall (fig_10)
"""
import numpy as np
import pandas as pd
import os
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    RANDOM_STATE, OMICS_SHORT_NAMES, OMICS_COLORS,
    FIGURES_DIR, RESULTS_DIR, MODELS_DIR, PREPROCESSED_DIR,
    FIGURE_DPI
)
from src.utils import print_step


def prepare_shap_data(X_final, y):
    """
    Step 56-58: Split data and fit best model fresh for SHAP.
    Do NOT reuse the cross-validated version -- fit fresh.
    """
    print_step(24, "Preparing data for SHAP analysis")

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print(f"       Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def train_best_model_for_shap(X_train, y_train, xgb_params=None):
    """
    Fit the best model fresh on training data for SHAP analysis.
    Uses XGBoost (tuned) as it's tree-based and works well with TreeExplainer.
    """
    print_step(25, "Training best model fresh for SHAP...")

    # Use XGBoost as it works best with TreeExplainer
    if xgb_params:
        params = {k.replace("clf__", ""): v for k, v in xgb_params.items()}
    else:
        params = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1}

    # Scale the data first
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = XGBClassifier(
        **params, random_state=RANDOM_STATE, n_jobs=-1,
        eval_metric="mlogloss", use_label_encoder=False, verbosity=0
    )
    model.fit(X_train_scaled, y_train)

    print(f"       Model trained: XGBoost with params {params}")

    return model, scaler


def compute_shap_values(model, X_test, scaler, feature_names):
    """
    Step 59-60: Compute SHAP values using TreeExplainer.
    For binary classification, expands to both classes:
      class 0 SHAP = -shap_values, class 1 SHAP = +shap_values.
    Returns shap_values with shape (n_samples, n_features, n_classes).
    """
    print_step(26, "Computing SHAP values (TreeExplainer)...")

    X_test_scaled = scaler.transform(X_test)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_test_df)

    if isinstance(raw_shap, list):
        # Multi-class: list of arrays per class
        shap_values = np.stack(raw_shap, axis=-1)  # (n_samples, n_features, n_classes)
        print(f"       SHAP values shape: {shap_values.shape} (samples x features x classes)")
    else:
        # Binary: single 2D array for positive class
        # Expand: class 0 = -values, class 1 = +values (symmetric for binary)
        shap_values = np.stack([-raw_shap, raw_shap], axis=-1)
        print(f"       SHAP values shape: {shap_values.shape} (binary -> expanded to 2 classes)")

    return shap_values, X_test_df, explainer


def plot_global_beeswarm(shap_values, X_test_df, label_encoder=None):
    """
    Figure 7 -- Global SHAP Beeswarm: shows top 20 most important features.
    Uses mean |SHAP| across classes for a combined global view.
    """
    print_step(27, "Generating Global SHAP Beeswarm (fig_07)...")

    # Use mean absolute SHAP across classes for global view
    mean_abs_shap = np.abs(shap_values).mean(axis=2)

    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        mean_abs_shap, X_test_df,
        plot_type="dot", max_display=20, show=False
    )

    plt.title("Global SHAP Feature Importance (Top 20)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig_07_shap_beeswarm.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


def compute_omics_attribution(shap_values, feature_names, label_encoder=None):
    """
    Steps 63-67 -- Cross-Omics SHAP Attribution: THE KEY SCIENTIFIC FINDING.

    For each class, compute sum of |SHAP values| per omics group,
    then convert to percentages. Result: 4 x n_classes table.
    """
    print_step(28, "Computing Cross-Omics SHAP Attribution (KEY FINDING)...")

    n_classes = shap_values.shape[2]

    if label_encoder is not None:
        class_names = list(label_encoder.classes_)
    else:
        class_names = [f"Class_{i}" for i in range(n_classes)]

    # Shorten class names for display
    short_names = []
    for cn in class_names:
        if "ductal" in cn.lower():
            short_names.append("IDC")
        elif "lobular" in cn.lower():
            short_names.append("ILC")
        else:
            short_names.append(cn[:20])

    # Group feature indices by omics prefix
    omics_indices = {}
    for prefix, name in OMICS_SHORT_NAMES.items():
        indices = [i for i, f in enumerate(feature_names) if f.startswith(prefix)]
        if indices:
            omics_indices[name] = indices

    # Compute attribution percentages
    attribution = {}
    for cls_idx in range(n_classes):
        cls_name = short_names[cls_idx] if cls_idx < len(short_names) else f"Class_{cls_idx}"
        cls_shap = shap_values[:, :, cls_idx]
        total_abs_shap = np.abs(cls_shap).sum()

        for omics_name, indices in omics_indices.items():
            omics_shap = np.abs(cls_shap[:, indices]).sum()
            pct = (omics_shap / total_abs_shap) * 100 if total_abs_shap > 0 else 0.0

            if omics_name not in attribution:
                attribution[omics_name] = {}
            attribution[omics_name][cls_name] = round(pct, 2)

    # Convert to DataFrame
    attr_df = pd.DataFrame(attribution).T
    attr_df.index.name = "Omics Layer"

    # Add total row to verify 100%
    attr_df.loc["TOTAL"] = attr_df.sum()

    print(f"\n       Cross-Omics SHAP Attribution Table (%)")
    print(f"       {attr_df.to_string()}")

    # Save
    attr_path = os.path.join(RESULTS_DIR, "omics_attribution.csv")
    attr_df.to_csv(attr_path)
    print(f"\n       Saved -> {attr_path}")

    return attr_df


def plot_omics_attribution(attr_df):
    """
    Figure 8 -- Omics Attribution Grouped Bar Chart.
    X-axis = omics type, Y-axis = % contribution, grouped bars = classes.
    THIS IS THE PAPER FIGURE for any conference submission.
    """
    print_step(29, "Generating Omics Attribution Bar Chart (fig_08)...")

    # Remove TOTAL row for plotting
    plot_df = attr_df.drop("TOTAL", errors="ignore")

    fig, ax = plt.subplots(figsize=(12, 7))

    n_omics = len(plot_df)
    n_classes = len(plot_df.columns)
    bar_width = 0.35
    x = np.arange(n_omics)

    # Use distinct colors for IDC vs ILC
    class_colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]

    for i, cls_name in enumerate(plot_df.columns):
        offset = (i - n_classes / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, plot_df[cls_name],
            width=bar_width, label=cls_name,
            color=class_colors[i % len(class_colors)],
            edgecolor="white", linewidth=0.5
        )
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 2:  # Only label bars > 2%
                ax.text(
                    bar.get_x() + bar.get_width() / 2., height + 0.8,
                    f"{height:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold"
                )

    ax.set_xlabel("Omics Layer", fontsize=13, fontweight="bold")
    ax.set_ylabel("SHAP Attribution (%)", fontsize=13, fontweight="bold")
    ax.set_title("Cross-Omics SHAP Attribution per Histological Subtype",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, fontsize=12)
    ax.legend(title="Subtype", fontsize=11, title_fontsize=12,
              loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(plot_df.max()) * 1.2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_08_omics_attribution.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


def plot_per_class_shap(shap_values, X_test_df, label_encoder=None):
    """
    Figure 9 -- Per-Class SHAP Summary (one per class).
    For binary: generates one figure for IDC, one for ILC.
    """
    print_step(30, "Generating Per-Class SHAP plots (fig_09)...")

    n_classes = shap_values.shape[2]

    if label_encoder is not None:
        class_names = list(label_encoder.classes_)
    else:
        class_names = [f"Class_{i}" for i in range(n_classes)]

    for cls_idx in range(n_classes):
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class_{cls_idx}"

        # Shorten for title
        if "ductal" in cls_name.lower():
            short = "IDC (Infiltrating Ductal Carcinoma)"
        elif "lobular" in cls_name.lower():
            short = "ILC (Infiltrating Lobular Carcinoma)"
        else:
            short = cls_name

        cls_shap = shap_values[:, :, cls_idx]

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            cls_shap, X_test_df,
            plot_type="dot", max_display=15, show=False
        )
        plt.title(f"SHAP Feature Importance -- {short}",
                  fontsize=13, fontweight="bold")
        plt.tight_layout()

        # Clean filename
        if "ductal" in cls_name.lower():
            safe_name = "IDC"
        elif "lobular" in cls_name.lower():
            safe_name = "ILC"
        else:
            safe_name = cls_name.replace(" ", "_").replace("/", "_")

        path = os.path.join(FIGURES_DIR, f"fig_09_shap_{safe_name}.png")
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        print(f"       Saved -> {path}")


def plot_waterfall(shap_values, X_test_df, y_test, model, scaler, label_encoder=None):
    """
    Figure 10 -- Patient Waterfall: one correctly classified patient per class.
    """
    print_step(31, "Generating Patient Waterfall plots (fig_10)...")

    X_test_scaled = scaler.transform(X_test_df.values)
    y_pred = model.predict(X_test_scaled)

    # Find correctly classified patients
    correct_mask = (y_pred == y_test)
    correct_indices = np.where(correct_mask)[0]

    if len(correct_indices) == 0:
        print("       [!] No correctly classified patients found!")
        return

    n_classes = shap_values.shape[2]

    if label_encoder is not None:
        class_names = list(label_encoder.classes_)
    else:
        class_names = [f"Class_{i}" for i in range(n_classes)]

    # Pick one correct patient per class
    patients_plotted = 0
    for cls_idx in range(n_classes):
        # Find a correctly classified patient of this class
        cls_correct = [i for i in correct_indices if y_test[i] == cls_idx]
        if len(cls_correct) == 0:
            continue

        patient_idx = cls_correct[0]
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class_{cls_idx}"

        # Shorten name
        if "ductal" in cls_name.lower():
            short = "IDC"
        elif "lobular" in cls_name.lower():
            short = "ILC"
        else:
            short = cls_name

        plt.figure(figsize=(12, 8))

        patient_shap = shap_values[patient_idx, :, cls_idx]

        # Create an Explanation object
        explanation = shap.Explanation(
            values=patient_shap,
            base_values=0,
            data=X_test_df.iloc[patient_idx].values,
            feature_names=X_test_df.columns.tolist()
        )

        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.title(f"Patient #{patient_idx} -- Predicted: {short}",
                  fontsize=13, fontweight="bold")
        plt.tight_layout()

        patients_plotted += 1
        path = os.path.join(FIGURES_DIR, f"fig_10_waterfall_p{patients_plotted}.png")
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        print(f"       Saved -> {path}")


def run_shap_analysis(X_final, y, label_encoder, xgb_params=None):
    """
    Execute the full SHAP analysis pipeline (Day 4).
    Returns the omics attribution DataFrame.
    """
    feature_names = list(X_final.columns) if hasattr(X_final, 'columns') else \
        [f"f_{i}" for i in range(X_final.shape[1])]

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_shap_data(X_final, y)

    # Train model fresh
    model, scaler = train_best_model_for_shap(X_train.values, y_train, xgb_params)

    # Compute SHAP values (always returns 3D: samples x features x classes)
    shap_values, X_test_df, explainer = compute_shap_values(
        model, X_test.values, scaler, feature_names
    )

    # Figure 7: Global Beeswarm
    plot_global_beeswarm(shap_values, X_test_df, label_encoder)

    # Compute Omics Attribution (KEY FINDING)
    attr_df = compute_omics_attribution(shap_values, feature_names, label_encoder)

    # Figure 8: Omics Attribution Bar
    plot_omics_attribution(attr_df)

    # Figure 9: Per-Class SHAP (one for IDC, one for ILC)
    plot_per_class_shap(shap_values, X_test_df, label_encoder)

    # Figure 10: Patient Waterfall (one per class)
    plot_waterfall(shap_values, X_test_df, y_test, model, scaler, label_encoder)

    return attr_df, model, scaler
