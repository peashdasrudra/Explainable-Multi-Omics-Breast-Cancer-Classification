"""
fusion_comparison.py — Early vs Late Multi-Omics Fusion Comparison
====================================================================
Compares two multi-omics integration strategies for breast cancer
subtype classification:

Early Fusion (Feature-Level Integration):
    Concatenate all omics features into a single feature matrix and
    train a single model (Stacking Ensemble). This is the standard
    approach used throughout the pipeline.

Late Fusion (Decision-Level Integration):
    Train a separate XGBoost classifier per omics layer, generate
    per-model prediction probabilities, and average them (soft vote)
    to produce the final classification.

Rationale:
    If late fusion outperforms early fusion, it suggests that
    per-omics models capture complementary signals that are lost
    when features are naively concatenated. If early fusion wins,
    inter-omics feature interactions are important for classification.
"""
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.config import (
    RANDOM_STATE, OMICS_SHORT_NAMES, RESULTS_DIR, FIGURES_DIR,
    MODELS_DIR, FIGURE_DPI
)
from src.baseline_models import get_global_cv, evaluate_model_cv
from src.utils import print_step


def get_early_fusion_results(all_results):
    """
    Early Fusion = Stacking Ensemble result from Day 3.
    It already concatenates all omics features into a single model.
    """
    print_step(32, "Early Fusion (= Stacking Ensemble from Day 3)")

    if "Stacking Ensemble" in all_results:
        stacking = all_results["Stacking Ensemble"]
        f1 = stacking["f1_macro"][0]
        auc = stacking["roc_auc"][0]
        print(f"       F1-Macro: {f1:.4f}, AUC-ROC: {auc:.4f}")
        return stacking
    else:
        print("       [!] Stacking Ensemble results not found!")
        return None


def run_late_fusion(X_final, y, label_encoder):
    """
    Late Fusion: Train a separate XGBClassifier per omics group,
    generate prediction probabilities, average (soft vote), argmax.
    """
    print_step(33, "Late Fusion (per-omics model -> soft vote)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    feature_names = list(X_final.columns) if hasattr(X_final, 'columns') else []

    # Group features by omics
    omics_features = {}
    for prefix, name in OMICS_SHORT_NAMES.items():
        cols = [c for c in feature_names if c.startswith(prefix)]
        if cols:
            omics_features[name] = cols

    print(f"       Omics groups: {', '.join([f'{n}({len(c)})' for n, c in omics_features.items()])}")

    # Train per-omics models
    per_omics_probas = []
    per_omics_models = {}

    for omics_name, cols in omics_features.items():
        col_indices = [feature_names.index(c) for c in cols]

        X_tr_omics = X_train.iloc[:, col_indices] if hasattr(X_train, 'iloc') else X_train[:, col_indices]
        X_te_omics = X_test.iloc[:, col_indices] if hasattr(X_test, 'iloc') else X_test[:, col_indices]

        # StandardScaler + SMOTE + XGBoost
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

        proba = model.predict_proba(X_te_scaled)
        per_omics_probas.append(proba)
        per_omics_models[omics_name] = (model, scaler)

        pred = model.predict(X_te_scaled)
        f1 = f1_score(y_test, pred, average="macro")
        print(f"       {omics_name}: F1-Macro={f1:.4f} (features: {len(cols)})")

    # Soft voting: average probabilities
    avg_proba = np.mean(per_omics_probas, axis=0)
    late_pred = np.argmax(avg_proba, axis=1)

    # Compute metrics
    late_metrics = {
        "accuracy": accuracy_score(y_test, late_pred),
        "precision": precision_score(y_test, late_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, late_pred, average="weighted"),
        "f1_macro": f1_score(y_test, late_pred, average="macro"),
        "mcc": matthews_corrcoef(y_test, late_pred),
    }

    # AUC-ROC
    n_classes = len(np.unique(y))
    if n_classes == 2:
        late_metrics["roc_auc"] = roc_auc_score(y_test, avg_proba[:, 1])
    else:
        late_metrics["roc_auc"] = roc_auc_score(y_test, avg_proba, multi_class="ovr")

    print(f"\n       Late Fusion F1-Macro: {late_metrics['f1_macro']:.4f}")
    print(f"       Late Fusion AUC-ROC: {late_metrics['roc_auc']:.4f}")

    # Confusion matrix for late fusion
    cm = confusion_matrix(y_test, late_pred)

    return late_metrics, late_pred, y_test, cm, per_omics_models


def build_fusion_comparison_table(early_results, late_metrics):
    """
    Build fusion comparison table: Early Fusion vs Late Fusion.
    """
    print_step(34, "Building Fusion Comparison Table")

    rows = []

    # Early Fusion (from Stacking results)
    if early_results:
        rows.append({
            "Fusion Strategy": "Early Fusion (concat -> Stacking)",
            "F1-Macro": f"{early_results['f1_macro'][0]:.4f} ± {early_results['f1_macro'][1]:.4f}",
            "AUC-ROC": f"{early_results['roc_auc'][0]:.4f} ± {early_results['roc_auc'][1]:.4f}",
            "Model": "Stacking (RF + XGB + LightGBM)",
        })

    # Late Fusion
    rows.append({
        "Fusion Strategy": "Late Fusion (per-omics -> soft vote)",
        "F1-Macro": f"{late_metrics['f1_macro']:.4f}",
        "AUC-ROC": f"{late_metrics['roc_auc']:.4f}",
        "Model": "XGBoost x 4 (one per omics)",
    })

    fusion_df = pd.DataFrame(rows)

    # Save
    path = os.path.join(RESULTS_DIR, "fusion_comparison.csv")
    fusion_df.to_csv(path, index=False)
    print(f"       Saved -> {path}")

    print(f"\n       Fusion Comparison:")
    print(f"       {fusion_df.to_string(index=False)}")

    return fusion_df


def run_fusion_comparison(X_final, y, all_results, label_encoder):
    """
    Execute the full fusion comparison (Day 5).
    Returns fusion_df, late metrics, and confusion matrix data.
    """
    # Early Fusion = Stacking result
    early_results = get_early_fusion_results(all_results)

    # Late Fusion
    late_metrics, late_pred, y_test, late_cm, per_omics_models = \
        run_late_fusion(X_final, y, label_encoder)

    # Build comparison table
    fusion_df = build_fusion_comparison_table(early_results, late_metrics)

    return fusion_df, late_metrics, late_pred, y_test, late_cm
