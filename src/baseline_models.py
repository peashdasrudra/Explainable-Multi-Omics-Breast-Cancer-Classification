"""
baseline_models.py — Five Baseline Classifiers with SMOTE-inside-CV
=====================================================================
Evaluates five diverse baseline classifiers using Stratified K-Fold
Cross-Validation with SMOTE applied exclusively inside training folds.

CRITICAL DESIGN DECISION:
    Uses imblearn.pipeline.Pipeline (NOT sklearn.pipeline.Pipeline).
    This ensures that SMOTE oversampling is applied ONLY to training
    folds during cross-validation — preventing synthetic minority
    samples from leaking into validation folds. Using sklearn.Pipeline
    with SMOTE is a widespread methodological error that inflates
    reported performance metrics.

Models:
    1. Logistic Regression (L2, max_iter=1000)
    2. Support Vector Machine (RBF kernel, probability=True)
    3. K-Nearest Neighbors (k=5)
    4. Gaussian Naive Bayes
    5. Random Forest (n_estimators=200)

Metrics:
    Accuracy, Precision (weighted), Recall (weighted), F1-Macro,
    AUC-ROC (one-vs-rest), Matthews Correlation Coefficient (MCC)

References:
    Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling
    Technique. Journal of Artificial Intelligence Research, 16, 321-357.
"""
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, CV_SPLITS, RESULTS_DIR
from src.utils import print_step


def get_baseline_models():
    """Return dict of 5 baseline classifiers."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE
        ),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def get_global_cv():
    """Return the global StratifiedKFold object -- same for ALL experiments."""
    return StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )


def build_smote_pipeline(model):
    """
    Build an imblearn Pipeline with StandardScaler + SMOTE + classifier.
    CRITICAL: imblearn Pipeline handles SMOTE correctly inside CV folds.
    """
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", model),
    ])


def get_scoring():
    """Return scoring metrics dict for cross_validate."""
    return {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1_macro": "f1_macro",
        "roc_auc": "roc_auc_ovr",
        "mcc": make_scorer(matthews_corrcoef),
    }


def evaluate_model_cv(model_name, model, X, y, skf):
    """
    Evaluate a single model using Stratified K-Fold CV with SMOTE inside pipeline.
    Returns dict of metric_name: (mean, std).
    """
    pipeline = build_smote_pipeline(model)
    scoring = get_scoring()

    cv_results = cross_validate(
        pipeline, X, y, cv=skf, scoring=scoring,
        return_train_score=False, n_jobs=-1
    )

    results = {}
    for metric_name in scoring:
        key = f"test_{metric_name}"
        scores = cv_results[key]
        results[metric_name] = (np.mean(scores), np.std(scores))

    return results


def run_baselines(X, y):
    """
    Run all 5 baseline models with SMOTE-inside-CV.
    Returns results dict and DataFrame.
    """
    models = get_baseline_models()
    skf = get_global_cv()
    all_results = {}

    print(f"       Using StratifiedKFold(n_splits={CV_SPLITS}, random_state={RANDOM_STATE})")
    print(f"       Pipeline: StandardScaler -> SMOTE -> Classifier\n")

    for i, (name, model) in enumerate(models.items(), start=1):
        print_step(12 + i, f"Evaluating {name}...")
        results = evaluate_model_cv(name, model, X, y, skf)
        all_results[name] = results

        # Print summary
        f1 = results["f1_macro"]
        auc = results["roc_auc"]
        mcc = results["mcc"]
        print(f"       -> F1-Macro: {f1[0]:.4f}±{f1[1]:.4f}  "
              f"AUC-ROC: {auc[0]:.4f}±{auc[1]:.4f}  "
              f"MCC: {mcc[0]:.4f}±{mcc[1]:.4f}")

    return all_results


def results_to_dataframe(all_results):
    """
    Convert results dict to a clean DataFrame.
    Format: Models as rows, Metrics as columns, values as "mean ± std".
    """
    rows = []
    for model_name, metrics in all_results.items():
        row = {"Model": model_name}
        for metric_name, (mean, std) in metrics.items():
            row[metric_name] = f"{mean:.4f} ± {std:.4f}"
        rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns
    col_order = ["Model", "accuracy", "precision", "recall", "f1_macro", "roc_auc", "mcc"]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    return df


def results_to_numeric_dataframe(all_results):
    """
    Convert results dict to numeric DataFrame (means only) for plotting.
    """
    rows = []
    for model_name, metrics in all_results.items():
        row = {"Model": model_name}
        for metric_name, (mean, std) in metrics.items():
            row[metric_name] = mean
            row[f"{metric_name}_std"] = std
        rows.append(row)

    return pd.DataFrame(rows)


def save_baseline_results(results_df, numeric_df):
    """Save baseline results to CSV."""
    path = os.path.join(RESULTS_DIR, "results_baseline.csv")
    results_df.to_csv(path, index=False)
    print(f"\n  [OK] Saved baseline results to {path}")

    # Also save numeric version for easy plotting
    num_path = os.path.join(RESULTS_DIR, "results_baseline_numeric.csv")
    numeric_df.to_csv(num_path, index=False)

    return path
