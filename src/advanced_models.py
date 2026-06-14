"""
advanced_models.py -- Day 3: XGBoost + LightGBM tuning + Stacking Ensemble.

Hyperparameter tuning via GridSearchCV with SMOTE-inside-CV.
Stacking Ensemble: RF + XGB(best) + LightGBM(best) -> LogisticRegression meta-learner.
"""
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import (
    RANDOM_STATE, XGB_PARAM_GRID, LGBM_PARAM_GRID,
    RESULTS_DIR, MODELS_DIR
)
from src.baseline_models import (
    get_global_cv, build_smote_pipeline, evaluate_model_cv,
    results_to_dataframe, results_to_numeric_dataframe
)
from src.utils import print_step


def tune_xgboost(X, y, skf):
    """
    GridSearchCV for XGBoost with SMOTE-inside-CV pipeline.
    Grid: n_estimators=[100,200], max_depth=[3,5,7], learning_rate=[0.05,0.1]
    """
    print_step(18, "Tuning XGBoost (GridSearchCV)...")

    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", XGBClassifier(
            random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric="mlogloss", use_label_encoder=False,
            verbosity=0
        )),
    ])

    grid = GridSearchCV(
        pipeline, XGB_PARAM_GRID, cv=skf,
        scoring="f1_macro", n_jobs=-1, refit=True
    )
    grid.fit(X, y)

    print(f"       Best params: {grid.best_params_}")
    print(f"       Best F1-Macro: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def tune_lightgbm(X, y, skf):
    """
    GridSearchCV for LightGBM with SMOTE-inside-CV pipeline.
    Same grid structure as XGBoost for fair comparison.
    """
    print_step(19, "Tuning LightGBM (GridSearchCV)...")

    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", LGBMClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )),
    ])

    grid = GridSearchCV(
        pipeline, LGBM_PARAM_GRID, cv=skf,
        scoring="f1_macro", n_jobs=-1, refit=True
    )
    grid.fit(X, y)

    print(f"       Best params: {grid.best_params_}")
    print(f"       Best F1-Macro: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def build_stacking_ensemble(xgb_best_params, lgbm_best_params):
    """
    Build Stacking Ensemble:
    Base: RF(200) + XGB(best) + LightGBM(best)
    Meta: LogisticRegression
    """
    print_step(20, "Building Stacking Ensemble...")

    # Extract classifier params (remove 'clf__' prefix)
    xgb_params = {k.replace("clf__", ""): v for k, v in xgb_best_params.items()}
    lgbm_params = {k.replace("clf__", ""): v for k, v in lgbm_best_params.items()}

    base_learners = [
        ("rf", RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        )),
        ("xgb", XGBClassifier(
            **xgb_params, random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric="mlogloss", use_label_encoder=False, verbosity=0
        )),
        ("lgbm", LGBMClassifier(
            **lgbm_params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )),
    ]

    stacking = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=5,
        n_jobs=-1,
    )

    print(f"       Base learners: RF, XGB({xgb_params}), LightGBM({lgbm_params})")
    print(f"       Meta-learner: LogisticRegression")

    return stacking


def run_advanced_models(X, y, baseline_results):
    """
    Run XGBoost/LightGBM tuning + Stacking Ensemble.
    Merges results with baseline results into full 8-model comparison table.
    Returns combined results dict and best model info.
    """
    skf = get_global_cv()

    # Tune XGBoost
    xgb_best, xgb_params, xgb_score = tune_xgboost(X, y, skf)

    # Tune LightGBM
    lgbm_best, lgbm_params, lgbm_score = tune_lightgbm(X, y, skf)

    # Build and evaluate tuned XGBoost
    print_step(21, "Evaluating tuned XGBoost...")
    xgb_clf = XGBClassifier(
        **{k.replace("clf__", ""): v for k, v in xgb_params.items()},
        random_state=RANDOM_STATE, n_jobs=-1,
        eval_metric="mlogloss", use_label_encoder=False, verbosity=0
    )
    xgb_results = evaluate_model_cv("XGBoost (tuned)", xgb_clf, X, y, skf)
    f1 = xgb_results["f1_macro"]
    print(f"       -> F1-Macro: {f1[0]:.4f}±{f1[1]:.4f}")

    # Build and evaluate tuned LightGBM
    print_step(22, "Evaluating tuned LightGBM...")
    lgbm_clf = LGBMClassifier(
        **{k.replace("clf__", ""): v for k, v in lgbm_params.items()},
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    lgbm_results = evaluate_model_cv("LightGBM (tuned)", lgbm_clf, X, y, skf)
    f1 = lgbm_results["f1_macro"]
    print(f"       -> F1-Macro: {f1[0]:.4f}±{f1[1]:.4f}")

    # Build and evaluate Stacking Ensemble
    stacking = build_stacking_ensemble(xgb_params, lgbm_params)
    print_step(23, "Evaluating Stacking Ensemble...")
    stacking_results = evaluate_model_cv("Stacking Ensemble", stacking, X, y, skf)
    f1 = stacking_results["f1_macro"]
    print(f"       -> F1-Macro: {f1[0]:.4f}±{f1[1]:.4f}")

    # Merge all results
    all_results = dict(baseline_results)  # Copy baseline
    all_results["XGBoost (tuned)"] = xgb_results
    all_results["LightGBM (tuned)"] = lgbm_results
    all_results["Stacking Ensemble"] = stacking_results

    # Find best model
    best_model_name = max(all_results, key=lambda k: all_results[k]["f1_macro"][0])
    best_f1 = all_results[best_model_name]["f1_macro"][0]
    print(f"\n  [OK] Best model: {best_model_name} (F1-Macro: {best_f1:.4f})")

    # Check stacking vs best individual
    stacking_f1 = stacking_results["f1_macro"][0]
    individual_best = max(
        (v["f1_macro"][0] for k, v in all_results.items() if k != "Stacking Ensemble"),
    )
    if stacking_f1 >= individual_best:
        print(f"  [OK] Stacking outperforms all individual models")
    else:
        print(f"  [!!] Stacking ({stacking_f1:.4f}) < best individual ({individual_best:.4f})")

    # Save all models
    models_to_save = {
        "xgb_tuned": xgb_clf,
        "lgbm_tuned": lgbm_clf,
        "stacking": stacking,
        "xgb_best_params": xgb_params,
        "lgbm_best_params": lgbm_params,
    }

    for name, model in models_to_save.items():
        path = os.path.join(MODELS_DIR, f"{name}.joblib")
        joblib.dump(model, path)

    print(f"  [OK] Saved all models to {MODELS_DIR}")

    # Build and save full 8-model comparison table
    results_df = results_to_dataframe(all_results)
    numeric_df = results_to_numeric_dataframe(all_results)

    results_df.to_csv(os.path.join(RESULTS_DIR, "results_all_models.csv"), index=False)
    numeric_df.to_csv(os.path.join(RESULTS_DIR, "results_all_models_numeric.csv"), index=False)
    print(f"  [OK] Saved full 8-model comparison table")

    return all_results, best_model_name, xgb_params, lgbm_params
