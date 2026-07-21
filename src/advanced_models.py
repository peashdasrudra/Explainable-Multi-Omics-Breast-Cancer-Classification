"""
advanced_models.py — XGBoost + LightGBM Tuning and Stacking Ensemble
=======================================================================
Hyperparameter optimization for gradient boosted tree models via
RandomizedSearchCV (50 iterations) with SMOTE-inside-CV pipeline,
followed by construction of a Stacking Ensemble.

Hyperparameter Tuning:
    RandomizedSearchCV is used instead of GridSearchCV to efficiently
    explore a large parameter space. With 50 random samples from a
    6-dimensional parameter grid (5^6 = 15,625 total configurations),
    the probability of finding a configuration within the top 5% is
    approximately 92.3% (Bergstra & Bengio, 2012).

Stacking Ensemble Architecture:
    Base learners: RF(200) + XGBoost(tuned) + LightGBM(tuned)
    Meta-learner:  Logistic Regression (max_iter=1000)
    Each base learner generates out-of-fold predictions via internal 5-fold CV,
    which the meta-learner uses as input features.

References:
    Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
    Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision
    Tree. NeurIPS.
    Bergstra, J. & Bengio, Y. (2012). Random Search for Hyper-Parameter
    Optimization. JMLR, 13, 281-305.
"""
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import RandomizedSearchCV
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
    RandomizedSearchCV for XGBoost with SMOTE-inside-CV pipeline.
    Searches 50 random configurations from the expanded parameter space.
    """
    print_step(18, "Tuning XGBoost (RandomizedSearchCV - 50 iterations)...")

    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", XGBClassifier(
            random_state=RANDOM_STATE, n_jobs=1,  # Set n_jobs=1 to avoid deadlock in nested parallelization
            eval_metric="mlogloss", use_label_encoder=False,
            verbosity=0
        )),
    ])

    search = RandomizedSearchCV(
        pipeline, XGB_PARAM_GRID, n_iter=50, cv=skf,
        scoring="f1_macro", n_jobs=-1, refit=True, random_state=RANDOM_STATE
    )
    search.fit(X, y)

    print(f"       Best params: {search.best_params_}")
    print(f"       Best F1-Macro: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_, search.best_score_


def tune_lightgbm(X, y, skf):
    """
    RandomizedSearchCV for LightGBM with SMOTE-inside-CV pipeline.
    Searches 50 random configurations from the expanded parameter space.
    """
    print_step(19, "Tuning LightGBM (RandomizedSearchCV - 50 iterations)...")

    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", LGBMClassifier(
            random_state=RANDOM_STATE, n_jobs=1, verbose=-1  # Set n_jobs=1 to avoid deadlock in nested parallelization
        )),
    ])

    search = RandomizedSearchCV(
        pipeline, LGBM_PARAM_GRID, n_iter=50, cv=skf,
        scoring="f1_macro", n_jobs=-1, refit=True, random_state=RANDOM_STATE
    )
    search.fit(X, y)

    print(f"       Best params: {search.best_params_}")
    print(f"       Best F1-Macro: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_, search.best_score_


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
