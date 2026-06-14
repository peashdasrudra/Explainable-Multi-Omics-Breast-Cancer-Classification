"""
feature_selection.py -- Day 1-2: Three-stage consensus feature selection funnel.

Stage 1: Variance Threshold (remove near-zero variance features)
Stage 2: ANOVA F-test + Mutual Information per omics group (union selection)
Stage 3: RF + XGB consensus importance ranking (keep top-N)

This multi-source agreement eliminates single-method bias -- a documented
gap in feature selection literature.
"""
import numpy as np
import pandas as pd
import os
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import (
    RANDOM_STATE, VARIANCE_THRESHOLD, STAGE2_K_PER_OMICS,
    STAGE3_TOP_N, OMICS_SHORT_NAMES, PREPROCESSED_DIR, RESULTS_DIR
)
from src.utils import print_step


def stage1_variance_filter(X, threshold=VARIANCE_THRESHOLD):
    """
    Stage 1 -- Variance Threshold Filter.
    Removes features with variance below the threshold.
    Expected: 1837 -> ~1400 features.
    """
    print_step(9, f"Stage 1: Variance Threshold (threshold={threshold})")

    n_before = X.shape[1]
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)

    # Get the mask of selected features
    mask = selector.get_support()
    selected_cols = X.columns[mask].tolist()

    n_after = len(selected_cols)
    print(f"       Features: {n_before} -> {n_after} (removed {n_before - n_after})")

    # Return as DataFrame with column names preserved
    X_stage1 = pd.DataFrame(X_filtered, columns=selected_cols, index=X.index)
    return X_stage1, selected_cols


def stage2_anova_mi_filter(X, y, omics_groups, k_per_omics=STAGE2_K_PER_OMICS):
    """
    Stage 2 -- ANOVA F-test + Mutual Information per omics group.
    For each omics group:
      - Select top-K features by ANOVA F-test
      - Select top-K features by Mutual Information
      - Keep the UNION (OR logic, not AND)
    Expected: ~1400 -> ~300 features.
    """
    print_step(10, f"Stage 2: ANOVA + MI filter (k={k_per_omics} per omics)")

    all_selected = []

    for prefix, name in OMICS_SHORT_NAMES.items():
        # Get columns for this omics group that survived Stage 1
        omics_cols = [c for c in X.columns if c.startswith(prefix)]
        n_omics = len(omics_cols)

        if n_omics == 0:
            print(f"       {name}: 0 features -- skipping")
            continue

        X_omics = X[omics_cols].values

        # Adjust k if fewer features than k_per_omics
        k = min(k_per_omics, n_omics)

        # ANOVA F-test
        anova_selector = SelectKBest(f_classif, k=k)
        anova_selector.fit(X_omics, y)
        anova_mask = anova_selector.get_support()
        anova_selected = set(np.array(omics_cols)[anova_mask])

        # Mutual Information
        mi_selector = SelectKBest(mutual_info_classif, k=k)
        mi_selector.fit(X_omics, y)
        mi_mask = mi_selector.get_support()
        mi_selected = set(np.array(omics_cols)[mi_mask])

        # UNION of both methods (OR logic)
        union_selected = anova_selected | mi_selected
        all_selected.extend(list(union_selected))

        print(f"       {name}: {n_omics} -> ANOVA={len(anova_selected)}, "
              f"MI={len(mi_selected)}, Union={len(union_selected)}")

    # Build Stage 2 DataFrame
    X_stage2 = X[all_selected].copy()
    print(f"       Total after Stage 2: {X.shape[1]} -> {X_stage2.shape[1]}")

    return X_stage2, all_selected


def stage3_consensus_ranking(X, y, top_n=STAGE3_TOP_N):
    """
    Stage 3 -- RF + XGB Consensus Importance Ranking.
    Train both models, average their feature importances, keep top-N.
    Expected: ~300 -> 75 features.
    """
    print_step(11, f"Stage 3: RF + XGB consensus ranking (top-{top_n})")

    feature_names = X.columns.tolist()
    X_arr = X.values

    # Train Random Forest
    print(f"       Training RandomForest(n_estimators=200)...")
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_arr, y)
    rf_importances = rf.feature_importances_

    # Train XGBoost
    print(f"       Training XGBClassifier(n_estimators=200)...")
    xgb = XGBClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
        eval_metric="mlogloss", use_label_encoder=False
    )
    xgb.fit(X_arr, y)
    xgb_importances = xgb.feature_importances_

    # Average importances (consensus)
    avg_importances = (rf_importances + xgb_importances) / 2

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "rf_importance": rf_importances,
        "xgb_importance": xgb_importances,
        "avg_importance": avg_importances,
    }).sort_values("avg_importance", ascending=False)

    # Keep top-N
    top_features = importance_df.head(top_n)["feature"].tolist()
    X_final = X[top_features].copy()

    print(f"       Features: {X.shape[1]} -> {top_n}")
    print(f"       Top 5 consensus features: {top_features[:5]}")

    return X_final, top_features, importance_df


def run_feature_selection(X, y, omics_groups):
    """
    Execute the full 3-stage feature selection funnel.
    Returns X_final, final feature list, funnel numbers, importance_df.
    """
    n_original = X.shape[1]

    # Stage 1: Variance Threshold
    X_s1, s1_features = stage1_variance_filter(X)
    n_stage1 = X_s1.shape[1]

    # Stage 2: ANOVA + MI per omics
    X_s2, s2_features = stage2_anova_mi_filter(X_s1, y, omics_groups)
    n_stage2 = X_s2.shape[1]

    # Stage 3: RF + XGB Consensus
    X_final, final_features, importance_df = stage3_consensus_ranking(X_s2, y)
    n_stage3 = X_final.shape[1]

    # Record funnel
    funnel = {
        "Original": n_original,
        "Stage 1 (Variance)": n_stage1,
        "Stage 2 (ANOVA+MI)": n_stage2,
        "Stage 3 (Consensus)": n_stage3,
    }

    print(f"\n  [OK] Feature Selection Funnel:")
    print(f"       {n_original} -> {n_stage1} -> {n_stage2} -> {n_stage3}")

    # Save Stage 2 feature list with omics membership
    stage2_info = []
    for feat in s2_features:
        omics = "unknown"
        for prefix, name in OMICS_SHORT_NAMES.items():
            if feat.startswith(prefix):
                omics = name
                break
        stage2_info.append({"feature": feat, "omics_group": omics})

    s2_df = pd.DataFrame(stage2_info)
    s2_path = os.path.join(PREPROCESSED_DIR, "feature_stage2.csv")
    s2_df.to_csv(s2_path, index=False)

    # Save final feature list with omics membership
    final_info = []
    for feat in final_features:
        omics = "unknown"
        for prefix, name in OMICS_SHORT_NAMES.items():
            if feat.startswith(prefix):
                omics = name
                break
        final_info.append({"feature": feat, "omics_group": omics})

    final_df = pd.DataFrame(final_info)
    final_path = os.path.join(PREPROCESSED_DIR, "feature_final.csv")
    final_df.to_csv(final_path, index=False)

    # Save importance rankings
    imp_path = os.path.join(RESULTS_DIR, "consensus_importances.csv")
    importance_df.to_csv(imp_path, index=False)

    # Save X_final and feature names for SHAP
    np.save(os.path.join(PREPROCESSED_DIR, "X_final.npy"), X_final.values)
    np.save(os.path.join(PREPROCESSED_DIR, "feature_names.npy"), np.array(final_features))

    print(f"  [OK] Saved feature lists and importance rankings")

    return X_final, final_features, funnel, importance_df
