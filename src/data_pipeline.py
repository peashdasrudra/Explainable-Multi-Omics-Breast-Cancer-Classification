"""
data_pipeline.py — TCGA BRCA Data Loading, Cleaning, and Preparation
======================================================================
Loads the TCGA BRCA multi-omics dataset, performs content-based column
deduplication, handles missing values, encodes histological subtype labels,
and prepares the dataset for the 3-stage feature selection pipeline.

Dataset:
    TCGA Breast Invasive Carcinoma (BRCA) cohort with 4 omics layers:
    - mRNA Expression (rs_*): 604 features
    - Copy Number Variation (cn_*): 761 features
    - DNA Methylation (mu_*): 249 features
    - Protein/RPPA (pp_*): 223 features
    Total: 705 patients × 1,837 unique features (after deduplication)

Target:
    Histological subtype — binary classification:
    - Class 0: Infiltrating Ductal Carcinoma (IDC) — 574 samples (81.4%)
    - Class 1: Infiltrating Lobular Carcinoma (ILC) — 131 samples (18.6%)
    - Imbalance ratio: 4.4:1 (addressed via SMOTE-inside-CV)

Reference:
    TCGA Network (2012). Comprehensive molecular portraits of human
    breast tumours. Nature, 490(7418), 61-70.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.config import (
    DATA_PATH, CLINICAL_COLS, PRIMARY_TARGET,
    OMICS_PREFIX_MAP, OMICS_SHORT_NAMES, PREPROCESSED_DIR
)
from src.utils import print_step
import os


def load_raw_data():
    """Step 1-2: Load the raw TCGA BRCA multi-omics CSV."""
    print_step(1, f"Loading raw data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"       Raw dataset shape: {df.shape}")
    return df


def deduplicate_columns(df):
    """
    Step 3: Remove columns with identical content (not just names).
    Pandas auto-renames duplicate column names on CSV load (.1, .2 suffixes),
    but ~99 cn_ columns have identical values because they represent genes at
    the same chromosomal locus with identical copy-number profiles.
    Expected: 1,936 -> 1,837 unique features.
    """
    print_step(2, "Content-based column deduplication")
    shape_before = df.shape[1]
    df_dedup = df.T.drop_duplicates().T
    removed = shape_before - df_dedup.shape[1]
    print(f"       Removed {removed} content-duplicate columns: {shape_before} -> {df_dedup.shape[1]}")
    return df_dedup


def separate_features_and_target(df, target_name=PRIMARY_TARGET):
    """
    Step 4-5: Separate features from clinical targets.
    Drops rows where the target is NaN and removes clinical columns from features.
    """
    print_step(3, f"Separating features and target: '{target_name}'")

    # Drop rows where target is missing
    df_clean = df.dropna(subset=[target_name]).copy()
    dropped = len(df) - len(df_clean)
    if dropped > 0:
        print(f"       Dropped {dropped} rows with missing target values")

    # Separate feature columns (exclude all clinical)
    feature_cols = [c for c in df_clean.columns if c not in CLINICAL_COLS]
    X = df_clean[feature_cols].copy()
    y = df_clean[target_name].copy()

    print(f"       Features shape: {X.shape}, Target samples: {len(y)}")
    return X, y, feature_cols


def identify_omics_groups(feature_cols):
    """
    Step 4 (continued): Group features by their omics layer prefix.
    Returns a dict mapping omics prefix -> list of column names.
    """
    print_step(4, "Identifying omics layer groups")
    omics_groups = {}
    for prefix, name in OMICS_SHORT_NAMES.items():
        cols = [c for c in feature_cols if c.startswith(prefix)]
        omics_groups[prefix] = cols
        print(f"       {name} ({prefix}*): {len(cols)} features")

    # Check for ungrouped features
    grouped = sum(len(v) for v in omics_groups.values())
    ungrouped = len(feature_cols) - grouped
    if ungrouped > 0:
        print(f"       [!] {ungrouped} features not matched to any omics prefix")

    return omics_groups


def handle_missing_values(X, threshold=0.20):
    """
    Step 6: Handle missing values.
    - Drop features with >20% missing values
    - Impute remaining with median
    """
    print_step(5, f"Handling missing values (threshold={threshold*100:.0f}%)")

    # Check missing rates
    missing_rate = X.isnull().mean()
    high_missing = missing_rate[missing_rate > threshold]

    if len(high_missing) > 0:
        print(f"       Dropping {len(high_missing)} features with >{threshold*100:.0f}% missing")
        X = X.drop(columns=high_missing.index)

    # Impute remaining with median
    remaining_missing = X.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"       Imputing {remaining_missing} remaining missing values with median")
        X = X.fillna(X.median())
    else:
        print(f"       No missing values remaining -- dataset is clean")

    return X


def encode_labels(y):
    """
    Step 7: Encode categorical labels to numeric using LabelEncoder.
    Returns encoded labels, the encoder, and a mapping dict.
    """
    print_step(6, "Encoding labels")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"       Label mapping: {label_map}")

    return y_encoded, le, label_map


def check_class_distribution(y, le):
    """
    Step 8: Check and report class distribution and imbalance ratio.
    """
    print_step(7, "Class distribution check")

    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / min_count

    for cls_idx, count in zip(unique, counts):
        cls_name = le.inverse_transform([cls_idx])[0]
        pct = count / len(y) * 100
        print(f"       Class {cls_idx} ({cls_name}): {count} samples ({pct:.1f}%)")

    print(f"       Imbalance ratio: {imbalance_ratio:.1f}:1")

    return dict(zip(unique, counts))


def verify_data(X, y, expected_features=1837):
    """
    Step 9: Verify dataset shape and integrity.
    """
    print_step(8, "Data verification")
    print(f"       Samples: {X.shape[0]}")
    print(f"       Features: {X.shape[1]} (expected ~{expected_features})")
    print(f"       Target classes: {len(np.unique(y))}")
    print(f"       Any NaN in features: {X.isnull().any().any() if hasattr(X, 'isnull') else 'N/A (numpy)'}")
    return True


def run_data_pipeline():
    """
    Execute the full Day 1 data pipeline.
    Returns X (features DataFrame), y (encoded labels), label_encoder, omics_groups.
    """
    # Load and deduplicate
    df = load_raw_data()
    df = deduplicate_columns(df)

    # Separate features and target
    X, y_raw, feature_cols = separate_features_and_target(df)

    # Identify omics groups
    omics_groups = identify_omics_groups(feature_cols)

    # Handle missing values
    X = handle_missing_values(X)

    # Update feature_cols after potential drops
    feature_cols = list(X.columns)

    # Update omics groups after potential feature drops
    omics_groups = identify_omics_groups(feature_cols)

    # Encode labels
    y, label_encoder, label_map = encode_labels(y_raw)

    # Check distribution
    class_dist = check_class_distribution(y, label_encoder)

    # Verify
    verify_data(X, y)

    # Save omics group membership for SHAP analysis later
    omics_membership = []
    for prefix, cols in omics_groups.items():
        short_name = OMICS_SHORT_NAMES[prefix]
        for col in cols:
            omics_membership.append({"feature": col, "omics_group": short_name, "prefix": prefix})

    omics_df = pd.DataFrame(omics_membership)
    omics_path = os.path.join(PREPROCESSED_DIR, "omics_membership.csv")
    omics_df.to_csv(omics_path, index=False)
    print(f"\n  [OK] Saved omics membership to {omics_path}")

    return X, y, label_encoder, omics_groups, class_dist


if __name__ == "__main__":
    from src.utils import set_all_seeds
    set_all_seeds()
    X, y, le, omics_groups, class_dist = run_data_pipeline()
    print(f"\n  [DONE] Data pipeline complete. X={X.shape}, y={y.shape}")
