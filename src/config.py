"""
config.py — Global Configuration
=================================
Centralized configuration for all pipeline parameters, file paths,
hyperparameter grids, and visualization settings.

Design Rationale:
    All tunable parameters are defined here to ensure reproducibility
    and to provide a single point of configuration for the entire
    pipeline. Changing a parameter here propagates to all modules.

Sections:
    1. Reproducibility settings (random seed, CV splits)
    2. File paths (data, outputs)
    3. Dataset configuration (target column, clinical columns)
    4. Omics layer mapping (prefix → name)
    5. Feature selection parameters (per-stage thresholds)
    6. Model hyperparameter grids (XGBoost, LightGBM)
    7. Visualization settings (DPI, color palettes)
"""
import os

# ═══════════════════════════════════════════════════════════════════
# 1. REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════
# All random seeds across the pipeline are locked to this value.
# This guarantees that every run produces identical results.
RANDOM_STATE = 42

# Number of folds for Stratified K-Fold Cross-Validation.
# 5 folds is standard for biomedical datasets of this size (~700 samples).
CV_SPLITS = 5

# ═══════════════════════════════════════════════════════════════════
# 2. FILE PATHS
# ═══════════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "brca_data_w_subtypes.csv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PREPROCESSED_DIR = os.path.join(OUTPUT_DIR, "preprocessed")

# Create output directories if they do not exist
for d in [FIGURES_DIR, RESULTS_DIR, MODELS_DIR, PREPROCESSED_DIR]:
    os.makedirs(d, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# 3. DATASET CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
# Primary target for binary classification: IDC vs ILC
PRIMARY_TARGET = "histological.type"

# Clinical columns to exclude from features (these are targets/labels,
# not molecular measurements)
CLINICAL_COLS = [
    "ER.Status", "HER2.Final.Status", "histological.type",
    "vital.status", "PR.Status"
]

# ═══════════════════════════════════════════════════════════════════
# 4. OMICS LAYER MAPPING
# ═══════════════════════════════════════════════════════════════════
# Column prefix → full omics layer name (for academic outputs)
OMICS_PREFIX_MAP = {
    "rs_": "mRNA Expression",
    "cn_": "Copy Number Variation",
    "mu_": "DNA Methylation",
    "pp_": "Protein (RPPA)",
}

# Column prefix → short name (for figures and concise tables)
OMICS_SHORT_NAMES = {
    "rs_": "mRNA",
    "cn_": "CNV",
    "mu_": "Methylation",
    "pp_": "Protein",
}

# ═══════════════════════════════════════════════════════════════════
# 5. FEATURE SELECTION PARAMETERS
# ═══════════════════════════════════════════════════════════════════
# Stage 1: Variance Threshold — removes features with near-zero variance.
# Threshold of 0.01 removes constant/near-constant features that carry
# no discriminative signal.
VARIANCE_THRESHOLD = 0.01

# Stage 2: ANOVA F-test + Mutual Information — select top-K features
# per omics group using the UNION of both methods. K=75 per omics
# retains sufficient diversity while removing clearly irrelevant features.
STAGE2_K_PER_OMICS = 75

# Stage 3: RF + XGBoost Consensus — keep top-N features by averaged
# tree-based importance scores. N=75 produces a compact but expressive
# feature set that balances model performance with interpretability.
STAGE3_TOP_N = 75

# ═══════════════════════════════════════════════════════════════════
# 6. MODEL HYPERPARAMETER GRIDS
# ═══════════════════════════════════════════════════════════════════
# XGBoost parameter distributions for RandomizedSearchCV.
# Search space covers depth, learning rate, regularization, and
# stochastic gradient boosting parameters.
XGB_PARAM_GRID = {
    "clf__n_estimators": [100, 150, 200, 250, 300],
    "clf__max_depth": [3, 4, 5, 6, 7, 8],
    "clf__learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
    "clf__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "clf__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "clf__min_child_weight": [1, 2, 3, 5, 7],
}

# LightGBM parameter distributions for RandomizedSearchCV.
LGBM_PARAM_GRID = {
    "clf__n_estimators": [100, 150, 200, 250, 300],
    "clf__max_depth": [3, 4, 5, 6, 7, 8],
    "clf__learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
    "clf__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "clf__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "clf__min_child_weight": [1, 2, 3, 5, 7],
}

# ═══════════════════════════════════════════════════════════════════
# 7. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════
# DPI for saved figures. 300 DPI is the standard for publication-quality
# figures suitable for journal submission and thesis printing.
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

# Color palette for omics layers — used consistently across all figures
OMICS_COLORS = {
    "mRNA": "#E74C3C",        # Red
    "CNV": "#3498DB",         # Blue
    "Methylation": "#2ECC71", # Green
    "Protein": "#F39C12",     # Orange
}

# Color palette for models — used in comparison charts
MODEL_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#34495E",
]
