"""
config.py -- Global configuration for the thesis pipeline.

Explainable Multi-Omics Breast Cancer Classification Using
Consensus Feature Selection, Ensemble Learning & Cross-Omics SHAP Attribution
on TCGA BRCA.
"""
import os

# ─────────────────────────── Reproducibility ───────────────────────────
RANDOM_STATE = 42
CV_SPLITS = 5

# ─────────────────────────── Paths ─────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "brca_data_w_subtypes.csv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PREPROCESSED_DIR = os.path.join(OUTPUT_DIR, "preprocessed")

# Create output directories
for d in [FIGURES_DIR, RESULTS_DIR, MODELS_DIR, PREPROCESSED_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────── Dataset ───────────────────────────────────
# Primary target for classification
PRIMARY_TARGET = "histological.type"

# All clinical target columns (excluded from features)
CLINICAL_COLS = [
    "ER.Status", "HER2.Final.Status", "histological.type",
    "vital.status", "PR.Status"
]

# ─────────────────────────── Omics Layer Mapping ───────────────────────
# Column prefix -> human-readable omics layer name
OMICS_PREFIX_MAP = {
    "rs_": "mRNA Expression",
    "cn_": "Copy Number Variation",
    "mu_": "DNA Methylation",
    "pp_": "Protein (RPPA)",
}

# Short names for plotting
OMICS_SHORT_NAMES = {
    "rs_": "mRNA",
    "cn_": "CNV",
    "mu_": "Methylation",
    "pp_": "Protein",
}

# ─────────────────────────── Feature Selection Parameters ──────────────
# Stage 1: Variance Threshold
VARIANCE_THRESHOLD = 0.01

# Stage 2: ANOVA + MI -- select top-K per omics group
STAGE2_K_PER_OMICS = 75

# Stage 3: RF + XGB Consensus -- keep top-N overall
STAGE3_TOP_N = 75

# ─────────────────────────── Model Hyperparameters ─────────────────────
# XGBoost grid search
XGB_PARAM_GRID = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [3, 5, 7],
    "clf__learning_rate": [0.05, 0.1],
}

# LightGBM grid search
LGBM_PARAM_GRID = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [3, 5, 7],
    "clf__learning_rate": [0.05, 0.1],
}

# ─────────────────────────── Visualization ─────────────────────────────
FIGURE_DPI = 150
FIGURE_FORMAT = "png"

# Color palette for omics layers
OMICS_COLORS = {
    "mRNA": "#E74C3C",
    "CNV": "#3498DB",
    "Methylation": "#2ECC71",
    "Protein": "#F39C12",
}

# Color palette for models
MODEL_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#34495E",
]
