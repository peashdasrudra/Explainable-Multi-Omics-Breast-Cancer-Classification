# Multi-Omics Cancer Classification using CRO and xAI

This repository contains the code and resources for a thesis on **Multi-Omics Cancer Classification using Distributed Chemical Reaction Optimization (DC-CRO) and Explainable AI (xAI)**.

## 📌 Project Overview
The objective of this research is to classify breast cancer (BRCA) subtypes using multi-omics data (RNA-Seq, Copy Number Variation, Mutation, and Protein expression) and optimize the classification process using Chemical Reaction Optimization (CRO) for feature selection.

## 🚀 Current Progress: Week 1 Completed
The baseline pipeline has been established with the following achievements:
- **Data Deduplication:** Handled high-dimensional feature spaces.
- **Target Preparation:** Prepared 3 clinical targets (`ER.Status`, `HER2.Final.Status`, and `histological.type`).
- **Data Leakage Prevention:** Used proper pipelines to ensure SMOTE is applied only within cross-validation folds.
- **Baseline Models:** Evaluated Random Forest, SVM, and XGBoost using Stratified 5-Fold Cross-Validation.

### 📊 Baseline Results Summary
XGBoost emerged as the best-performing model across all targets. The full results are saved in the `outputs/baseline_metrics/` directory.

## 📁 Project Structure
```text
├── outputs/
│   ├── baseline_metrics/     # CSV files with pivoted evaluation metrics
│   └── preprocessed/         # Cleaned data and .npy arrays for future use
├── src/
│   ├── baseline.py           # Cross-validation and model training logic
│   ├── preprocess.py         # Data loading, cleaning, and SMOTE handling
│   └── utils.py              # Reproducibility settings (random seeds)
├── data/
│   └── brca_data_w_subtypes.csv  # Raw Multi-Omics dataset
├── requirements.txt          # Python dependencies
└── run_week1.py              # Master script to run the Week 1 pipeline
```

## 🛠️ How to Run
1. **Install Dependencies:**
   ```bash
   python -m pip install -r requirements.txt
   ```
2. **Execute Pipeline:**
   ```bash
   python run_week1.py
   ```

## 🔮 Next Steps (Week 2)
- Implementation of the **Distributed Chemical Reaction Optimization (DC-CRO)** skeleton.
- Feature selection using CRO to improve the performance of weaker models on imbalanced targets.
