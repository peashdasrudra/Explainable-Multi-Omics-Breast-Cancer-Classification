"""
shap_stability.py -- SHAP Ranking Stability Analysis.

Addresses reviewer criticism: "SHAP values come from a single split.
How stable are the feature rankings?"

This module runs SHAP on multiple random splits and measures:
1. Jaccard similarity of top-20 feature sets across splits
2. Spearman rank correlation of SHAP importance rankings
3. Whether E-Cadherin remains #1 across all splits

Generates:
    fig_23_shap_stability.png -- Jaccard + rank correlation summary
    shap_stability_results.csv -- Per-split top-20 rankings
"""
import numpy as np
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    RANDOM_STATE, OMICS_SHORT_NAMES,
    FIGURES_DIR, RESULTS_DIR, FIGURE_DPI
)
from src.utils import set_all_seeds, print_section, print_step


def run_shap_stability(X_final, y, n_splits=5, top_k=20):
    """
    Run SHAP on n_splits different random train/test splits.
    For each split: train XGBoost, compute SHAP, record top-K rankings.
    Returns stability metrics.
    """
    print_section("SHAP STABILITY ANALYSIS")

    feature_names = list(X_final.columns) if hasattr(X_final, 'columns') else \
        [f"f_{i}" for i in range(X_final.shape[1])]
    X_arr = X_final.values if hasattr(X_final, 'values') else X_final

    all_rankings = []       # List of ordered feature lists per split
    all_top_k_sets = []     # Sets of top-K features per split
    ecad_ranks = []         # E-Cadherin rank per split
    all_mean_abs_shap = []  # Mean |SHAP| per feature per split

    for split_idx in range(n_splits):
        seed = RANDOM_STATE + split_idx * 7  # Different seed per split
        print_step(split_idx + 1, f"Split {split_idx + 1}/{n_splits} (seed={seed})")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y, test_size=0.2, stratify=y, random_state=seed
        )

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train XGBoost
        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric="mlogloss", use_label_encoder=False, verbosity=0
        )
        model.fit(X_train_s, y_train)

        # SHAP
        explainer = shap.TreeExplainer(model)
        X_test_df = pd.DataFrame(X_test_s, columns=feature_names)
        shap_values = explainer.shap_values(X_test_df)

        # For binary: shap_values is 2D (positive class) or list of 2D
        if isinstance(shap_values, list):
            mean_abs = np.abs(np.stack(shap_values, axis=-1)).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)

        # Rank features by mean |SHAP|
        ranking_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        ranking_df["rank"] = range(1, len(ranking_df) + 1)
        ranked_features = ranking_df["feature"].tolist()

        all_rankings.append(ranked_features)
        all_top_k_sets.append(set(ranked_features[:top_k]))
        all_mean_abs_shap.append(mean_abs)

        # E-Cadherin rank
        ecad_rank = None
        for r, f in enumerate(ranked_features, 1):
            if "E.Cadherin" in f or "E-Cadherin" in f:
                ecad_rank = r
                break
        ecad_ranks.append(ecad_rank)

        print(f"       Top-5: {ranked_features[:5]}")
        print(f"       E-Cadherin rank: #{ecad_rank if ecad_rank else 'NOT FOUND'}")

    # Compute pairwise Jaccard similarity of top-K sets
    n = len(all_top_k_sets)
    jaccard_values = []
    for i in range(n):
        for j in range(i + 1, n):
            intersection = len(all_top_k_sets[i] & all_top_k_sets[j])
            union = len(all_top_k_sets[i] | all_top_k_sets[j])
            jaccard = intersection / union if union > 0 else 0
            jaccard_values.append(jaccard)

    mean_jaccard = np.mean(jaccard_values)

    # Compute pairwise Spearman rank correlation (on mean |SHAP| vectors)
    spearman_values = []
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(all_mean_abs_shap[i], all_mean_abs_shap[j])
            spearman_values.append(rho)

    mean_spearman = np.mean(spearman_values)

    # Summary
    print(f"\n  [SHAP STABILITY RESULTS]")
    print(f"  Mean Jaccard Similarity (top-{top_k}): {mean_jaccard:.4f}")
    print(f"  Mean Spearman Rank Correlation:        {mean_spearman:.4f}")
    print(f"  E-Cadherin ranks across splits:        {ecad_ranks}")
    ecad_top1 = sum(1 for r in ecad_ranks if r is not None and r == 1)
    ecad_top5 = sum(1 for r in ecad_ranks if r is not None and r <= 5)
    print(f"  E-Cadherin as #1: {ecad_top1}/{n_splits} splits")
    print(f"  E-Cadherin in top-5: {ecad_top5}/{n_splits} splits")

    # Save per-split top-20 rankings
    stability_data = {}
    for split_idx, ranking in enumerate(all_rankings):
        stability_data[f"Split_{split_idx + 1}"] = ranking[:top_k]
    stability_df = pd.DataFrame(stability_data)
    stability_df.index = [f"Rank_{i+1}" for i in range(top_k)]
    stability_path = os.path.join(RESULTS_DIR, "shap_stability_results.csv")
    stability_df.to_csv(stability_path)
    print(f"  Saved -> {stability_path}")

    # Save summary metrics
    summary = {
        "metric": ["mean_jaccard_top20", "mean_spearman_rho",
                    "ecad_as_rank1", "ecad_in_top5", "n_splits"],
        "value": [mean_jaccard, mean_spearman,
                  f"{ecad_top1}/{n_splits}", f"{ecad_top5}/{n_splits}", n_splits]
    }
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(RESULTS_DIR, "shap_stability_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    return {
        "mean_jaccard": mean_jaccard,
        "mean_spearman": mean_spearman,
        "jaccard_values": jaccard_values,
        "spearman_values": spearman_values,
        "ecad_ranks": ecad_ranks,
        "all_rankings": all_rankings,
        "top_k": top_k,
        "n_splits": n_splits,
    }


def plot_shap_stability(results):
    """
    Generate a combined stability figure showing:
    Left: Jaccard similarity distribution
    Right: E-Cadherin rank across splits
    """
    print_step("FIG", "Generating SHAP stability plot (fig_23)...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    top_k = results["top_k"]
    n_splits = results["n_splits"]

    # Panel 1: Jaccard similarity
    ax = axes[0]
    jaccard_vals = results["jaccard_values"]
    ax.bar(range(len(jaccard_vals)), jaccard_vals, color="#3498DB",
           edgecolor="white", alpha=0.8)
    ax.axhline(y=results["mean_jaccard"], color="#E74C3C", linestyle="--",
               linewidth=2, label=f"Mean = {results['mean_jaccard']:.3f}")
    ax.set_xlabel("Split Pair", fontsize=11)
    ax.set_ylabel(f"Jaccard Similarity (top-{top_k})", fontsize=11)
    ax.set_title(f"Top-{top_k} Feature Set\nOverlap Across Splits", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: Spearman rank correlation
    ax = axes[1]
    spearman_vals = results["spearman_values"]
    ax.bar(range(len(spearman_vals)), spearman_vals, color="#2ECC71",
           edgecolor="white", alpha=0.8)
    ax.axhline(y=results["mean_spearman"], color="#E74C3C", linestyle="--",
               linewidth=2, label=f"Mean = {results['mean_spearman']:.3f}")
    ax.set_xlabel("Split Pair", fontsize=11)
    ax.set_ylabel("Spearman Rank Correlation", fontsize=11)
    ax.set_title("SHAP Importance\nRank Correlation", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 3: E-Cadherin rank across splits
    ax = axes[2]
    ecad_ranks = results["ecad_ranks"]
    valid_ranks = [r if r is not None else top_k + 5 for r in ecad_ranks]
    colors = ["#2ECC71" if r is not None and r == 1 else "#3498DB" if r is not None and r <= 5
              else "#F39C12" for r in ecad_ranks]
    ax.bar(range(1, n_splits + 1), valid_ranks, color=colors, edgecolor="white")
    for i, rank in enumerate(ecad_ranks):
        label = f"#{rank}" if rank is not None else "N/A"
        ax.text(i + 1, valid_ranks[i] + 0.3, label, ha="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Split", fontsize=11)
    ax.set_ylabel("E-Cadherin Rank", fontsize=11)
    ax.set_title("E-Cadherin Rank\nStability Across Splits", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.set_ylim(max(valid_ranks) + 2, 0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle("SHAP Feature Ranking Stability Analysis",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig_23_shap_stability.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {path}")


# =============================================
# MAIN EXECUTION
# =============================================
def main():
    """Run SHAP stability analysis as a standalone script."""
    set_all_seeds(42)

    from src.data_pipeline import run_data_pipeline
    from src.feature_selection import run_feature_selection

    print_section("SHAP STABILITY ANALYSIS -- Publication-Critical")

    # Load data
    X, y, label_encoder, omics_groups, _ = run_data_pipeline()
    X_final, _, _, _ = run_feature_selection(X, y, omics_groups)

    # Run stability analysis
    results = run_shap_stability(X_final, y, n_splits=5, top_k=20)

    # Generate plot
    plot_shap_stability(results)

    print_section("SHAP STABILITY ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
