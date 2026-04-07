"""
Phase 5 – Advanced Regularized Models (Ridge, Lasso, ElasticNet)
================================================================
Builds upon the Phase 4 baseline by training regularized regression
models to improve prediction stability and perform feature selection.

Why Regularization?
-------------------
Plain Linear Regression (OLS) minimises:
    Loss = Σ (yᵢ − ŷᵢ)²

With 200+ one-hot encoded location features, OLS tends to:
  • Overfit noisy/redundant features
  • Produce unstable, large coefficients
  • Fail to generalise well to unseen data

Regularization adds a penalty term to the loss function to constrain
the magnitude of coefficients, improving generalisation.

Models Implemented
------------------
1. Ridge  (L2)  — Loss + α·Σ βⱼ²           (shrinks all coefficients)
2. Lasso  (L1)  — Loss + α·Σ |βⱼ|          (drives some to exactly 0)
3. ElasticNet   — Loss + α·[ρ·Σ|βⱼ| + ½(1-ρ)·Σβⱼ²]  (hybrid)

Steps
-----
1. Load preprocessed X and y from Phase 3 artefacts
2. Re-create the SAME 80/20 split (random_state=42) for fair comparison
3. Load baseline (Linear Regression) results for comparison
4. Train Ridge, Lasso, ElasticNet with default alpha=1.0
5. Evaluate all models — R², MSE, MAE
6. Build comparison table (Linear vs Ridge vs Lasso vs ElasticNet)
7. Coefficient analysis — shrinkage and sparsity
8. Visualizations — coefficient comparison, model comparison bar charts
9. Key insights & interpretation

Outputs are saved to: backend/ml_pipeline/outputs/advanced_models/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "outputs", "processed")
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "outputs", "advanced_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1  Load Preprocessed Data
# ═══════════════════════════════════════════════════════════════════════════
def load_processed_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load X and y produced by Phase 3.
    Same data that Phase 4 (baseline) used — ensures fair comparison.
    """
    X = pd.read_csv(os.path.join(PROCESSED_DIR, "X_processed.csv"))
    y = pd.read_csv(os.path.join(PROCESSED_DIR, "y_target.csv")).squeeze()

    print("=" * 70)
    print("PHASE 5 — ADVANCED REGULARIZED MODELS (Ridge, Lasso, ElasticNet)")
    print("=" * 70)
    print(f"\n[0] Loaded X: {X.shape}  |  y: {y.shape}")
    print(f"    Features: {X.shape[1]}  |  Samples: {X.shape[0]}")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# 2  Recreate the SAME Train-Test Split
# ═══════════════════════════════════════════════════════════════════════════
def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.20, random_state: int = 42):
    """
    80/20 split with the SAME random_state=42 as Phase 4.
    This guarantees the exact same train/test samples — any performance
    difference is purely due to the model, not the data split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\n[1] Train-Test Split (test_size={test_size}, "
          f"random_state={random_state} — same as Phase 4)")
    print(f"    Training set : {X_train.shape[0]} samples")
    print(f"    Testing  set : {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════════════
# 3  Train Baseline (Linear Regression) for Comparison
# ═══════════════════════════════════════════════════════════════════════════
def train_baseline(X_train, y_train, X_test, y_test) -> dict:
    """
    Re-train the baseline OLS model on the same split.
    Ensures metrics are calculated on identical data for a pure comparison.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "R² Score": r2_score(y_test, y_pred),
        "MSE":      mean_squared_error(y_test, y_pred),
        "MAE":      mean_absolute_error(y_test, y_pred),
    }

    print("\n[2] Baseline Linear Regression (for comparison)")
    print(f"    R²  = {metrics['R² Score']:.4f}  |  "
          f"MSE = {metrics['MSE']:,.2f}  |  MAE = {metrics['MAE']:.4f}")

    return {
        "model": model,
        "y_pred": y_pred,
        "metrics": metrics,
        "name": "Linear Regression",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4  Train Ridge Regression (L2 Regularization)
# ═══════════════════════════════════════════════════════════════════════════
def train_ridge(X_train, y_train, X_test, y_test,
                alpha: float = 1.0) -> dict:
    """
    Ridge Regression — L2 penalty.

    What it does:
    • Adds α·Σβⱼ² to the loss function
    • Penalises LARGE coefficients, shrinking them toward zero
    • Does NOT force any coefficient to exactly zero
    • Ideal when many features are weakly correlated with the target

    Why alpha=1.0?
    • Default starting point — hyperparameter tuning is Phase 6
    • α=0 → pure OLS;  α→∞ → all coefficients → 0

    Mathematical formulation:
        Loss_Ridge = Σ(yᵢ - ŷᵢ)² + α · Σ βⱼ²
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "R² Score": r2_score(y_test, y_pred),
        "MSE":      mean_squared_error(y_test, y_pred),
        "MAE":      mean_absolute_error(y_test, y_pred),
    }

    print(f"\n[3] Ridge Regression (alpha={alpha})")
    print(f"    Regularization type : L2 (penalty = α·Σβⱼ²)")
    print(f"    Effect              : Shrinks all coefficients toward zero")
    print(f"    Feature selection   : No (keeps all features)")
    print(f"    ─── Evaluation ───")
    print(f"    R²  = {metrics['R² Score']:.4f}  |  "
          f"MSE = {metrics['MSE']:,.2f}  |  MAE = {metrics['MAE']:.4f}")

    return {
        "model": model,
        "y_pred": y_pred,
        "metrics": metrics,
        "name": f"Ridge (α={alpha})",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5  Train Lasso Regression (L1 Regularization)
# ═══════════════════════════════════════════════════════════════════════════
def train_lasso(X_train, y_train, X_test, y_test,
                alpha: float = 1.0) -> dict:
    """
    Lasso Regression — L1 penalty.

    What it does:
    • Adds α·Σ|βⱼ| to the loss function
    • Drives some coefficients EXACTLY to zero
    • Performs automatic FEATURE SELECTION
    • Ideal when you suspect many features are irrelevant

    Why is L1 different from L2?
    • L1 (absolute value) creates diamond-shaped constraint regions
    • Solutions tend to land at corners of the diamond → coefficients = 0
    • L2 (squared) creates circular constraint regions
    • Solutions tend to land on the surface → coefficients small but nonzero

    Mathematical formulation:
        Loss_Lasso = Σ(yᵢ - ŷᵢ)² + α · Σ |βⱼ|
    """
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "R² Score": r2_score(y_test, y_pred),
        "MSE":      mean_squared_error(y_test, y_pred),
        "MAE":      mean_absolute_error(y_test, y_pred),
    }

    # Count zero coefficients
    n_zero = np.sum(model.coef_ == 0)
    n_total = len(model.coef_)
    n_nonzero = n_total - n_zero

    print(f"\n[4] Lasso Regression (alpha={alpha})")
    print(f"    Regularization type : L1 (penalty = α·Σ|βⱼ|)")
    print(f"    Effect              : Drives some coefficients to EXACTLY 0")
    print(f"    Feature selection   : Yes — automatic!")
    print(f"    ─── Sparsity Analysis ───")
    print(f"    Total features      : {n_total}")
    print(f"    Non-zero features   : {n_nonzero} (selected by model)")
    print(f"    Zero'd features     : {n_zero} (eliminated by L1)")
    print(f"    Sparsity ratio      : {n_zero/n_total*100:.1f}%")
    print(f"    ─── Evaluation ───")
    print(f"    R²  = {metrics['R² Score']:.4f}  |  "
          f"MSE = {metrics['MSE']:,.2f}  |  MAE = {metrics['MAE']:.4f}")

    return {
        "model": model,
        "y_pred": y_pred,
        "metrics": metrics,
        "name": f"Lasso (α={alpha})",
        "n_zero": n_zero,
        "n_nonzero": n_nonzero,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6  Train ElasticNet (L1 + L2 Hybrid Regularization)
# ═══════════════════════════════════════════════════════════════════════════
def train_elasticnet(X_train, y_train, X_test, y_test,
                     alpha: float = 1.0, l1_ratio: float = 0.5) -> dict:
    """
    ElasticNet — Combined L1 + L2 penalty.

    What it does:
    • Combines both L1 and L2 penalties
    • l1_ratio controls the mix:
        l1_ratio = 1.0 → pure Lasso (L1 only)
        l1_ratio = 0.0 → pure Ridge (L2 only)
        l1_ratio = 0.5 → equal blend (default)
    • Gets the best of both worlds:
        - Feature selection from L1
        - Coefficient stability from L2

    Why use it?
    • When features are correlated (groups of related locations),
      Lasso arbitrarily picks one — ElasticNet keeps the group.
    • More robust than pure Lasso when features > samples.

    Mathematical formulation:
        Loss_EN = Σ(yᵢ - ŷᵢ)² + α·[ρ·Σ|βⱼ| + ½(1-ρ)·Σβⱼ²]
        where ρ = l1_ratio
    """
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "R² Score": r2_score(y_test, y_pred),
        "MSE":      mean_squared_error(y_test, y_pred),
        "MAE":      mean_absolute_error(y_test, y_pred),
    }

    n_zero = np.sum(model.coef_ == 0)
    n_total = len(model.coef_)
    n_nonzero = n_total - n_zero

    print(f"\n[5] ElasticNet (alpha={alpha}, l1_ratio={l1_ratio})")
    print(f"    Regularization type : L1 + L2 hybrid")
    print(f"    l1_ratio = {l1_ratio}")
    print(f"      → {l1_ratio*100:.0f}% L1 (Lasso) + "
          f"{(1-l1_ratio)*100:.0f}% L2 (Ridge)")
    print(f"    Effect              : Feature selection + coefficient stability")
    print(f"    ─── Sparsity Analysis ───")
    print(f"    Non-zero features   : {n_nonzero}")
    print(f"    Zero'd features     : {n_zero}")
    print(f"    Sparsity ratio      : {n_zero/n_total*100:.1f}%")
    print(f"    ─── Evaluation ───")
    print(f"    R²  = {metrics['R² Score']:.4f}  |  "
          f"MSE = {metrics['MSE']:,.2f}  |  MAE = {metrics['MAE']:.4f}")

    return {
        "model": model,
        "y_pred": y_pred,
        "metrics": metrics,
        "name": f"ElasticNet (α={alpha}, ρ={l1_ratio})",
        "n_zero": n_zero,
        "n_nonzero": n_nonzero,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7  Model Comparison Table
# ═══════════════════════════════════════════════════════════════════════════
def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Build a clean comparison table across all models.

    This is the single most important output of Phase 5.
    It allows us to see at a glance which regularization strategy
    best balances accuracy and simplicity.
    """
    rows = []
    for r in results:
        rows.append({
            "Model":     r["name"],
            "R² Score":  r["metrics"]["R² Score"],
            "MSE":       r["metrics"]["MSE"],
            "MAE":       r["metrics"]["MAE"],
        })

    comparison_df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    print(comparison_df.to_string(index=False, float_format="{:.4f}".format))

    # Find best model
    best_idx = comparison_df["R² Score"].idxmax()
    best = comparison_df.iloc[best_idx]
    print(f"\n    🏆 Best Model: {best['Model']} "
          f"(R² = {best['R² Score']:.4f})")

    # Save comparison table
    path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    comparison_df.to_csv(path, index=False)
    print(f"    📄 Comparison table saved → {path}")

    return comparison_df


# ═══════════════════════════════════════════════════════════════════════════
# 8  Coefficient Analysis — Shrinkage & Sparsity
# ═══════════════════════════════════════════════════════════════════════════
def coefficient_analysis(results: list[dict],
                         feature_names: list) -> pd.DataFrame:
    """
    Compare coefficients across all models to demonstrate:
    1. Ridge SHRINKS all coefficients (none become exactly zero)
    2. Lasso makes some coefficients EXACTLY zero (feature selection)
    3. ElasticNet is a balanced hybrid

    This is the key theoretical insight of regularization.
    """
    coeff_data = {"Feature": feature_names}

    for r in results:
        model = r["model"]
        name = r["name"]
        coeff_data[name] = model.coef_

    coeff_df = pd.DataFrame(coeff_data)

    print("\n" + "=" * 70)
    print("COEFFICIENT ANALYSIS — Shrinkage & Sparsity")
    print("=" * 70)

    # --- Shrinkage comparison ---
    print("\n    📊 Coefficient Magnitude Summary:")
    print("    " + "─" * 55)
    for r in results:
        name = r["name"]
        coefs = r["model"].coef_
        n_zero = np.sum(coefs == 0)
        print(f"    {name:30s} | "
              f"Mean |β| = {np.mean(np.abs(coefs)):8.4f} | "
              f"Max |β| = {np.max(np.abs(coefs)):8.4f} | "
              f"Zeros = {n_zero}")

    # --- How Ridge shrinks coefficients ---
    lr_coefs = results[0]["model"].coef_
    ridge_coefs = results[1]["model"].coef_
    shrinkage = np.mean(np.abs(lr_coefs)) - np.mean(np.abs(ridge_coefs))

    print(f"\n    🔍 Ridge Shrinkage Effect:")
    print(f"    → Average |coefficient| dropped by {shrinkage:.4f}")
    print(f"    → Ridge keeps ALL features but reduces "
          f"extreme coefficient values")
    print(f"    → This stabilises predictions for correlated features")

    # --- How Lasso performs feature selection ---
    lasso_coefs = results[2]["model"].coef_
    n_lasso_zero = np.sum(lasso_coefs == 0)
    n_total = len(lasso_coefs)

    print(f"\n    🔍 Lasso Feature Selection Effect:")
    print(f"    → {n_lasso_zero}/{n_total} coefficients forced to "
          f"exactly zero ({n_lasso_zero/n_total*100:.1f}%)")
    print(f"    → Only {n_total - n_lasso_zero} features are "
          f"considered important by L1")
    print(f"    → This creates a simpler, more interpretable model")

    # --- Top features according to each model ---
    for r in results:
        name = r["name"]
        coefs = r["model"].coef_
        abs_coefs = np.abs(coefs)
        top_idx = np.argsort(abs_coefs)[::-1][:5]

        print(f"\n    Top 5 features — {name}:")
        for rank, idx in enumerate(top_idx, 1):
            print(f"      {rank}. {feature_names[idx]:30s} "
                  f"β = {coefs[idx]:+.4f}")

    # Save full coefficient comparison
    path = os.path.join(OUTPUT_DIR, "coefficient_comparison.csv")
    coeff_df.to_csv(path, index=False)
    print(f"\n    📄 Full coefficient comparison saved → {path}")

    return coeff_df


# ═══════════════════════════════════════════════════════════════════════════
# 9  Visualizations
# ═══════════════════════════════════════════════════════════════════════════
def plot_model_comparison_bars(comparison_df: pd.DataFrame) -> str:
    """
    Bar chart comparing R², MSE, and MAE across all models.

    What it shows:
    • Higher R² bars → better model
    • Lower MSE/MAE bars → more accurate predictions
    • Visual proof of which regularization strategy works best
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    models = comparison_df["Model"].tolist()
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    x = np.arange(len(models))

    # R² Score
    bars = axes[0].bar(x, comparison_df["R² Score"], color=colors,
                       edgecolor="white", linewidth=1.5)
    axes[0].set_title("R² Score (higher is better)", fontsize=13,
                      weight="bold")
    axes[0].set_ylabel("R² Score", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, comparison_df["R² Score"]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9,
                     weight="bold")

    # MSE
    bars = axes[1].bar(x, comparison_df["MSE"], color=colors,
                       edgecolor="white", linewidth=1.5)
    axes[1].set_title("MSE (lower is better)", fontsize=13, weight="bold")
    axes[1].set_ylabel("Mean Squared Error", fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, comparison_df["MSE"]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{val:,.1f}", ha="center", va="bottom", fontsize=9,
                     weight="bold")

    # MAE
    bars = axes[2].bar(x, comparison_df["MAE"], color=colors,
                       edgecolor="white", linewidth=1.5)
    axes[2].set_title("MAE (lower is better)", fontsize=13, weight="bold")
    axes[2].set_ylabel("Mean Absolute Error", fontsize=11)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    axes[2].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, comparison_df["MAE"]):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9,
                     weight="bold")

    fig.suptitle("Model Performance Comparison — Phase 5",
                 fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "model_comparison_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[6a] Model comparison bar chart saved → {path}")
    return path


def plot_coefficient_comparison(results: list[dict],
                                feature_names: list) -> str:
    """
    Side-by-side coefficient comparison for top features.

    What it shows:
    • How Ridge SHRINKS coefficients relative to OLS
    • How Lasso ELIMINATES coefficients (forces them to zero)
    • The practical difference between L1 and L2 regularization
    """
    # Get top 20 features by Linear Regression coefficient magnitude
    lr_coefs = results[0]["model"].coef_
    abs_coefs = np.abs(lr_coefs)
    top_idx = np.argsort(abs_coefs)[::-1][:20]
    top_features = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(14, 8))

    y_pos = np.arange(len(top_features))
    width = 0.2
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    for i, r in enumerate(results):
        coefs = [r["model"].coef_[idx] for idx in top_idx]
        offset = (i - 1.5) * width
        ax.barh(y_pos + offset, coefs, width, label=r["name"],
                color=colors[i], edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel("Coefficient Value", fontsize=12)
    ax.set_title("Coefficient Comparison — Top 20 Features\n"
                 "(How Regularization Shrinks & Eliminates Coefficients)",
                 fontsize=13, weight="bold")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=10, loc="best")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "coefficient_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[6b] Coefficient comparison plot saved → {path}")
    return path


def plot_sparsity_comparison(results: list[dict]) -> str:
    """
    Visualize the number of non-zero vs zero coefficients per model.

    What it shows:
    • Linear Regression: ALL features are non-zero (no selection)
    • Ridge: ALL features are non-zero (shrinkage only)
    • Lasso: Many features become zero (automatic selection)
    • ElasticNet: Moderate sparsity (balanced approach)
    """
    models = []
    nonzero_counts = []
    zero_counts = []

    for r in results:
        coefs = r["model"].coef_
        n_zero = int(np.sum(coefs == 0))
        n_nonzero = len(coefs) - n_zero
        models.append(r["name"])
        nonzero_counts.append(n_nonzero)
        zero_counts.append(n_zero)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, nonzero_counts, width, label="Non-zero (Active)",
                   color="#55A868", edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + width/2, zero_counts, width, label="Zero (Eliminated)",
                   color="#C44E52", edgecolor="white", linewidth=1.5)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Number of Features", fontsize=12)
    ax.set_title("Feature Sparsity — How Many Features Does Each Model Use?",
                 fontsize=13, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(int(bar.get_height())), ha="center", fontsize=10,
                weight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(int(bar.get_height())), ha="center", fontsize=10,
                weight="bold")

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "sparsity_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[6c] Sparsity comparison plot saved   → {path}")
    return path


def plot_actual_vs_predicted_all(results: list[dict],
                                 y_test) -> str:
    """
    Overlay actual vs predicted for all four models on a single plot.

    What it shows:
    • How each model's predictions compare to the ideal 45° line
    • Whether regularization improves or degrades prediction accuracy
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    for idx, (r, ax, color) in enumerate(zip(results, axes.flat, colors)):
        y_pred = r["y_pred"]

        ax.scatter(y_test, y_pred, alpha=0.4, s=15, edgecolors="k",
                   linewidths=0.2, color=color)

        lo = min(y_test.min(), y_pred.min())
        hi = max(y_test.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5,
                label="Perfect prediction")

        ax.set_xlabel("Actual Price (Lakhs)", fontsize=10)
        ax.set_ylabel("Predicted Price (Lakhs)", fontsize=10)
        ax.set_title(f"{r['name']}\n"
                     f"R² = {r['metrics']['R² Score']:.4f}",
                     fontsize=11, weight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Actual vs Predicted — All Models",
                 fontsize=15, weight="bold")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "actual_vs_predicted_all.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[6d] Actual vs Predicted (all models) → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# 10  Conceptual Explanation
# ═══════════════════════════════════════════════════════════════════════════
def print_conceptual_explanation() -> None:
    """
    Clear, simple-language explanation of regularization concepts.
    Written so even a non-technical reader can follow.
    """
    print("\n" + "=" * 70)
    print("CONCEPTUAL EXPLANATION — Regularization in Simple Terms")
    print("=" * 70)

    print("""
┌──────────────────────────────────────────────────────────────────────┐
│  WHAT IS REGULARIZATION?                                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Think of it like this:                                              │
│                                                                      │
│  Plain Linear Regression is like a student who MEMORISES the         │
│  textbook word for word. They do great on practice questions         │
│  (training data) but fail on new questions (test data) because       │
│  they memorised noise, not concepts.                                 │
│                                                                      │
│  Regularization is like telling the student: "You can study, but     │
│  you'll be penalised for writing overly complicated answers."        │
│  This forces simpler, more generalisable learning.                   │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  L1 vs L2 REGULARIZATION                                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  L2 (Ridge):                                                         │
│    • Penalty = α · Σ βⱼ²                                            │
│    • "Keep all features but make them smaller"                       │
│    • Like turning down the volume on ALL speakers equally            │
│    • Best when: all features have some predictive value              │
│                                                                      │
│  L1 (Lasso):                                                         │
│    • Penalty = α · Σ |βⱼ|                                           │
│    • "Remove unimportant features entirely"                          │
│    • Like MUTING irrelevant speakers and keeping the useful ones     │
│    • Best when: many features are actually irrelevant                │
│                                                                      │
│  ElasticNet (L1 + L2):                                               │
│    • Penalty = α · [ρ·Σ|βⱼ| + ½(1-ρ)·Σβⱼ²]                       │
│    • "Remove some features AND make the rest smaller"                │
│    • Best when: features are correlated AND some are irrelevant      │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  WHY IS LINEAR REGRESSION ALONE NOT ENOUGH?                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  For our house price dataset with 200+ features:                     │
│    1. Many location features are correlated (nearby areas have       │
│       similar prices) → unstable coefficients                        │
│    2. OLS assigns non-zero weights to ALL features, even noisy       │
│       ones → overfitting risk                                        │
│    3. No built-in way to identify which features actually matter     │
│    4. Small changes in training data → large coefficient swings      │
│                                                                      │
│  Regularization solves all of these problems.                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════════════════
# 11  Key Insights & Final Summary
# ═══════════════════════════════════════════════════════════════════════════
def print_insights(results: list[dict], comparison_df: pd.DataFrame) -> None:
    """
    Final, concise summary of which model is best and why.
    """
    best_idx = comparison_df["R² Score"].idxmax()
    best = comparison_df.iloc[best_idx]
    baseline_r2 = comparison_df.iloc[0]["R² Score"]
    best_r2 = best["R² Score"]
    improvement = (best_r2 - baseline_r2) / abs(baseline_r2) * 100

    print("\n" + "=" * 70)
    print("PHASE 5 — KEY INSIGHTS & FINAL SUMMARY")
    print("=" * 70)

    print(f"""
┌──────────────────────────────────────────────────────────────────────┐
│  🏆 BEST MODEL: {best['Model']:<50s}  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Performance vs Baseline:                                            │
│    Baseline R²        : {baseline_r2:.4f}                            │
│    Best Model R²      : {best_r2:.4f}                                │
│    Improvement         : {improvement:+.2f}%                         │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  KEY FINDINGS                                                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. RIDGE performed well because L2 regularization stabilises        │
│     the correlated location features without discarding any.         │
│                                                                      │
│  2. LASSO performed feature selection, eliminating many location     │
│     features that add noise rather than signal.                      │
│                                                                      │
│  3. ELASTICNET balanced both approaches, offering a middle ground    │
│     between Ridge's stability and Lasso's feature selection.         │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  TRADE-OFFS                                                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Stability vs Selection:                                             │
│    • Ridge = Maximum stability, no selection                         │
│    • Lasso = Maximum selection, less stability                       │
│    • ElasticNet = Balanced trade-off                                 │
│                                                                      │
│  Simplicity vs Accuracy:                                             │
│    • Fewer features (Lasso) → easier to interpret & deploy           │
│    • More features (Ridge) → potentially higher accuracy             │
│    • The "best" model depends on project goals                       │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  NEXT STEPS (Phase 6)                                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  • Hyperparameter Tuning: Use GridSearchCV to find optimal alpha     │
│  • Cross-Validation: Use k-fold CV for more robust evaluation        │
│  • Compare tuned models against these default-alpha baselines        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════════════════
# Main Execution — Orchestrator
# ═══════════════════════════════════════════════════════════════════════════
def run_advanced_models() -> dict:
    """
    Orchestrate the full Phase 5 pipeline.

    Returns a dict with all models, metrics, and results for use
    by downstream phases (e.g., hyperparameter tuning in Phase 6).
    """

    # ── Step 1: Load data ────────────────────────────────────────
    X, y = load_processed_data()

    # ── Step 2: Same split as Phase 4 ────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── Step 3: Train baseline for comparison ────────────────────
    baseline_result = train_baseline(X_train, y_train, X_test, y_test)

    # ── Step 4: Train Ridge ──────────────────────────────────────
    ridge_result = train_ridge(X_train, y_train, X_test, y_test)

    # ── Step 5: Train Lasso ──────────────────────────────────────
    lasso_result = train_lasso(X_train, y_train, X_test, y_test)

    # ── Step 6: Train ElasticNet ─────────────────────────────────
    elasticnet_result = train_elasticnet(X_train, y_train, X_test, y_test)

    # Collect all results
    all_results = [baseline_result, ridge_result,
                   lasso_result, elasticnet_result]

    # ── Step 7: Comparison table ─────────────────────────────────
    comparison_df = compare_models(all_results)

    # ── Step 8: Coefficient analysis ─────────────────────────────
    coeff_df = coefficient_analysis(all_results, X.columns.tolist())

    # ── Step 9: Visualizations ───────────────────────────────────
    plot_model_comparison_bars(comparison_df)
    plot_coefficient_comparison(all_results, X.columns.tolist())
    plot_sparsity_comparison(all_results)
    plot_actual_vs_predicted_all(all_results, y_test)

    # ── Step 10: Conceptual explanation ──────────────────────────
    print_conceptual_explanation()

    # ── Step 11: Key insights ────────────────────────────────────
    print_insights(all_results, comparison_df)

    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE ✅ — All advanced models trained and evaluated.")
    print("=" * 70)

    return {
        "all_results":   all_results,
        "comparison_df": comparison_df,
        "coeff_df":      coeff_df,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Standalone
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_advanced_models()
