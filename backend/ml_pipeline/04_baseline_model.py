"""
Phase 4 – Baseline Model (Linear Regression)
=============================================
Establishes a baseline predictive model for house price prediction.

Steps
-----
1. Load preprocessed X and y from Phase 3 artefacts
2. Train-Test split (80/20)
3. Train Linear Regression model
4. Predict on test set
5. Evaluate — R², MSE, MAE
6. Visualizations — Actual vs Predicted, Residual Plot
7. Coefficient analysis
8. Insights & interpretation

Outputs are saved to: backend/ml_pipeline/outputs/baseline/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "outputs", "processed")
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "outputs", "baseline")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1  Load preprocessed data
# ═══════════════════════════════════════════════════════════════════════════
def load_processed_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load X and y produced by Phase 3."""
    X = pd.read_csv(os.path.join(PROCESSED_DIR, "X_processed.csv"))
    y = pd.read_csv(os.path.join(PROCESSED_DIR, "y_target.csv")).squeeze()
    print("=" * 60)
    print("PHASE 4 — BASELINE MODEL (LINEAR REGRESSION)")
    print("=" * 60)
    print(f"\n[0] Loaded X: {X.shape}  |  y: {y.shape}")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# 2  Train-Test Split
# ═══════════════════════════════════════════════════════════════════════════
def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.20, random_state: int = 42):
    """
    80/20 stratified-random split.
    random_state=42 ensures reproducibility.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\n[1] Train-Test Split (test_size={test_size}, random_state={random_state})")
    print(f"    Training set : {X_train.shape[0]} samples")
    print(f"    Testing  set : {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════════════
# 3  Train Linear Regression
# ═══════════════════════════════════════════════════════════════════════════
def train_model(X_train, y_train) -> LinearRegression:
    """Fit an OLS Linear Regression on the training data."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\n[2] Linear Regression model trained ✅")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 4  Predictions
# ═══════════════════════════════════════════════════════════════════════════
def predict(model: LinearRegression, X_test) -> np.ndarray:
    """Generate predictions on the test set."""
    y_pred = model.predict(X_test)
    print(f"[3] Predictions generated for {len(y_pred)} test samples ✅")
    return y_pred


# ═══════════════════════════════════════════════════════════════════════════
# 5  Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════
def evaluate(y_test, y_pred) -> dict:
    """Compute R², MSE, and MAE."""
    metrics = {
        "R² Score":             r2_score(y_test, y_pred),
        "Mean Squared Error":   mean_squared_error(y_test, y_pred),
        "Mean Absolute Error":  mean_absolute_error(y_test, y_pred),
    }

    print("\n[4] Model Evaluation Metrics")
    print("    ─" * 15)
    for name, val in metrics.items():
        print(f"    {name:25s}: {val:,.4f}")

    # Interpretation
    r2 = metrics["R² Score"]
    print("\n    📊 Interpretation:")
    if r2 >= 0.85:
        print(f"    → R² = {r2:.4f} — Strong fit. Model explains "
              f"{r2*100:.1f}% of variance in house prices.")
    elif r2 >= 0.60:
        print(f"    → R² = {r2:.4f} — Moderate fit. Model captures general trends "
              "but misses non-linear patterns.")
    else:
        print(f"    → R² = {r2:.4f} — Weak fit. Linear model is insufficient; "
              "regularization or non-linear methods needed.")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# 6  Visualizations
# ═══════════════════════════════════════════════════════════════════════════
def plot_actual_vs_predicted(y_test, y_pred) -> str:
    """
    Scatter plot of actual vs predicted prices.

    What it shows:
    • Points near the 45° diagonal → model predicts accurately.
    • Systematic deviations indicate bias (over/under-prediction).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.45, s=18, edgecolors="k", linewidths=0.3,
               color="#4C72B0")

    # Perfect-prediction line
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")

    ax.set_xlabel("Actual Price (Lakhs)", fontsize=12)
    ax.set_ylabel("Predicted Price (Lakhs)", fontsize=12)
    ax.set_title("Actual vs Predicted House Prices", fontsize=14, weight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[5a] Actual vs Predicted plot saved → {path}")
    return path


def plot_residuals(y_test, y_pred) -> str:
    """
    Residual plot (predicted vs residuals).

    What it shows:
    • Residuals should be randomly scattered around 0 (no pattern).
    • A funnel shape → heteroscedasticity (variance depends on price).
    • Curved patterns → model is missing non-linear relationships.
    """
    residuals = y_test - y_pred

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.45, s=18, edgecolors="k", linewidths=0.3,
               color="#DD8452")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Predicted Price (Lakhs)", fontsize=12)
    ax.set_ylabel("Residuals (Actual − Predicted)", fontsize=12)
    ax.set_title("Residual Plot", fontsize=14, weight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "residuals.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[5b] Residual plot saved            → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# 7  Coefficient Analysis
# ═══════════════════════════════════════════════════════════════════════════
def coefficient_analysis(model: LinearRegression, feature_names: list) -> pd.DataFrame:
    """
    Map coefficients to feature names, sort by absolute influence.

    Interpretation:
    • Positive coefficient → feature increases predicted price.
    • Negative coefficient → feature decreases predicted price.
    • Larger absolute value → stronger influence.
    """
    coeff_df = pd.DataFrame({
        "Feature":     feature_names,
        "Coefficient": model.coef_,
    })
    coeff_df["Abs_Coefficient"] = coeff_df["Coefficient"].abs()
    coeff_df.sort_values("Abs_Coefficient", ascending=False, inplace=True)
    coeff_df.reset_index(drop=True, inplace=True)

    print("\n[6] Coefficient Analysis")
    print("    ─" * 15)
    print(f"    Intercept: {model.intercept_:,.4f}\n")
    print("    Top 15 most influential features:")
    print(coeff_df.head(15).to_string(index=False))

    # Interpretation
    top = coeff_df.iloc[0]
    print(f"\n    📊 Most impactful feature: '{top['Feature']}' "
          f"(coeff = {top['Coefficient']:+.4f})")
    print("    → Positive coefficients raise the predicted price; "
          "negative lower it.")
    print("    → One-hot encoded location features dominate because premium "
          "neighbourhoods carry inherent price premiums.")

    # Save full table
    path = os.path.join(OUTPUT_DIR, "coefficients.csv")
    coeff_df.to_csv(path, index=False)
    print(f"\n    Full coefficient table saved → {path}")

    return coeff_df


# ═══════════════════════════════════════════════════════════════════════════
# 8  Summary & Insights
# ═══════════════════════════════════════════════════════════════════════════
def print_insights(metrics: dict) -> None:
    """Print final interpretation and motivation for next phases."""
    r2 = metrics["R² Score"]
    mse = metrics["Mean Squared Error"]
    mae = metrics["Mean Absolute Error"]

    print("\n" + "=" * 60)
    print("PHASE 4 — INSIGHTS & INTERPRETATION")
    print("=" * 60)

    print(f"""
┌─────────────────────────────────────────────────────────┐
│  BASELINE PERFORMANCE SUMMARY                           │
├─────────────────────────────────────────────────────────┤
│  R² Score              : {r2:>10.4f}                     │
│  Mean Squared Error    : {mse:>10.2f}                     │
│  Mean Absolute Error   : {mae:>10.4f}                     │
└─────────────────────────────────────────────────────────┘

1. MODEL PERFORMANCE
   • The Linear Regression baseline explains {r2*100:.1f}% of the
     variance in Bengaluru house prices.
   • On average, predictions deviate by ≈ {mae:.2f} Lakhs (MAE).

2. LIMITATIONS OF LINEAR REGRESSION
   • Assumes a strict LINEAR relationship between features and price,
     but real estate pricing is inherently non-linear.
   • Highly sensitive to multicollinearity — many one-hot encoded
     location features are near-redundant.
   • No built-in feature selection; all features contribute equally
     to complexity, risking overfitting on high-dimensional data.

3. WHY RIDGE / LASSO NEXT?
   • Ridge (L2) penalises large coefficients, reducing overfitting
     and stabilising estimates when features are correlated.
   • Lasso (L1) drives some coefficients exactly to zero, performing
     automatic feature selection — ideal for our 200+ encoded
     features.
   • Comparing Ridge vs Lasso vs baseline will reveal the best
     bias-variance trade-off for this dataset.
""")


# ═══════════════════════════════════════════════════════════════════════════
# Main execution
# ═══════════════════════════════════════════════════════════════════════════
def run_baseline() -> dict:
    """Orchestrate the full Phase 4 pipeline and return results dict."""
    # Load
    X, y = load_processed_data()

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train
    model = train_model(X_train, y_train)

    # Predict
    y_pred = predict(model, X_test)

    # Evaluate
    metrics = evaluate(y_test, y_pred)

    # Visualize
    plot_actual_vs_predicted(y_test, y_pred)
    plot_residuals(y_test, y_pred)

    # Coefficient analysis
    coeff_df = coefficient_analysis(model, X.columns.tolist())

    # Insights
    print_insights(metrics)

    return {
        "model": model,
        "metrics": metrics,
        "coefficients": coeff_df,
        "y_test": y_test,
        "y_pred": y_pred,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Standalone
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_baseline()
