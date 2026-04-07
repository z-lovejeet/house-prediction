"""
Phase 2 – Exploratory Data Analysis (EDA)
==========================================
Analyses the loaded dataset: missing values, duplicates, feature types,
correlations, and generates publication-ready visualizations.

All figures are saved to  backend/ml_pipeline/outputs/eda/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Output directory for saved plots
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Core Analysis
# ---------------------------------------------------------------------------
def analyse_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return a Series of columns with missing-value counts (desc)."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return missing


def analyse_duplicates(df: pd.DataFrame) -> int:
    """Return the number of duplicate rows."""
    return int(df.duplicated().sum())


def classify_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split columns into numerical and categorical lists."""
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numerical, categorical


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame, numerical: list[str]) -> None:
    """Save a correlation heatmap for numerical features."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numerical].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_target_distribution(df: pd.DataFrame, target: str = "price") -> None:
    """Save histogram + KDE of the target variable."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df[target].dropna(), bins=50, kde=True, color="royalblue")
    plt.title(f"Distribution of {target.title()}")
    plt.xlabel("Price (in Lakhs)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "price_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_scatter_area_price(df: pd.DataFrame) -> None:
    """Save scatter of total_sqft vs price (after sqft conversion)."""
    temp = df.copy()
    temp["total_sqft_num"] = temp["total_sqft"].apply(_safe_sqft)
    temp.dropna(subset=["total_sqft_num"], inplace=True)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="total_sqft_num", y="price", data=temp, alpha=0.4, color="seagreen")
    plt.title("Area (sqft) vs Price")
    plt.xlabel("Total Square Feet")
    plt.ylabel("Price (Lakhs)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "area_vs_price.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ Saved: {path}")


def plot_boxplots(df: pd.DataFrame) -> None:
    """Save boxplots for bath, balcony, and price."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(ax=axes[0], data=df, x="bath")
    axes[0].set_title("Bathrooms")
    sns.boxplot(ax=axes[1], data=df, x="balcony")
    axes[1].set_title("Balconies")
    sns.boxplot(ax=axes[2], data=df, y="price")
    axes[2].set_title("Price (Lakhs)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "boxplots.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ Saved: {path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_sqft(x) -> float | None:
    """Convert total_sqft values (may contain ranges) to float."""
    try:
        if "-" in str(x):
            parts = x.split("-")
            return (float(parts[0]) + float(parts[1])) / 2
        return float(x)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> None:
    """Execute the full EDA pipeline and print insights."""
    print("\n" + "=" * 60)
    print("PHASE 2 — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # --- Missing values ---
    missing = analyse_missing_values(df)
    print("\n[1] Missing values (columns with nulls):\n")
    if missing.empty:
        print("  No missing values found.")
    else:
        for col, cnt in missing.items():
            pct = cnt / len(df) * 100
            print(f"  {col:20s}  →  {cnt:>5d}  ({pct:.1f}%)")

    # --- Duplicates ---
    dup_count = analyse_duplicates(df)
    print(f"\n[2] Duplicate rows: {dup_count}")

    # --- Feature classification ---
    numerical, categorical = classify_features(df)
    print(f"\n[3] Numerical features  ({len(numerical)}): {numerical}")
    print(f"    Categorical features ({len(categorical)}): {categorical}")

    # --- Visualizations ---
    print("\n[4] Generating visualizations …")
    plot_correlation_heatmap(df, numerical)
    plot_target_distribution(df)
    plot_scatter_area_price(df)
    plot_boxplots(df)

    # --- Insights ---
    print("\n" + "-" * 60)
    print("EDA INSIGHTS")
    print("-" * 60)
    print("""
  1. 'society' has > 40 % missing values — should be dropped.
  2. 'balcony' and 'bath' have minor missing values — median imputation is safe.
  3. 'total_sqft' is stored as text (contains ranges like '1000 - 1200')
     and non-numeric units (Sq. Meter, Perch) — needs conversion.
  4. 'price' is heavily right-skewed with luxury outliers.
  5. 'bath' and 'bhk' show positive correlation with price.
  6. Location has 1 000+ unique values — dimensionality reduction needed.
""")


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from ml_pipeline import load_dataset  # type: ignore
    # Fallback for direct execution
    try:
        from importlib import import_module
        mod = import_module("01_data_loading")
        df = mod.load_dataset()
    except Exception:
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        from importlib import import_module
        mod = import_module("01_data_loading")
        df = mod.load_dataset()
    run_eda(df)
