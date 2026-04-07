"""
Phase 3 – Data Cleaning & Preprocessing
========================================
Takes the raw DataFrame and produces model-ready features (X) and target (y).

Steps performed
---------------
1. Drop duplicates
2. Handle missing values (drop / impute)
3. Fix data types (total_sqft range → numeric)
4. Feature engineering (bhk extraction, price_per_sqft)
5. Outlier removal (sqft-per-bhk filter + location-wise price_per_sqft)
6. Dimensionality reduction for location (sparse → "other")
7. One-Hot Encoding for categorical columns
8. Feature scaling with StandardScaler
9. Separate X and y

The cleaned artefacts (CSV + scaler) are persisted to
backend/ml_pipeline/outputs/processed/
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _convert_sqft(value) -> float | None:
    """Convert total_sqft string (may be a range) to float."""
    try:
        if "-" in str(value):
            parts = value.split("-")
            return (float(parts[0].strip()) + float(parts[1].strip())) / 2
        return float(value)
    except (ValueError, TypeError):
        return None


def _remove_pps_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers using mean ± 1 std of price_per_sqft per location."""
    frames: list[pd.DataFrame] = []
    for _, group in df.groupby("location"):
        mean = group["price_per_sqft"].mean()
        std = group["price_per_sqft"].std()
        filtered = group[
            (group["price_per_sqft"] > (mean - std))
            & (group["price_per_sqft"] <= (mean + std))
        ]
        frames.append(filtered)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------
def clean_and_preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """
    Run the full cleaning pipeline.

    Returns
    -------
    X : pd.DataFrame   — scaled, encoded feature matrix
    y : pd.Series       — target variable (price)
    scaler : StandardScaler — fitted scaler (needed for deployment)
    """
    print("\n" + "=" * 60)
    print("PHASE 3 — DATA CLEANING & PREPROCESSING")
    print("=" * 60)

    # ── 1. Drop duplicates ────────────────────────────────────────
    df = df.drop_duplicates().copy()
    print(f"\n[1] After dropping duplicates : {df.shape}")

    # ── 2. Handle missing values ──────────────────────────────────
    # Drop 'society' (>40 % missing, not predictive after encoding)
    if "society" in df.columns:
        df.drop("society", axis=1, inplace=True)

    # Impute numeric columns with median
    for col in ["bath", "balcony"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Drop rows where essential categoricals are missing
    df.dropna(subset=["size", "location"], inplace=True)
    print(f"[2] After handling missing vals: {df.shape}")

    # ── 3. Fix data types — total_sqft ────────────────────────────
    df["total_sqft"] = df["total_sqft"].apply(_convert_sqft)
    df.dropna(subset=["total_sqft"], inplace=True)
    print(f"[3] After fixing total_sqft   : {df.shape}")

    # ── 4. Feature engineering ────────────────────────────────────
    df["bhk"] = df["size"].apply(lambda x: int(x.split(" ")[0]))
    df["price_per_sqft"] = df["price"] * 1e5 / df["total_sqft"]
    print(f"[4] Engineered features       : bhk, price_per_sqft")

    # ── 5. Outlier removal ────────────────────────────────────────
    # Remove houses with < 300 sqft per bedroom
    df = df[~(df["total_sqft"] / df["bhk"] < 300)]

    # Location-wise price_per_sqft outlier removal
    df = _remove_pps_outliers(df)
    print(f"[5] After outlier removal     : {df.shape}")

    # ── 6. Dimensionality reduction for location ──────────────────
    location_counts = df["location"].value_counts()
    sparse_locations = location_counts[location_counts <= 10].index
    df["location"] = df["location"].apply(
        lambda x: "other" if x in sparse_locations else x
    )
    unique_locs = df["location"].nunique()
    print(f"[6] Unique locations (reduced): {unique_locs}")

    # Drop helper columns no longer needed
    df.drop(["size", "price_per_sqft"], axis=1, inplace=True)

    # ── 7. One-Hot Encoding ───────────────────────────────────────
    cat_cols = ["location", "area_type", "availability"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"[7] After one-hot encoding    : {df.shape}")

    # ── 8. Separate X and y ───────────────────────────────────────
    y = df["price"]
    X = df.drop("price", axis=1)

    # ── 9. Feature scaling ────────────────────────────────────────
    num_cols = ["total_sqft", "bath", "balcony", "bhk"]
    num_cols = [c for c in num_cols if c in X.columns]

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    print(f"[8] Scaled numerical features : {num_cols}")

    print(f"\n✅ Final X shape: {X.shape}")
    print(f"✅ Final y shape: {y.shape}")

    return X, y, scaler


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def save_artefacts(X: pd.DataFrame, y: pd.Series, scaler: StandardScaler) -> None:
    """Write X, y, and scaler to OUTPUT_DIR for later phases."""
    X.to_csv(os.path.join(OUTPUT_DIR, "X_processed.csv"), index=False)
    y.to_csv(os.path.join(OUTPUT_DIR, "y_target.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n📁 Artefacts saved to: {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from importlib import import_module
    mod = import_module("01_data_loading")
    df = mod.load_dataset()
    X, y, scaler = clean_and_preprocess(df)
    save_artefacts(X, y, scaler)
