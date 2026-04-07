"""
Phase 1 – Data Loading
======================
Loads the Bengaluru House Price dataset, performs initial inspection,
and returns the raw DataFrame for downstream processing.
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DATASET_FILENAME = "Bengaluru_House_Data.csv"


def load_dataset(data_dir: str = DATA_DIR, filename: str = DATASET_FILENAME) -> pd.DataFrame:
    """Load the CSV dataset and return it as a pandas DataFrame."""
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print fundamental properties of the dataset."""
    print("=" * 60)
    print("PHASE 1 — DATA LOADING & INSPECTION")
    print("=" * 60)

    # 1 ── First 5 rows
    print("\n[1] First 5 rows:\n")
    print(df.head().to_string())

    # 2 ── Shape
    print(f"\n[2] Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # 3 ── Data types
    print("\n[3] Data types:\n")
    print(df.dtypes.to_string())

    # 4 ── Summary statistics (numerical)
    print("\n[4] Summary statistics (numerical):\n")
    print(df.describe().to_string())

    # 5 ── Summary statistics (categorical)
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    if cat_cols:
        print("\n[5] Summary statistics (categorical):\n")
        print(df[cat_cols].describe().to_string())


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_dataset()
    inspect_dataset(df)
