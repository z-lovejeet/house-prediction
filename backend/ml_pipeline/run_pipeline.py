#!/usr/bin/env python3
"""
run_pipeline.py
===============
Master script that orchestrates Phase 1 → 2 → 3 → 4 → 5 → 6 sequentially.

Usage:
    cd backend/ml_pipeline/
    python run_pipeline.py
  OR
    cd backend/
    python -m ml_pipeline.run_pipeline
"""

import sys
import pathlib

# Ensure the ml_pipeline package is importable regardless of CWD
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from importlib import import_module


def main() -> None:
    # ── Phase 1: Data Loading ─────────────────────────────────────
    phase1 = import_module("01_data_loading")
    df = phase1.load_dataset()
    phase1.inspect_dataset(df)

    # ── Phase 2: EDA ──────────────────────────────────────────────
    phase2 = import_module("02_eda")
    phase2.run_eda(df)

    # ── Phase 3: Data Cleaning & Preprocessing ────────────────────
    phase3 = import_module("03_data_cleaning")
    X, y, scaler = phase3.clean_and_preprocess(df)
    phase3.save_artefacts(X, y, scaler)

    # ── Phase 4: Baseline Model (Linear Regression) ───────────────
    phase4 = import_module("04_baseline_model")
    phase4.run_baseline()

    # ── Phase 5: Advanced Models (Ridge, Lasso, ElasticNet) ───────
    phase5 = import_module("05_advanced_models")
    phase5.run_advanced_models()

    # ── Phase 6: Optimization (Hyperparameter Tuning + CV) ────────
    phase6 = import_module("06_optimization")
    phase6.run_optimization()

    print("\n" + "=" * 60)
    print("6 PHASES COMPLETED — all models trained, tuned, and compared.")
    print("=" * 60)


if __name__ == "__main__":
    main()

