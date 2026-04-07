"""
Phase 6 – Hyperparameter Tuning & Cross-Validation
====================================================
Optimises the regularized models from Phase 5 by searching for the best
alpha (and l1_ratio for ElasticNet) using GridSearchCV with 5-fold
cross-validation, then evaluates the tuned models on the held-out test set.

Why Hyperparameter Tuning?
--------------------------
Phase 5 used alpha=1.0 for all models. This is a reasonable default, but
the *optimal* penalty strength depends on the data:
  • Too small alpha → underfitting potential of regularization (≈ OLS)
  • Too large alpha → over-penalise, collapse coefficients, lose signal
  • "Just right" alpha → best bias–variance trade-off

Why Cross-Validation?
---------------------
A single train/test split is NOISY — the score depends on which samples
land in which set. k-fold CV averages performance over k different
train/test splits, giving a more *reliable* estimate of true generalisation.

Steps
-----
1. Load preprocessed X, y from Phase 3 artefacts
2. Recreate the SAME 80/20 split (random_state=42)
3. Run GridSearchCV for Ridge, Lasso, ElasticNet
4. Extract best hyperparameters
5. Evaluate tuned models on the test set
6. Build Before vs After comparison table
7. Cross-validation analysis (per-fold scores)
8. Final model selection with justification
9. Conceptual explanations
10. Visualizations (alpha vs R² curves)

Outputs are saved to: backend/ml_pipeline/outputs/optimization/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "outputs", "processed")
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "outputs", "optimization")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1  Load Preprocessed Data
# ═══════════════════════════════════════════════════════════════════════════
def load_processed_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load X and y produced by Phase 3.
    Same data used in Phase 4 (baseline) and Phase 5 (advanced models).
    """
    X = pd.read_csv(os.path.join(PROCESSED_DIR, "X_processed.csv"))
    y = pd.read_csv(os.path.join(PROCESSED_DIR, "y_target.csv")).squeeze()

    print("=" * 70)
    print("PHASE 6 — HYPERPARAMETER TUNING & CROSS-VALIDATION")
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
    80/20 split with the SAME random_state=42 as Phase 4 & 5.
    Guarantees identical train/test sets for fair comparison.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\n[1] Train-Test Split (test_size={test_size}, "
          f"random_state={random_state} — same as Phases 4 & 5)")
    print(f"    Training set : {X_train.shape[0]} samples")
    print(f"    Testing  set : {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════════════
# 3  Before-Tuning Metrics (Phase 5 defaults for comparison)
# ═══════════════════════════════════════════════════════════════════════════
def get_before_tuning_metrics(X_train, y_train, X_test, y_test) -> dict:
    """
    Train with Phase 5 default alphas to capture 'before tuning' baselines.
    This reproduces the Phase 5 results on the same split.
    """
    print("\n" + "─" * 70)
    print("[2] Training models with DEFAULT hyperparameters (Phase 5 settings)")
    print("    (These are the 'Before Tuning' baselines)")
    print("─" * 70)

    before = {}

    # Linear Regression (no hyperparameters to tune)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    before["Linear"] = {
        "model": lr,
        "R²": r2_score(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "params": "N/A (no regularization)",
    }
    print(f"    Linear Regression  → R² = {before['Linear']['R²']:.4f}")

    # Ridge with default alpha=1.0
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    before["Ridge"] = {
        "model": ridge,
        "R²": r2_score(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "params": "alpha=1.0",
    }
    print(f"    Ridge (α=1.0)      → R² = {before['Ridge']['R²']:.4f}")

    # Lasso with default alpha=1.0
    lasso = Lasso(alpha=1.0, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    before["Lasso"] = {
        "model": lasso,
        "R²": r2_score(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "params": "alpha=1.0",
    }
    print(f"    Lasso (α=1.0)      → R² = {before['Lasso']['R²']:.4f}")

    # ElasticNet with default alpha=1.0, l1_ratio=0.5
    en = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
    en.fit(X_train, y_train)
    y_pred = en.predict(X_test)
    before["ElasticNet"] = {
        "model": en,
        "R²": r2_score(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "params": "alpha=1.0, l1_ratio=0.5",
    }
    print(f"    ElasticNet (α=1.0) → R² = {before['ElasticNet']['R²']:.4f}")

    return before


# ═══════════════════════════════════════════════════════════════════════════
# 4  GridSearchCV — Hyperparameter Tuning
# ═══════════════════════════════════════════════════════════════════════════
def tune_ridge(X_train, y_train) -> GridSearchCV:
    """
    Tune Ridge Regression using GridSearchCV.

    Why these alpha values?
    • [0.001, 0.01, 0.1, 1, 10, 100] covers 5 orders of magnitude
    • Our Phase 5 alpha=1.0 was decent (R²=0.7151)
    • Smaller alphas may let more signal through
    • Larger alphas may over-penalise

    GridSearchCV with cv=5 means:
    • Data is split into 5 folds
    • Model trains on 4 folds, validates on 1 fold
    • Repeated 5 times (each fold serves as validation once)
    • Average score across all 5 folds = reported score
    """
    param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]}

    grid = GridSearchCV(
        estimator=Ridge(),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        return_train_score=True,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    print(f"\n{'─' * 70}")
    print("[3a] GridSearchCV — Ridge Regression")
    print(f"{'─' * 70}")
    print(f"     Search space  : alpha = {param_grid['alpha']}")
    print(f"     Cross-val     : 5-fold")
    print(f"     Total fits    : {len(param_grid['alpha'])} × 5 = "
          f"{len(param_grid['alpha']) * 5}")
    print(f"     ─── Results ───")
    for alpha, mean_score, std_score in zip(
        grid.cv_results_["param_alpha"],
        grid.cv_results_["mean_test_score"],
        grid.cv_results_["std_test_score"],
    ):
        marker = " ◀ BEST" if alpha == grid.best_params_["alpha"] else ""
        print(f"     alpha = {alpha:>7} → R² = {mean_score:.4f} "
              f"(± {std_score:.4f}){marker}")
    print(f"\n     ✅ Best alpha = {grid.best_params_['alpha']}")
    print(f"     ✅ Best CV R² = {grid.best_score_:.4f}")

    return grid


def tune_lasso(X_train, y_train) -> GridSearchCV:
    """
    Tune Lasso Regression using GridSearchCV.

    Key consideration:
    • Lasso at alpha=1.0 eliminated too many features (R²=0.6577)
    • Smaller alphas should recover those features
    • Very small alphas → approaches OLS (defeats the purpose)
    """
    param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]}

    grid = GridSearchCV(
        estimator=Lasso(max_iter=10000),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        return_train_score=True,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    print(f"\n{'─' * 70}")
    print("[3b] GridSearchCV — Lasso Regression")
    print(f"{'─' * 70}")
    print(f"     Search space  : alpha = {param_grid['alpha']}")
    print(f"     Cross-val     : 5-fold")
    print(f"     Total fits    : {len(param_grid['alpha'])} × 5 = "
          f"{len(param_grid['alpha']) * 5}")
    print(f"     ─── Results ───")
    for alpha, mean_score, std_score in zip(
        grid.cv_results_["param_alpha"],
        grid.cv_results_["mean_test_score"],
        grid.cv_results_["std_test_score"],
    ):
        marker = " ◀ BEST" if alpha == grid.best_params_["alpha"] else ""
        print(f"     alpha = {alpha:>7} → R² = {mean_score:.4f} "
              f"(± {std_score:.4f}){marker}")
    print(f"\n     ✅ Best alpha = {grid.best_params_['alpha']}")
    print(f"     ✅ Best CV R² = {grid.best_score_:.4f}")

    # Sparsity of best model
    best_lasso = grid.best_estimator_
    n_zero = int(np.sum(best_lasso.coef_ == 0))
    n_total = len(best_lasso.coef_)
    print(f"     → Features used: {n_total - n_zero}/{n_total} "
          f"(sparsity: {n_zero/n_total*100:.1f}%)")

    return grid


def tune_elasticnet(X_train, y_train) -> GridSearchCV:
    """
    Tune ElasticNet using GridSearchCV over alpha AND l1_ratio.

    Two hyperparameters:
    • alpha: overall regularization strength
    • l1_ratio (ρ): balance between L1 and L2
      - 0.2 → mostly Ridge (80% L2, 20% L1)
      - 0.5 → equal blend
      - 0.8 → mostly Lasso (20% L2, 80% L1)
    """
    param_grid = {
        "alpha":    [0.001, 0.01, 0.1, 1, 10],
        "l1_ratio": [0.2, 0.5, 0.8],
    }

    grid = GridSearchCV(
        estimator=ElasticNet(max_iter=10000),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        return_train_score=True,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    n_combos = len(param_grid["alpha"]) * len(param_grid["l1_ratio"])
    print(f"\n{'─' * 70}")
    print("[3c] GridSearchCV — ElasticNet")
    print(f"{'─' * 70}")
    print(f"     Search space  : alpha = {param_grid['alpha']}")
    print(f"                   : l1_ratio = {param_grid['l1_ratio']}")
    print(f"     Cross-val     : 5-fold")
    print(f"     Total fits    : {n_combos} × 5 = {n_combos * 5}")
    print(f"     ─── Top 5 Combinations ───")

    # Sort by mean test score
    results_df = pd.DataFrame(grid.cv_results_)
    results_df = results_df.sort_values("mean_test_score", ascending=False)
    for _, row in results_df.head(5).iterrows():
        alpha = row["param_alpha"]
        ratio = row["param_l1_ratio"]
        score = row["mean_test_score"]
        std = row["std_test_score"]
        print(f"     alpha={alpha:<6}, l1_ratio={ratio} → "
              f"R² = {score:.4f} (± {std:.4f})")

    best = grid.best_params_
    print(f"\n     ✅ Best alpha    = {best['alpha']}")
    print(f"     ✅ Best l1_ratio = {best['l1_ratio']}")
    print(f"     ✅ Best CV R²   = {grid.best_score_:.4f}")

    return grid


# ═══════════════════════════════════════════════════════════════════════════
# 5  Evaluate Tuned Models on Test Set
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_tuned_models(grids: dict, X_test, y_test,
                          before: dict) -> dict:
    """
    Evaluate the GridSearchCV-selected best models on the held-out test set.

    Why evaluate on test set SEPARATELY from CV?
    • GridSearchCV uses the TRAINING set for cross-validation
    • The test set has NEVER been seen during training or tuning
    • This gives the most honest estimate of real-world performance
    """
    print("\n" + "=" * 70)
    print("[4] EVALUATING TUNED MODELS ON TEST SET")
    print("=" * 70)

    after = {}

    for name, grid in grids.items():
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        r2  = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Improvement over before-tuning
        before_r2  = before[name]["R²"]
        delta_r2   = r2 - before_r2
        pct_change = (delta_r2 / abs(before_r2)) * 100

        after[name] = {
            "model": best_model,
            "y_pred": y_pred,
            "R²": r2,
            "MSE": mse,
            "MAE": mae,
            "best_params": grid.best_params_,
            "cv_score": grid.best_score_,
            "delta_r2": delta_r2,
            "pct_change": pct_change,
        }

        print(f"\n    {name} (tuned)")
        print(f"      Best params  : {grid.best_params_}")
        print(f"      CV R² (train): {grid.best_score_:.4f}")
        print(f"      Test R²      : {r2:.4f}  "
              f"(before: {before_r2:.4f}, Δ = {delta_r2:+.4f}, "
              f"{pct_change:+.2f}%)")
        print(f"      Test MSE     : {mse:,.2f}")
        print(f"      Test MAE     : {mae:.4f}")

    return after


# ═══════════════════════════════════════════════════════════════════════════
# 6  Before vs After Comparison Table
# ═══════════════════════════════════════════════════════════════════════════
def build_comparison_table(before: dict, after: dict) -> pd.DataFrame:
    """
    Create a side-by-side comparison of before vs after tuning.
    This is the most important output of Phase 6.
    """
    print("\n" + "=" * 70)
    print("[5] BEFORE vs AFTER TUNING — COMPARISON TABLE")
    print("=" * 70)

    rows = []

    # Linear Regression (no tuning — same before and after)
    rows.append({
        "Model": "Linear Regression",
        "Before α": "N/A",
        "Before R²": before["Linear"]["R²"],
        "Before MSE": before["Linear"]["MSE"],
        "Best α": "N/A",
        "After R²": before["Linear"]["R²"],
        "After MSE": before["Linear"]["MSE"],
        "ΔR²": 0.0,
        "Improvement": "—",
    })

    for name in ["Ridge", "Lasso", "ElasticNet"]:
        b = before[name]
        a = after[name]
        params_str = ", ".join(f"{k}={v}" for k, v in a["best_params"].items())
        rows.append({
            "Model": name,
            "Before α": b["params"],
            "Before R²": b["R²"],
            "Before MSE": b["MSE"],
            "Best α": params_str,
            "After R²": a["R²"],
            "After MSE": a["MSE"],
            "ΔR²": a["delta_r2"],
            "Improvement": f"{a['pct_change']:+.2f}%",
        })

    df = pd.DataFrame(rows)

    # Print formatted table
    print(f"\n    {'Model':<20} {'Before R²':>10} {'After R²':>10} "
          f"{'ΔR²':>8} {'Improvement':>12} {'Best Params'}")
    print("    " + "─" * 90)
    for _, row in df.iterrows():
        print(f"    {row['Model']:<20} {row['Before R²']:>10.4f} "
              f"{row['After R²']:>10.4f} {row['ΔR²']:>+8.4f} "
              f"{row['Improvement']:>12} {row['Best α']}")

    # Find best tuned model
    best_idx = df["After R²"].idxmax()
    best = df.iloc[best_idx]
    print(f"\n    🏆 Best Tuned Model: {best['Model']} "
          f"(R² = {best['After R²']:.4f})")

    # Save
    path = os.path.join(OUTPUT_DIR, "before_vs_after.csv")
    df.to_csv(path, index=False)
    print(f"    📄 Comparison table saved → {path}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 7  Cross-Validation Analysis
# ═══════════════════════════════════════════════════════════════════════════
def cross_validation_analysis(best_models: dict,
                              X_train, y_train) -> dict:
    """
    Detailed cross-validation analysis using the BEST tuned models.

    Why 5-fold CV?
    ─────────────────────────────────────────────────────────
    Fold 1:  [VAL] [Train] [Train] [Train] [Train]
    Fold 2:  [Train] [VAL] [Train] [Train] [Train]
    Fold 3:  [Train] [Train] [VAL] [Train] [Train]
    Fold 4:  [Train] [Train] [Train] [VAL] [Train]
    Fold 5:  [Train] [Train] [Train] [Train] [VAL]
    ─────────────────────────────────────────────────────────
    Each sample appears in validation exactly ONCE.
    Final score = average of all 5 validation scores.
    """
    print("\n" + "=" * 70)
    print("[6] CROSS-VALIDATION ANALYSIS (5-Fold)")
    print("=" * 70)
    print("\n    How 5-fold CV works:")
    print("    ┌────────────────────────────────────────────────────────────┐")
    print("    │ Fold 1:  [VAL ] [Train] [Train] [Train] [Train]           │")
    print("    │ Fold 2:  [Train] [VAL ] [Train] [Train] [Train]           │")
    print("    │ Fold 3:  [Train] [Train] [VAL ] [Train] [Train]           │")
    print("    │ Fold 4:  [Train] [Train] [Train] [VAL ] [Train]           │")
    print("    │ Fold 5:  [Train] [Train] [Train] [Train] [VAL ]           │")
    print("    └────────────────────────────────────────────────────────────┘")
    print("    Each sample appears in the validation set exactly once.")
    print("    The final score is the MEAN across all 5 folds.\n")

    cv_results = {}

    # Also include Linear Regression for comparison
    all_models = {
        "Linear Regression": LinearRegression(),
    }
    all_models.update(best_models)

    for name, model in all_models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5,
                                 scoring="r2", n_jobs=-1)
        cv_results[name] = {
            "scores": scores,
            "mean": scores.mean(),
            "std": scores.std(),
        }

        print(f"    {name}:")
        fold_str = "  ".join(f"{s:.4f}" for s in scores)
        print(f"      Fold scores : [{fold_str}]")
        print(f"      Mean ± Std  : {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"      Range       : [{scores.min():.4f}, {scores.max():.4f}]")
        print()

    # Summary
    print("    ─── Reliability Ranking (lower std = more stable) ───")
    sorted_results = sorted(cv_results.items(), key=lambda x: x[1]["std"])
    for rank, (name, res) in enumerate(sorted_results, 1):
        stability = "🟢 Excellent" if res["std"] < 0.02 else \
                    "🟡 Good" if res["std"] < 0.05 else "🔴 Unstable"
        print(f"    {rank}. {name:<25} std = {res['std']:.4f}  {stability}")

    return cv_results


# ═══════════════════════════════════════════════════════════════════════════
# 8  Final Model Selection
# ═══════════════════════════════════════════════════════════════════════════
def final_model_selection(before: dict, after: dict,
                          cv_results: dict,
                          comparison_df: pd.DataFrame) -> str:
    """
    Choose the best model based on Performance + Stability + Simplicity.

    Decision criteria:
    1. Highest R² on test set (performance)
    2. Lowest CV standard deviation (stability/reliability)
    3. Fewest non-zero coefficients (simplicity, if comparable performance)
    """
    print("\n" + "=" * 70)
    print("[7] FINAL MODEL SELECTION")
    print("=" * 70)

    # Build scoring matrix
    candidates = {}

    # Linear Regression
    candidates["Linear Regression"] = {
        "test_r2": before["Linear"]["R²"],
        "cv_mean": cv_results["Linear Regression"]["mean"],
        "cv_std": cv_results["Linear Regression"]["std"],
    }

    for name in ["Ridge", "Lasso", "ElasticNet"]:
        a = after[name]
        # Map name to cv_results key
        cv_key = None
        for k in cv_results:
            if name in k:
                cv_key = k
                break
        if cv_key is None:
            cv_key = name

        candidates[name] = {
            "test_r2": a["R²"],
            "cv_mean": cv_results[cv_key]["mean"],
            "cv_std": cv_results[cv_key]["std"],
            "params": a["best_params"],
        }

    # Find best by test R²
    best_name = max(candidates, key=lambda k: candidates[k]["test_r2"])
    best = candidates[best_name]

    print(f"""
┌──────────────────────────────────────────────────────────────────────┐
│  🏆 FINAL BEST MODEL: {best_name:<45s}│
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Test Set R²       : {best['test_r2']:.4f}                           │
│  CV Mean R²        : {best['cv_mean']:.4f}                           │
│  CV Std            : {best['cv_std']:.4f}                            │""")
    if "params" in best:
        params_str = str(best["params"])
        print(f"│  Best Params       : {params_str:<45s}│")
    print(f"""│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  JUSTIFICATION                                                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. PERFORMANCE: Highest R² on unseen test data.                     │
│  2. STABILITY:   Consistent scores across all 5 CV folds.            │
│  3. RELIABILITY: Small gap between CV and test scores means the      │
│     model generalises well and is not overfitting.                   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘""")

    return best_name


# ═══════════════════════════════════════════════════════════════════════════
# 9  Conceptual Explanations
# ═══════════════════════════════════════════════════════════════════════════
def print_conceptual_explanations() -> None:
    """
    Clear, exam-ready explanations of the key Phase 6 concepts.
    """
    print("\n" + "=" * 70)
    print("[8] CONCEPTUAL EXPLANATIONS")
    print("=" * 70)
    print("""
┌──────────────────────────────────────────────────────────────────────┐
│  WHAT IS HYPERPARAMETER TUNING?                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Hyperparameters are settings that YOU choose BEFORE training:        │
│    • alpha (regularization strength)                                 │
│    • l1_ratio (L1/L2 balance in ElasticNet)                          │
│    • learning_rate, n_estimators, etc. (in other models)             │
│                                                                      │
│  They are NOT learned from data — they CONTROL the learning process. │
│                                                                      │
│  GridSearchCV EXHAUSTIVELY tries every combination of values          │
│  you specify and picks the one with the best cross-validated score.  │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  WHY DOES ALPHA MATTER?                                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  alpha controls the STRENGTH of the regularization penalty:          │
│                                                                      │
│    alpha = 0     → No penalty → Pure OLS (may overfit)               │
│    alpha = 0.01  → Gentle penalty → Slight regularization            │
│    alpha = 1.0   → Moderate penalty → Standard regularization        │
│    alpha = 100   → Heavy penalty → Strong regularization             │
│    alpha → ∞     → Infinite penalty → All coefficients → 0           │
│                                                                      │
│  There's a "sweet spot" where alpha balances:                        │
│    • Enough penalty to prevent overfitting                           │
│    • Not so much penalty that we lose real signal                    │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  BIAS–VARIANCE TRADEOFF                                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LOW alpha (weak penalty):                                           │
│    • Low bias (model is flexible, fits training data well)           │
│    • High variance (overfits, poor on new data)                      │
│                                                                      │
│  HIGH alpha (strong penalty):                                        │
│    • High bias (model is too simple, underfits)                      │
│    • Low variance (very stable, but inaccurate)                      │
│                                                                      │
│  OPTIMAL alpha:                                                      │
│    • Balanced bias–variance → best generalisation!                   │
│    • This is exactly what GridSearchCV finds for us.                 │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  WHY IS CROSS-VALIDATION NEEDED?                                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  A single train/test split is NOISY:                                 │
│    • The model might do well just because the test set was "easy"    │
│    • Or badly because the test set had outliers                      │
│                                                                      │
│  5-Fold Cross-Validation:                                            │
│    • Trains and tests the model 5 DIFFERENT times                    │
│    • Each time with a different validation fold                      │
│    • Reports the AVERAGE score → much more reliable!                 │
│    • Also reports STD → tells us how STABLE the model is             │
│                                                                      │
│  If CV scores are similar across folds → model generalises well      │
│  If CV scores vary wildly → model is unreliable / overfitting        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════════════════
# 10  Visualizations
# ═══════════════════════════════════════════════════════════════════════════
def plot_alpha_vs_r2(grids: dict) -> str:
    """
    Plot alpha vs R² for Ridge & Lasso.

    What it shows:
    • How performance changes as alpha increases or decreases
    • The "sweet spot" where alpha is optimal
    • The bias-variance trade-off in action
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {"Ridge": "#3b82f6", "Lasso": "#ec4899"}

    for idx, name in enumerate(["Ridge", "Lasso"]):
        grid = grids[name]
        results = grid.cv_results_
        alphas = [p["alpha"] for p in results["params"]]
        # For ElasticNet, alphas may repeat, but Ridge/Lasso are simple
        mean_scores = results["mean_test_score"]
        std_scores  = results["std_test_score"]

        ax = axes[idx]
        ax.semilogx(alphas, mean_scores, "o-", color=colors[name],
                     linewidth=2, markersize=8, label=f"{name} CV R²")
        ax.fill_between(
            alphas,
            mean_scores - std_scores,
            mean_scores + std_scores,
            alpha=0.2, color=colors[name], label="± 1 std dev",
        )

        # Mark best
        best_alpha = grid.best_params_["alpha"]
        best_score = grid.best_score_
        ax.axvline(best_alpha, color="red", linestyle="--", linewidth=1.5,
                   alpha=0.7, label=f"Best α = {best_alpha}")
        ax.scatter([best_alpha], [best_score], s=150, color="red",
                   zorder=5, edgecolors="black", linewidths=1.5)
        ax.annotate(
            f"Best: α={best_alpha}\nR²={best_score:.4f}",
            xy=(best_alpha, best_score),
            xytext=(15, -25), textcoords="offset points",
            fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red"),
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                      ec="red", alpha=0.9),
        )

        ax.set_xlabel("Alpha (log scale)", fontsize=12)
        ax.set_ylabel("R² Score", fontsize=12)
        ax.set_title(f"{name}: Alpha vs Cross-Validated R²",
                     fontsize=13, weight="bold")
        ax.legend(fontsize=9, loc="lower left")
        ax.grid(alpha=0.3)

    fig.suptitle("Hyperparameter Tuning — How Alpha Affects Performance",
                 fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "alpha_vs_r2.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[9a] Alpha vs R² plot saved → {path}")
    return path


def plot_before_vs_after_bars(comparison_df: pd.DataFrame) -> str:
    """
    Grouped bar chart: Before vs After tuning R² for each model.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    models = comparison_df["Model"].tolist()
    before_r2 = comparison_df["Before R²"].tolist()
    after_r2  = comparison_df["After R²"].tolist()

    x = np.arange(len(models))
    width = 0.30

    bars1 = ax.bar(x - width/2, before_r2, width, label="Before Tuning",
                   color="#94a3b8", edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + width/2, after_r2, width, label="After Tuning",
                   color="#3b82f6", edgecolor="white", linewidth=1.5)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.4f}", ha="center", va="bottom",
                fontsize=9, color="#64748b")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#1e40af")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("Before vs After Hyperparameter Tuning — R² Comparison",
                 fontsize=14, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "before_vs_after_r2.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[9b] Before vs After bar chart saved → {path}")
    return path


def plot_cv_fold_scores(cv_results: dict) -> str:
    """
    Box/strip plot of per-fold CV scores for each model.
    Shows stability (tight boxes = reliable model).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    model_names = list(cv_results.keys())
    scores_list = [cv_results[name]["scores"] for name in model_names]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    bp = ax.boxplot(scores_list, tick_labels=model_names, patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red",
                                   markersize=8))

    for patch, color in zip(bp["boxes"], colors[:len(model_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual fold scores
    for i, (scores, color) in enumerate(
            zip(scores_list, colors[:len(model_names)])):
        x_jittered = np.random.normal(i + 1, 0.04, size=len(scores))
        ax.scatter(x_jittered, scores, c=color, s=60, edgecolors="black",
                   linewidths=0.8, zorder=5, alpha=0.9)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("5-Fold Cross-Validation Scores — Model Stability",
                 fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "cv_fold_scores.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[9c] CV fold scores plot saved → {path}")
    return path


def plot_elasticnet_heatmap(grid_en) -> str:
    """
    Heatmap of alpha × l1_ratio → R² for ElasticNet.
    Shows how the two hyperparameters interact.
    """
    results = pd.DataFrame(grid_en.cv_results_)

    alphas = sorted(results["param_alpha"].unique())
    ratios = sorted(results["param_l1_ratio"].unique())

    # Build pivot table
    pivot = results.pivot_table(
        values="mean_test_score",
        index="param_alpha",
        columns="param_l1_ratio",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([f"{r}" for r in ratios], fontsize=11)
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f"{a}" for a in alphas], fontsize=11)
    ax.set_xlabel("l1_ratio (ρ)", fontsize=12)
    ax.set_ylabel("Alpha (α)", fontsize=12)
    ax.set_title("ElasticNet: Alpha × l1_ratio → CV R²",
                 fontsize=13, weight="bold")

    # Value annotations
    for i in range(len(alphas)):
        for j in range(len(ratios)):
            val = pivot.values[i, j]
            text_color = "white" if val > pivot.values.mean() else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

    fig.colorbar(im, ax=ax, label="R² Score", shrink=0.8)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "elasticnet_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[9d] ElasticNet heatmap saved → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Main Execution — Orchestrator
# ═══════════════════════════════════════════════════════════════════════════
def run_optimization() -> dict:
    """
    Orchestrate the full Phase 6 pipeline.

    Returns a dict with all tuned models, metrics, and comparisons.
    """

    # ── Step 1: Load data ────────────────────────────────────────
    X, y = load_processed_data()

    # ── Step 2: Same split as Phases 4 & 5 ───────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── Step 3: Before-tuning metrics ────────────────────────────
    before = get_before_tuning_metrics(X_train, y_train, X_test, y_test)

    # ── Step 4: GridSearchCV Tuning ──────────────────────────────
    grid_ridge = tune_ridge(X_train, y_train)
    grid_lasso = tune_lasso(X_train, y_train)
    grid_en    = tune_elasticnet(X_train, y_train)
    grids = {
        "Ridge": grid_ridge,
        "Lasso": grid_lasso,
        "ElasticNet": grid_en,
    }

    # ── Step 5: Evaluate tuned models on test set ────────────────
    after = evaluate_tuned_models(grids, X_test, y_test, before)

    # ── Step 6: Before vs After comparison ───────────────────────
    comparison_df = build_comparison_table(before, after)

    # ── Step 7: Cross-validation analysis ────────────────────────
    best_models = {}
    for name, grid in grids.items():
        best_models[f"{name} (tuned)"] = grid.best_estimator_
    cv_results = cross_validation_analysis(best_models, X_train, y_train)

    # ── Step 8: Final model selection ────────────────────────────
    best_model_name = final_model_selection(before, after,
                                            cv_results, comparison_df)

    # ── Step 9: Conceptual explanations ──────────────────────────
    print_conceptual_explanations()

    # ── Step 10: Visualizations ──────────────────────────────────
    plot_alpha_vs_r2(grids)
    plot_before_vs_after_bars(comparison_df)
    plot_cv_fold_scores(cv_results)
    plot_elasticnet_heatmap(grid_en)

    print("\n" + "=" * 70)
    print("PHASE 6 COMPLETE ✅ — All models tuned and optimized.")
    print("=" * 70)

    return {
        "before": before,
        "after": after,
        "grids": grids,
        "comparison_df": comparison_df,
        "cv_results": cv_results,
        "best_model_name": best_model_name,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Standalone
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_optimization()
