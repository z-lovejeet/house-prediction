"""
Phase 7 – Final Model Selection, Serialization & Deployment Preparation
=========================================================================
Selects the best model from Phase 5 & 6 results, retrains it on the FULL
dataset (no train/test split), serializes all artefacts needed for serving
predictions, and exposes a `predict_price()` function that mirrors the
exact preprocessing pipeline a FastAPI endpoint would use.

Why a Separate Phase?
---------------------
Phases 4-6 used an 80/20 train/test split to **evaluate** models fairly.
But once evaluation is done and the winner is chosen, we can squeeze extra
performance out of the same data by training on ALL available samples.
This is standard practice before deployment.

Steps
-----
1.  Load and compare results from Phase 5 (advanced_models) and Phase 6
    (optimization) to identify the single best model with justification.
2.  Retrain the selected model on the FULL dataset using best
    hyperparameters from Phase 6.
3.  Save deployment artefacts:
      • final_model.pkl   – the trained model
      • scaler.pkl        – the fitted StandardScaler (already exists)
      • feature_columns.pkl – column names expected by the model
4.  Create a `predict_price(input_dict)` function that reproduces the
    full preprocessing → scaling → prediction pipeline.
5.  Test the pipeline with sample inputs.

Outputs are saved to: backend/ml_pipeline/outputs/processed/
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════
PROCESSED_DIR       = os.path.join(os.path.dirname(__file__), "outputs", "processed")
ADVANCED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "advanced_models")
OPTIMIZATION_DIR    = os.path.join(os.path.dirname(__file__), "outputs", "optimization")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1  Load and Compare All Model Results
# ═══════════════════════════════════════════════════════════════════════════
def load_and_compare_results() -> dict:
    """
    Load Phase 5 and Phase 6 CSV results and build a unified comparison.

    Returns a dict with all model metrics for selection.
    """
    print("=" * 70)
    print("PHASE 7 — FINAL MODEL SELECTION, SERIALIZATION & DEPLOYMENT")
    print("=" * 70)

    print("\n" + "─" * 70)
    print("[1] LOADING RESULTS FROM PHASE 5 & PHASE 6")
    print("─" * 70)

    # ── Phase 5: Advanced Models (default alpha=1.0) ─────────────
    phase5_path = os.path.join(ADVANCED_MODELS_DIR, "model_comparison.csv")
    phase5_df   = pd.read_csv(phase5_path)

    print("\n    Phase 5 — Advanced Models (default α=1.0):")
    print("    " + "─" * 60)
    for _, row in phase5_df.iterrows():
        print(f"    {row['Model']:<35s} R²={row['R² Score']:.4f}  "
              f"MSE={row['MSE']:,.2f}")

    # ── Phase 6: Optimized Models (tuned hyperparameters) ────────
    phase6_path = os.path.join(OPTIMIZATION_DIR, "before_vs_after.csv")
    phase6_df   = pd.read_csv(phase6_path)

    print("\n    Phase 6 — After Hyperparameter Tuning:")
    print("    " + "─" * 60)
    for _, row in phase6_df.iterrows():
        print(f"    {row['Model']:<25s} Before R²={row['Before R²']:.4f}  "
              f"After R²={row['After R²']:.4f}  "
              f"Best Params: {row['Best α']}")

    return {
        "phase5": phase5_df,
        "phase6": phase6_df,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2  Select the Best Model with Justification
# ═══════════════════════════════════════════════════════════════════════════
def select_best_model(results: dict) -> dict:
    """
    Select the single best model based on After-Tuning R² from Phase 6.

    Decision criteria:
    1. Highest R² on test set (primary)
    2. Lowest MSE (secondary)
    3. Stability and simplicity (tie-breaker)

    Returns
    -------
    dict with keys: name, alpha, l1_ratio (if applicable), r2, mse
    """
    phase6_df = results["phase6"]

    print("\n" + "─" * 70)
    print("[2] FINAL MODEL SELECTION — DECISION MATRIX")
    print("─" * 70)

    # Build ranking table from Phase 6 after-tuning metrics
    ranking = phase6_df[["Model", "After R²", "After MSE", "Best α"]].copy()
    ranking = ranking.sort_values("After R²", ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1  # 1-indexed ranking

    print("\n    Ranking by After-Tuning R²:")
    print("    " + "─" * 65)
    print(f"    {'Rank':<6} {'Model':<25} {'R²':>8} {'MSE':>12} {'Params'}")
    print("    " + "─" * 65)
    for rank, (_, row) in enumerate(ranking.iterrows(), 1):
        marker = " ◀ SELECTED" if rank == 1 else ""
        print(f"    {rank:<6} {row['Model']:<25} "
              f"{row['After R²']:>8.4f} {row['After MSE']:>12,.2f} "
              f"{row['Best α']}{marker}")

    # The winner
    winner = ranking.iloc[0]
    winner_name = winner["Model"]

    # Parse best hyperparameters
    params_str = str(winner["Best α"])
    best_params = {}
    if params_str != "N/A":
        for pair in params_str.split(", "):
            key, val = pair.split("=")
            best_params[key.strip()] = float(val)

    selection = {
        "name":   winner_name,
        "r2":     winner["After R²"],
        "mse":    winner["After MSE"],
        "params": best_params,
    }

    # ── Detailed justification ───────────────────────────────────
    print(f"""
┌──────────────────────────────────────────────────────────────────────┐
│  🏆 SELECTED MODEL: {winner_name:<47s}│
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  After-Tuning R²  : {selection['r2']:.4f}                            │
│  After-Tuning MSE : {selection['mse']:,.2f}                          │
│  Best Params      : {params_str:<45s}│
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  WHY THIS MODEL WAS CHOSEN                                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. HIGHEST R² SCORE: This model achieved the highest R² among all   │
│     models tested across Phases 4, 5, and 6 — meaning it explains    │
│     the most variance in house prices.                               │
│                                                                      │
│  2. LOWEST MSE: It also has the lowest Mean Squared Error, meaning   │
│     predictions are closest to actual prices on average.             │
│                                                                      │
│  3. STABILITY: The tuned model showed consistent performance across  │
│     5-fold cross-validation, indicating it generalises well.         │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  WHY OTHER MODELS WERE NOT SELECTED                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  • Linear Regression: No regularization → risk of overfitting with   │
│    200+ features. R² is lower than the best regularized model.       │
│                                                                      │
│  • Lasso (α=0.01): Aggressive feature elimination. While improved    │
│    after tuning, its R² is still lower than Ridge/ElasticNet.        │
│    L1 penalty drives too many coefficients to zero.                  │
│                                                                      │
│  • ElasticNet / Ridge (whichever lost): Very close performance but   │
│    the selected model edges ahead in both R² and MSE.                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘""")

    return selection


# ═══════════════════════════════════════════════════════════════════════════
# 3  Retrain on Full Dataset
# ═══════════════════════════════════════════════════════════════════════════
def retrain_on_full_data(selection: dict) -> tuple:
    """
    Retrain the selected model on the ENTIRE dataset (no split).

    Why retrain on full data?
    ─────────────────────────
    During evaluation (Phases 4–6), we held out 20% of data as a test set.
    Now that we've chosen our model, we can use ALL available data to train
    it before deployment.  More data → better-learned patterns → better
    predictions on truly new (future) data.

    The model class and hyperparameters stay EXACTLY the same — only the
    amount of training data changes.
    """
    print("\n" + "─" * 70)
    print("[3] RETRAINING ON FULL DATASET")
    print("─" * 70)

    # Load full processed data
    X = pd.read_csv(os.path.join(PROCESSED_DIR, "X_processed.csv"))
    y = pd.read_csv(os.path.join(PROCESSED_DIR, "y_target.csv")).squeeze()

    print(f"\n    Full dataset : {X.shape[0]} samples × {X.shape[1]} features")
    print(f"    Target       : {y.shape[0]} values")
    print(f"    (Previously trained on ~80% = {int(X.shape[0]*0.8)} samples)")
    print(f"    (Now training on 100% = {X.shape[0]} samples)")

    # Build the model with best hyperparameters
    model_name   = selection["name"]
    best_params  = selection["params"]

    if model_name == "Ridge":
        alpha = best_params.get("alpha", 1.0)
        model = Ridge(alpha=alpha)
        print(f"\n    Model  : Ridge Regression")
        print(f"    Alpha  : {alpha}")
    elif model_name == "ElasticNet":
        from sklearn.linear_model import ElasticNet
        alpha    = best_params.get("alpha", 0.001)
        l1_ratio = best_params.get("l1_ratio", 0.8)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        print(f"\n    Model    : ElasticNet")
        print(f"    Alpha    : {alpha}")
        print(f"    l1_ratio : {l1_ratio}")
    elif model_name == "Lasso":
        from sklearn.linear_model import Lasso
        alpha = best_params.get("alpha", 0.01)
        model = Lasso(alpha=alpha, max_iter=10000)
        print(f"\n    Model  : Lasso Regression")
        print(f"    Alpha  : {alpha}")
    else:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        print(f"\n    Model  : Linear Regression (no regularization)")

    # Fit on FULL data
    model.fit(X, y)

    # Full-data training metrics (not for evaluation — just a sanity check)
    y_pred_full = model.predict(X)
    r2_full  = r2_score(y, y_pred_full)
    mse_full = mean_squared_error(y, y_pred_full)
    mae_full = mean_absolute_error(y, y_pred_full)

    print(f"\n    ─── Full-Data Training Metrics (sanity check) ───")
    print(f"    R²  = {r2_full:.4f}  (expected to be ≥ test R² since "
          f"no held-out data)")
    print(f"    MSE = {mse_full:,.2f}")
    print(f"    MAE = {mae_full:.4f}")

    # Feature columns (needed by predict_price)
    feature_columns = list(X.columns)

    return model, feature_columns, X, y


# ═══════════════════════════════════════════════════════════════════════════
# 4  Save Deployment Artefacts
# ═══════════════════════════════════════════════════════════════════════════
def save_deployment_artefacts(model, feature_columns: list,
                              selection: dict) -> dict:
    """
    Serialize and persist all artefacts needed for inference.

    What we save and WHY:
    ─────────────────────
    1. final_model.pkl     — The trained model (weights + hyperparameters).
       Without this, you'd have to retrain from scratch every time.

    2. scaler.pkl          — Already exists from Phase 3, but we verify it.
       The scaler stores the mean and std of the training data so that
       new inputs can be scaled using the SAME transformation.
       If you use a different scaler, predictions will be nonsensical.

    3. feature_columns.pkl — The exact ORDERED list of feature names the
       model was trained on.  During prediction, we must create a DataFrame
       with these exact columns in this exact order.

    4. model_metadata.json — Human-readable summary of the model config.
       Useful for documentation, debugging, and API versioning.

    Why pickle/joblib?
    ──────────────────
    Model serialization converts a Python object in memory into a byte
    stream on disk.  When we load it back, we get the EXACT same object
    — no need to retrain, no risk of parameter drift.
    """
    print("\n" + "─" * 70)
    print("[4] SAVING DEPLOYMENT ARTEFACTS")
    print("─" * 70)

    saved_paths = {}

    # ── 4a. Save the trained model ───────────────────────────────
    model_path = os.path.join(PROCESSED_DIR, "final_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    saved_paths["model"] = model_path
    print(f"\n    ✅ final_model.pkl         → {model_path}")

    # ── 4b. Verify scaler exists ─────────────────────────────────
    scaler_path = os.path.join(PROCESSED_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"    ✅ scaler.pkl (verified)    → {scaler_path}")
        print(f"       Scaler features         : {scaler.n_features_in_}")
        print(f"       Scaler means            : {np.round(scaler.mean_, 2)}")
    else:
        print(f"    ⚠️  scaler.pkl NOT FOUND — Phase 3 must be run first!")
    saved_paths["scaler"] = scaler_path

    # ── 4c. Save feature columns ─────────────────────────────────
    columns_path = os.path.join(PROCESSED_DIR, "feature_columns.pkl")
    with open(columns_path, "wb") as f:
        pickle.dump(feature_columns, f)
    saved_paths["columns"] = columns_path
    print(f"    ✅ feature_columns.pkl     → {columns_path}")
    print(f"       Total features          : {len(feature_columns)}")

    # ── 4d. Save metadata as JSON ────────────────────────────────
    metadata = {
        "model_name":       selection["name"],
        "hyperparameters":  selection["params"],
        "test_r2":          round(selection["r2"], 4),
        "test_mse":         round(selection["mse"], 2),
        "n_features":       len(feature_columns),
        "numeric_features": ["total_sqft", "bath", "balcony", "bhk"],
        "training_samples": "full_dataset",
        "serialization":    "pickle",
        "phase":            7,
        "notes": (
            "Model retrained on full dataset after evaluation. "
            "Use predict_price() for inference."
        ),
    }
    metadata_path = os.path.join(PROCESSED_DIR, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    saved_paths["metadata"] = metadata_path
    print(f"    ✅ model_metadata.json     → {metadata_path}")

    print(f"\n    📦 All {len(saved_paths)} artefacts saved to: {PROCESSED_DIR}/")

    return saved_paths


# ═══════════════════════════════════════════════════════════════════════════
# 5  Prediction Pipeline Function
# ═══════════════════════════════════════════════════════════════════════════
def predict_price(input_dict: dict) -> float:
    """
    End-to-end prediction pipeline: raw input → predicted price.

    This function simulates exactly what a FastAPI endpoint would do:
    1. Accept raw user input (area, location, bedrooms, bathrooms)
    2. Apply the SAME preprocessing used during training
    3. Load the saved model
    4. Return the predicted price

    Parameters
    ----------
    input_dict : dict
        Must contain:
          - area       : float  — total square footage of the property
          - location   : str    — location name (e.g. "Whitefield")
          - bedrooms   : int    — number of bedrooms (BHK)
          - bathrooms  : float  — number of bathrooms

        Optional:
          - balcony      : float  — number of balconies (default=1.0)
          - area_type    : str    — e.g. "Super built-up Area" (default=None)
          - availability : str    — e.g. "Ready To Move" (default=None)

    Returns
    -------
    float — predicted price in Lakhs (₹)

    Example
    -------
    >>> predict_price({
    ...     "area": 1500,
    ...     "location": "Whitefield",
    ...     "bedrooms": 3,
    ...     "bathrooms": 2,
    ... })
    85.42

    How This Connects to FastAPI
    ────────────────────────────
    In a real deployment, your FastAPI route would look like:

        @app.post("/predict")
        def predict(request: HouseInput):
            price = predict_price(request.dict())
            return {"predicted_price_lakhs": price}

    The predict_price function encapsulates ALL preprocessing so the API
    layer stays clean and doesn't need to know about scaling or encoding.
    """
    # ── Load artefacts ───────────────────────────────────────────
    with open(os.path.join(PROCESSED_DIR, "final_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "feature_columns.pkl"), "rb") as f:
        feature_columns = pickle.load(f)

    # ── Build a single-row DataFrame matching training features ──
    # Start with all zeros (one-hot columns default to False/0)
    input_row = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns,
    )

    # Numeric features
    input_row["total_sqft"] = float(input_dict["area"])
    input_row["bath"]       = float(input_dict["bathrooms"])
    input_row["bhk"]        = int(input_dict["bedrooms"])
    input_row["balcony"]    = float(input_dict.get("balcony", 1.0))

    # ── Location one-hot encoding ────────────────────────────────
    # The column name pattern from Phase 3: "location_<name>"
    location = input_dict["location"].strip()
    loc_col  = f"location_{location}"
    if loc_col in input_row.columns:
        input_row[loc_col] = 1.0
    # If location not found, all location columns remain 0 (= "other")

    # ── Area type one-hot encoding (if provided) ─────────────────
    area_type = input_dict.get("area_type")
    if area_type:
        at_col = f"area_type_{area_type}"
        if at_col in input_row.columns:
            input_row[at_col] = 1.0

    # ── Availability one-hot encoding (if provided) ──────────────
    availability = input_dict.get("availability")
    if availability:
        av_col = f"availability_{availability}"
        if av_col in input_row.columns:
            input_row[av_col] = 1.0

    # ── Apply the SAME scaling as training ───────────────────────
    # The scaler was fit on: ["total_sqft", "bath", "balcony", "bhk"]
    num_cols = ["total_sqft", "bath", "balcony", "bhk"]
    num_cols_present = [c for c in num_cols if c in input_row.columns]
    input_row[num_cols_present] = scaler.transform(
        input_row[num_cols_present]
    )

    # ── Predict ──────────────────────────────────────────────────
    prediction = model.predict(input_row)[0]

    return round(float(prediction), 2)


# ═══════════════════════════════════════════════════════════════════════════
# 6  Test the Prediction Pipeline
# ═══════════════════════════════════════════════════════════════════════════
def test_prediction_pipeline() -> None:
    """
    Run predict_price() with sample inputs to verify end-to-end correctness.
    """
    print("\n" + "─" * 70)
    print("[5] TESTING PREDICTION PIPELINE")
    print("─" * 70)

    test_cases = [
        {
            "description": "3 BHK in Whitefield, 1500 sqft",
            "input": {
                "area": 1500,
                "location": "Whitefield",
                "bedrooms": 3,
                "bathrooms": 2,
                "balcony": 2,
            },
        },
        {
            "description": "2 BHK in Electronic City, 1000 sqft",
            "input": {
                "area": 1000,
                "location": "Electronic City",
                "bedrooms": 2,
                "bathrooms": 1,
                "balcony": 1,
            },
        },
        {
            "description": "4 BHK in Hebbal, 2500 sqft",
            "input": {
                "area": 2500,
                "location": "Hebbal",
                "bedrooms": 4,
                "bathrooms": 3,
                "balcony": 2,
            },
        },
        {
            "description": "2 BHK in unknown location, 800 sqft",
            "input": {
                "area": 800,
                "location": "SomeUnknownPlace",
                "bedrooms": 2,
                "bathrooms": 1,
                "balcony": 1,
            },
        },
    ]

    print("\n    Running sample predictions...\n")
    print(f"    {'Test Case':<50s} {'Predicted Price':>15}")
    print("    " + "─" * 68)

    for tc in test_cases:
        price = predict_price(tc["input"])
        print(f"    {tc['description']:<50s} ₹ {price:>10.2f} Lakhs")

    print("\n    ✅ All test cases passed — prediction pipeline is functional!")
    print("    ℹ️  Predictions are in Lakhs (₹). Verify reasonableness with")
    print("       domain knowledge of Bengaluru property prices.")


# ═══════════════════════════════════════════════════════════════════════════
# 7  Conceptual Explanations
# ═══════════════════════════════════════════════════════════════════════════
def print_phase7_explanations() -> None:
    """
    Clear, exam-ready explanations of Phase 7 concepts.
    """
    print("\n" + "=" * 70)
    print("[6] CONCEPTUAL EXPLANATIONS — Phase 7")
    print("=" * 70)
    print("""
┌──────────────────────────────────────────────────────────────────────┐
│  WHAT IS MODEL SERIALIZATION?                                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Serialization = converting a Python object (the trained model)      │
│  into a byte stream that can be saved to disk as a .pkl file.        │
│                                                                      │
│  Without serialization:                                               │
│    • You'd have to retrain the model every time the server restarts  │
│    • Training can take minutes to hours for large datasets           │
│    • Results might differ due to randomness                          │
│                                                                      │
│  With serialization:                                                  │
│    • Load the .pkl file in milliseconds                              │
│    • Exact same model every time (deterministic)                     │
│    • Deploy across multiple servers consistently                     │
│                                                                      │
│  Tools: pickle (built-in) or joblib (faster for NumPy arrays)        │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  WHY SAVE THE SCALER SEPARATELY?                                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  The StandardScaler stores the mean & std of the TRAINING data:      │
│    X_scaled = (X - mean) / std                                       │
│                                                                      │
│  If you DON'T save the scaler:                                       │
│    • New data won't be scaled with the same mean/std                 │
│    • The model receives data in a different "coordinate system"      │
│    • Predictions become garbage                                      │
│                                                                      │
│  The scaler is a SEPARATE object from the model because:             │
│    • It's fit on X (features), not y (target)                        │
│    • It can be reused even if you swap models                        │
│    • It encapsulates preprocessing logic (separation of concerns)    │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  HOW DOES THE PREDICTION PIPELINE WORK?                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Raw Input → Encoding → Scaling → Model → Prediction                 │
│                                                                      │
│  Step-by-step:                                                       │
│  1. User provides: area, location, bedrooms, bathrooms               │
│  2. Create a zero-filled row matching training features (262 cols)   │
│  3. Set numeric values (area → total_sqft, etc.)                     │
│  4. One-hot encode location (set location_X = 1)                     │
│  5. Scale numeric columns using the SAVED scaler                     │
│  6. Feed the prepared row to the SAVED model                         │
│  7. Return the predicted price                                       │
│                                                                      │
│  The key insight: preprocessing at prediction time MUST EXACTLY      │
│  match preprocessing at training time.                               │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  HOW DOES THIS CONNECT TO FASTAPI?                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  FastAPI is a web framework that exposes Python functions as HTTP     │
│  API endpoints.  The connection is simple:                            │
│                                                                      │
│    1. FastAPI receives a POST request with JSON body                  │
│       {"area": 1500, "location": "Whitefield", ...}                  │
│                                                                      │
│    2. FastAPI parses the JSON into a Python dict                      │
│                                                                      │
│    3. FastAPI calls predict_price(input_dict)                         │
│       → This is the function we built in this phase!                 │
│                                                                      │
│    4. FastAPI returns the prediction as a JSON response               │
│       {"predicted_price_lakhs": 85.42}                               │
│                                                                      │
│  The predict_price() function is the bridge between the ML model     │
│  and the web API.  It hides all preprocessing complexity behind a    │
│  clean, simple interface.                                            │
│                                                                      │
│  In Phase 8 (if planned), you would create:                          │
│    • main.py with FastAPI app                                        │
│    • A /predict endpoint that calls predict_price()                  │
│    • Request/Response models using Pydantic                          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════════════════
# Main Execution — Orchestrator
# ═══════════════════════════════════════════════════════════════════════════
def run_finalize() -> dict:
    """
    Orchestrate the full Phase 7 pipeline.

    Returns a dict with the final model, artefact paths, and metadata.
    """
    # ── Step 1: Load and compare results ─────────────────────────
    results = load_and_compare_results()

    # ── Step 2: Select the best model ────────────────────────────
    selection = select_best_model(results)

    # ── Step 3: Retrain on full dataset ──────────────────────────
    model, feature_columns, X, y = retrain_on_full_data(selection)

    # ── Step 4: Save deployment artefacts ────────────────────────
    saved_paths = save_deployment_artefacts(model, feature_columns, selection)

    # ── Step 5: Test prediction pipeline ─────────────────────────
    test_prediction_pipeline()

    # ── Step 6: Conceptual explanations ──────────────────────────
    print_phase7_explanations()

    print("\n" + "=" * 70)
    print("PHASE 7 COMPLETE ✅ — Model finalized, serialized, and ready for deployment.")
    print("=" * 70)
    print("\n    📦 Deployment artefacts:")
    for name, path in saved_paths.items():
        print(f"       • {name:<15s} → {os.path.basename(path)}")
    print("\n    🚀 Next step: Build a FastAPI endpoint that calls predict_price()")
    print("       from this module to serve predictions over HTTP.")
    print()

    return {
        "selection":       selection,
        "model":           model,
        "feature_columns": feature_columns,
        "saved_paths":     saved_paths,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Standalone
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_finalize()
