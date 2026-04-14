import os
import pickle
import numpy as np
import pandas as pd
from backend.app.core.config import PROCESSED_DIR, OPT_DIR

MODEL_REGISTRY = {}
_scaler = None
_feature_columns = None

def _load_or_train_all_models():
    """Load all 4 model variants. Train + save if pkl files don't exist."""
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

    # Load shared artefacts
    X = pd.read_csv(os.path.join(PROCESSED_DIR, "X_processed.csv"))
    y = pd.read_csv(os.path.join(PROCESSED_DIR, "y_target.csv")).squeeze()

    # Read Phase 6 results for R²/MSE
    opt_df = pd.read_csv(os.path.join(OPT_DIR, "before_vs_after.csv"))

    # Model definitions: (key, class, kwargs, row_name)
    model_defs = [
        ("linear",     LinearRegression, {},
         "Linear Regression"),
        ("ridge",      Ridge,            {"alpha": 1.0},
         "Ridge"),
        ("lasso",      Lasso,            {"alpha": 0.01, "max_iter": 10000},
         "Lasso"),
        ("elasticnet", ElasticNet,       {"alpha": 0.001, "l1_ratio": 0.8, "max_iter": 10000},
         "ElasticNet"),
    ]

    for key, cls, kwargs, row_name in model_defs:
        pkl_path = os.path.join(PROCESSED_DIR, f"model_{key}.pkl")

        # Try to load existing pkl
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                model = pickle.load(f)
        else:
            # Train on full data and save
            model = cls(**kwargs)
            model.fit(X, y)
            with open(pkl_path, "wb") as f:
                pickle.dump(model, f)

        # Get metrics from Phase 6 results
        row = opt_df[opt_df["Model"] == row_name]
        if not row.empty:
            r2  = round(float(row["After R²"].values[0]), 4)
            mse = round(float(row["After MSE"].values[0]), 2)
            raw_params = str(row["Best α"].values[0])
            best_params = "None" if raw_params in ("nan", "N/A", "None") else raw_params
        else:
            r2, mse, best_params = 0, 0, "None"

        # Extract coefficient info
        coeffs = {}
        feature_names = list(X.columns)
        numeric_features = ["total_sqft", "bath", "balcony", "bhk"]
        for feat in numeric_features:
            idx = feature_names.index(feat)
            coeffs[feat] = round(float(model.coef_[idx]), 4)

        # Count non-zero coefficients (for sparsity)
        non_zero = int(np.sum(np.abs(model.coef_) > 1e-10))

        MODEL_REGISTRY[key] = {
            "model":       model,
            "r2":          r2,
            "mse":         mse,
            "params":      best_params,
            "coefficients": coeffs,
            "non_zero_features": non_zero,
            "total_features": len(feature_names),
            "description": {
                "linear":     "No regularization. Uses all features with equal weight.",
                "ridge":      "L2 regularization. Shrinks coefficients but keeps all features.",
                "lasso":      "L1 regularization. Drives some coefficients to exactly zero (feature selection).",
                "elasticnet": "L1 + L2 hybrid. Combines feature selection with coefficient shrinkage.",
            }[key],
        }

    print(f"[OK] Loaded {len(MODEL_REGISTRY)} models into registry")


def get_best_model_key():
    return max(MODEL_REGISTRY, key=lambda k: MODEL_REGISTRY[k]["r2"])


def load_shared_artefacts():
    with open(os.path.join(PROCESSED_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "feature_columns.pkl"), "rb") as f:
        feature_columns = pickle.load(f)
    return scaler, feature_columns


def predict_with_model(model, input_dict):
    """Run prediction using a specific model object."""
    global _scaler, _feature_columns
    if _scaler is None:
        _scaler, _feature_columns = load_shared_artefacts()

    # Build feature row
    input_row = pd.DataFrame(
        np.zeros((1, len(_feature_columns))),
        columns=_feature_columns,
    )

    input_row["total_sqft"] = float(input_dict["area"])
    input_row["bath"]       = float(input_dict["bathrooms"])
    input_row["bhk"]        = int(input_dict["bedrooms"])
    input_row["balcony"]    = float(input_dict.get("balcony", 1.0))

    # Location one-hot
    location = input_dict["location"].strip()
    loc_col = f"location_{location}"
    if loc_col in input_row.columns:
        input_row[loc_col] = 1.0

    # Scale numeric columns
    num_cols = ["total_sqft", "bath", "balcony", "bhk"]
    num_cols_present = [c for c in num_cols if c in input_row.columns]
    input_row[num_cols_present] = _scaler.transform(input_row[num_cols_present])

    prediction = model.predict(input_row)[0]
    return round(float(prediction), 2)
