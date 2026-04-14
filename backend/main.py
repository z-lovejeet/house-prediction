"""
Phase 8 – FastAPI Backend (Enhanced with Multi-Model Support)
==============================================================
Exposes trained ML models from the pipeline as a RESTful API.
Supports model selection (Linear, Ridge, Lasso, ElasticNet) and
comparison across all models for the same input.

Endpoints
─────────
  GET  /            → Health check
  GET  /models      → All available models with R², MSE, hyperparams
  POST /predict     → Predict with a specific model (default: best)
  POST /compare     → Run ALL models on the same input, return comparison
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════
BASE_DIR      = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(BASE_DIR, "ml_pipeline", "outputs", "processed")
OPT_DIR       = os.path.join(BASE_DIR, "ml_pipeline", "outputs", "optimization")
ADV_DIR       = os.path.join(BASE_DIR, "ml_pipeline", "outputs", "advanced_models")

# ═══════════════════════════════════════════════════════════════════════════
# 1  Load / Train All Models at Startup
# ═══════════════════════════════════════════════════════════════════════════

# Model registry: name → { model, r2, mse, params, description }
MODEL_REGISTRY = {}

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


# Determine the best model
def _get_best_model_key():
    return max(MODEL_REGISTRY, key=lambda k: MODEL_REGISTRY[k]["r2"])


# Load shared artefacts
def _load_shared_artefacts():
    with open(os.path.join(PROCESSED_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "feature_columns.pkl"), "rb") as f:
        feature_columns = pickle.load(f)
    return scaler, feature_columns

_scaler = None
_feature_columns = None


def _predict_with_model(model, input_dict):
    """Run prediction using a specific model object."""
    global _scaler, _feature_columns
    if _scaler is None:
        _scaler, _feature_columns = _load_shared_artefacts()

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


# ═══════════════════════════════════════════════════════════════════════════
# 2  FastAPI App
# ═══════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="House Price Prediction API",
    description=(
        "Multi-model house price prediction for Bengaluru.\n\n"
        "**Models**: Linear Regression, Ridge, Lasso, ElasticNet\n\n"
        "- `GET /models` — All models with R², MSE, coefficients\n"
        "- `POST /predict` — Predict with a specific model\n"
        "- `POST /compare` — Compare all 4 models on same input\n"
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    _load_or_train_all_models()


# ═══════════════════════════════════════════════════════════════════════════
# 3  Schemas
# ═══════════════════════════════════════════════════════════════════════════

class HouseInput(BaseModel):
    area: float = Field(..., gt=0, description="Total sq.ft", examples=[1500])
    bedrooms: int = Field(..., ge=1, le=20, examples=[3])
    bathrooms: int = Field(..., ge=1, le=20, examples=[2])
    location: str = Field(..., min_length=1, examples=["Whitefield"])
    balcony: float = Field(default=1.0, ge=0, examples=[2])
    model: Optional[str] = Field(
        default=None,
        description="Model to use: linear, ridge, lasso, elasticnet. Default = best.",
        examples=["elasticnet"],
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4  Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Status"])
def root():
    return {
        "status": "active",
        "message": "House Price Prediction API v2.0",
        "models_loaded": len(MODEL_REGISTRY),
        "best_model": _get_best_model_key(),
        "docs": "/docs",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/models", tags=["Models"])
def list_models():
    """Return all available models with their metrics and coefficients."""
    best = _get_best_model_key()
    result = []
    for key, info in MODEL_REGISTRY.items():
        result.append({
            "key":              key,
            "name":             key.replace("_", " ").title().replace("Elasticnet", "ElasticNet")
                                   .replace("Linear", "Linear Regression"),
            "r2":               info["r2"],
            "mse":              info["mse"],
            "params":           info["params"],
            "description":      info["description"],
            "coefficients":     info["coefficients"],
            "non_zero_features": info["non_zero_features"],
            "total_features":   info["total_features"],
            "is_best":          key == best,
        })
    # Sort by R² descending
    result.sort(key=lambda x: x["r2"], reverse=True)
    return {"models": result, "best_model": best}


@app.post("/predict", tags=["Prediction"])
def predict(house: HouseInput):
    """Predict using a specific model or the best model."""
    model_key = house.model or _get_best_model_key()

    if model_key not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}",
        )

    try:
        info = MODEL_REGISTRY[model_key]
        input_dict = {
            "area": house.area,
            "bedrooms": house.bedrooms,
            "bathrooms": house.bathrooms,
            "location": house.location,
            "balcony": house.balcony,
        }

        predicted_price = _predict_with_model(info["model"], input_dict)

        return {
            "predicted_price": predicted_price,
            "model_used":      model_key,
            "model_r2":        info["r2"],
            "model_mse":       info["mse"],
            "status":          "success",
            "currency":        "INR (Lakhs)",
            "input_received":  input_dict,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/compare", tags=["Prediction"])
def compare_all(house: HouseInput):
    """Run ALL models on the same input and return a comparison."""
    input_dict = {
        "area": house.area,
        "bedrooms": house.bedrooms,
        "bathrooms": house.bathrooms,
        "location": house.location,
        "balcony": house.balcony,
    }

    best_key = _get_best_model_key()
    results = []

    for key, info in MODEL_REGISTRY.items():
        try:
            price = _predict_with_model(info["model"], input_dict)
        except Exception:
            price = None

        results.append({
            "model":           key,
            "name":            key.replace("_", " ").title().replace("Elasticnet", "ElasticNet")
                                  .replace("Linear", "Linear Regression"),
            "predicted_price": price,
            "r2":              info["r2"],
            "mse":             info["mse"],
            "params":          info["params"],
            "coefficients":    info["coefficients"],
            "non_zero_features": info["non_zero_features"],
            "is_best":         key == best_key,
        })

    results.sort(key=lambda x: x["r2"], reverse=True)

    return {
        "status": "success",
        "input": input_dict,
        "comparisons": results,
        "best_model": best_key,
    }
