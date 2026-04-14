from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime

from backend.app.models.schemas import HouseInput
from backend.app.services.ml_registry import (
    MODEL_REGISTRY,
    get_best_model_key,
    predict_with_model,
    load_shared_artefacts
)
from backend.app.core.config import PROCESSED_DIR

router = APIRouter()

@router.get("/", tags=["Status"])
def root():
    return {
        "status": "active",
        "message": "House Price Prediction API v2.0",
        "models_loaded": len(MODEL_REGISTRY),
        "best_model": get_best_model_key(),
        "docs": "/docs",
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/models", tags=["Models"])
def list_models():
    """Return all available models with their metrics and coefficients."""
    best = get_best_model_key()
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


@router.post("/predict", tags=["Prediction"])
def predict(house: HouseInput):
    """Predict using a specific model or the best model."""
    model_key = house.model or get_best_model_key()

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

        predicted_price = predict_with_model(info["model"], input_dict)

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


@router.post("/compare", tags=["Prediction"])
def compare_all(house: HouseInput):
    """Run ALL models on the same input and return a comparison."""
    input_dict = {
        "area": house.area,
        "bedrooms": house.bedrooms,
        "bathrooms": house.bathrooms,
        "location": house.location,
        "balcony": house.balcony,
    }

    best_key = get_best_model_key()
    results = []

    for key, info in MODEL_REGISTRY.items():
        try:
            price = predict_with_model(info["model"], input_dict)
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


@router.post("/explain", tags=["Analysis"])
def explain_prediction(house: HouseInput):
    """Break down a prediction into per-feature contributions (waterfall)."""
    model_key = house.model or get_best_model_key()
    if model_key not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_key}'")

    from backend.app.services.ml_registry import _scaler, _feature_columns
    if _scaler is None:
        import backend.app.services.ml_registry as ml_reg
        ml_reg._scaler, ml_reg._feature_columns = load_shared_artefacts()
        _scaler, _feature_columns = ml_reg._scaler, ml_reg._feature_columns

    info = MODEL_REGISTRY[model_key]
    model = info["model"]

    # Build feature row (same as predict_with_model)
    input_row = pd.DataFrame(np.zeros((1, len(_feature_columns))), columns=_feature_columns)
    input_row["total_sqft"] = float(house.area)
    input_row["bath"] = float(house.bathrooms)
    input_row["bhk"] = int(house.bedrooms)
    input_row["balcony"] = float(house.balcony)

    loc_col = f"location_{house.location.strip()}"
    if loc_col in input_row.columns:
        input_row[loc_col] = 1.0

    # Scale numerics
    num_cols = ["total_sqft", "bath", "balcony", "bhk"]
    num_present = [c for c in num_cols if c in input_row.columns]
    input_row[num_present] = _scaler.transform(input_row[num_present])

    # Compute per-feature contributions: coeff_i * x_i
    values = input_row.values[0]
    coeffs = model.coef_
    intercept = float(model.intercept_)
    contributions = coeffs * values

    # Aggregate into categories
    feature_labels = {
        "total_sqft": "Area (sqft)",
        "bath": "Bathrooms",
        "bhk": "Bedrooms (BHK)",
        "balcony": "Balcony",
    }

    breakdown = []
    location_contrib = 0.0

    for i, col in enumerate(_feature_columns):
        contrib = float(contributions[i])
        if col in feature_labels:
            breakdown.append({
                "feature": feature_labels[col],
                "contribution": round(contrib, 2),
                "direction": "positive" if contrib >= 0 else "negative",
            })
        elif col.startswith("location_") and abs(contrib) > 0.001:
            location_contrib += contrib

    breakdown.append({
        "feature": f"Location ({house.location})",
        "contribution": round(location_contrib, 2),
        "direction": "positive" if location_contrib >= 0 else "negative",
    })

    breakdown.append({
        "feature": "Base Price (Intercept)",
        "contribution": round(intercept, 2),
        "direction": "positive" if intercept >= 0 else "negative",
    })

    # Sort by absolute contribution
    breakdown.sort(key=lambda x: abs(x["contribution"]), reverse=True)

    predicted = round(float(model.predict(input_row)[0]), 2)

    return {
        "status": "success",
        "model_used": model_key,
        "predicted_price": predicted,
        "breakdown": breakdown,
        "total": predicted,
    }


@router.get("/locations/stats", tags=["Analysis"])
def location_stats():
    """Return average price and count per location for heatmap."""
    X = pd.read_csv(os.path.join(PROCESSED_DIR, "X_processed.csv"))
    y = pd.read_csv(os.path.join(PROCESSED_DIR, "y_target.csv")).squeeze()

    location_cols = [c for c in X.columns if c.startswith("location_")]
    stats = []

    for col in location_cols:
        mask = X[col] == 1
        if mask.sum() > 0:
            name = col.replace("location_", "")
            avg_price = round(float(y[mask].mean()), 2)
            count = int(mask.sum())
            stats.append({
                "location": name,
                "avg_price": avg_price,
                "count": count,
            })

    stats.sort(key=lambda x: x["avg_price"], reverse=True)

    return {
        "status": "success",
        "locations": stats,
        "total_locations": len(stats),
    }


@router.get("/data/stats", tags=["Analysis"])
def data_stats():
    """Return distribution stats for input validation warnings."""
    X = pd.read_csv(os.path.join(PROCESSED_DIR, "X_processed.csv"))

    # Inverse-transform to get original scale
    scaler_path = os.path.join(PROCESSED_DIR, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        sc = pickle.load(f)

    num_cols = ["total_sqft", "bath", "balcony", "bhk"]
    original = X[num_cols].copy()
    original[num_cols] = sc.inverse_transform(original[num_cols])

    return {
        "status": "success",
        "distributions": {
            "area": {
                "min": round(float(original["total_sqft"].min()), 0),
                "max": round(float(original["total_sqft"].max()), 0),
                "mean": round(float(original["total_sqft"].mean()), 0),
                "q25": round(float(original["total_sqft"].quantile(0.25)), 0),
                "q75": round(float(original["total_sqft"].quantile(0.75)), 0),
                "median": round(float(original["total_sqft"].median()), 0),
            },
            "bedrooms": {
                "min": int(original["bhk"].min()),
                "max": int(original["bhk"].max()),
                "mean": round(float(original["bhk"].mean()), 1),
                "median": int(original["bhk"].median()),
                "typical_min": 1,
                "typical_max": 5,
            },
            "bathrooms": {
                "min": int(original["bath"].min()),
                "max": int(original["bath"].max()),
                "mean": round(float(original["bath"].mean()), 1),
                "median": int(original["bath"].median()),
            },
            "sqft_per_bhk": {
                "typical_min": 300,
                "typical_max": 800,
                "ideal": 500,
            },
        },
    }
