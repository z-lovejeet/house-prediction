"""
Phase 8 – FastAPI Backend Integration
=======================================
Exposes the trained ML model from Phase 7 as a RESTful API.

Architecture Overview
─────────────────────
  Next.js Frontend  ──HTTP POST──▶  FastAPI  ──▶  predict_price()  ──▶  Model
       (JSON)                        (API)          (Phase 7)          (.pkl)

How it works:
1. The frontend sends a JSON payload with house details (area, location, etc.)
2. FastAPI validates the payload using a Pydantic schema (HouseInput)
3. The validated data is converted to a dict and passed to predict_price()
4. predict_price() handles ALL preprocessing (encoding, scaling) internally
5. The model returns a prediction in Lakhs (₹)
6. FastAPI wraps the result in a JSON response and sends it back

Why FastAPI?
────────────
• Automatic request validation via Pydantic
• Auto-generated interactive docs (Swagger UI at /docs)
• Async-capable (handles many concurrent requests)
• Type hints = self-documenting code
• Built-in CORS middleware for frontend integration

Usage
─────
    # From project root:
    uvicorn backend.main:app --reload

    # Then open:
    #   http://127.0.0.1:8000       → root health check
    #   http://127.0.0.1:8000/docs  → interactive Swagger docs
"""

import os
import json
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── Import the prediction pipeline from Phase 7 ─────────────────────
# predict_price() encapsulates ALL preprocessing (encoding + scaling)
# and model loading, so the API layer stays clean.
from backend.ml_pipeline.finalize_and_save_model import predict_price


# ═══════════════════════════════════════════════════════════════════════════
# 1  Load Model Metadata (for /info endpoint)
# ═══════════════════════════════════════════════════════════════════════════
METADATA_PATH = os.path.join(
    os.path.dirname(__file__), "ml_pipeline", "outputs", "processed",
    "model_metadata.json"
)

_model_metadata = {}
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r") as f:
        _model_metadata = json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# 2  FastAPI App Initialization
# ═══════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="House Price Prediction API",
    description=(
        "Predicts house prices in Bengaluru based on area, location, "
        "bedrooms, and bathrooms. Built on an ElasticNet regression model "
        "trained on 9,200+ property listings.\n\n"
        "**Model**: ElasticNet (α=0.001, l1_ratio=0.8) — R² = 0.7158\n\n"
        "**Endpoints**:\n"
        "- `GET /` — Health check & API status\n"
        "- `GET /info` — Model metadata & configuration\n"
        "- `POST /predict` — Predict house price from input features\n"
    ),
    version="1.0.0",
    docs_url="/docs",        # Swagger UI
    redoc_url="/redoc",      # ReDoc alternative
)


# ═══════════════════════════════════════════════════════════════════════════
# 3  CORS Middleware — Required for Next.js Frontend
# ═══════════════════════════════════════════════════════════════════════════
#
# Why CORS?
# ─────────
# Browsers enforce the Same-Origin Policy: a frontend at localhost:3000
# CANNOT make requests to an API at localhost:8000 unless the API
# explicitly allows it via CORS headers.
#
# allow_origins=["*"] means "accept requests from ANY origin".
# In production, restrict this to your actual frontend domain.
#
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # ← Allow all origins (dev mode)
    allow_credentials=True,
    allow_methods=["*"],              # ← Allow GET, POST, etc.
    allow_headers=["*"],              # ← Allow all headers
)


# ═══════════════════════════════════════════════════════════════════════════
# 4  Request & Response Schemas (Pydantic)
# ═══════════════════════════════════════════════════════════════════════════
#
# Pydantic models give us:
# • Automatic JSON parsing & type coercion
# • Validation with clear error messages
# • Auto-generated OpenAPI/Swagger docs
#

class HouseInput(BaseModel):
    """
    Input schema for house price prediction.

    These are the raw features a user provides — the API handles all
    preprocessing (encoding, scaling) internally via predict_price().
    """
    area: float = Field(
        ...,
        gt=0,
        description="Total square footage of the property (e.g. 1500)",
        examples=[1500],
    )
    bedrooms: int = Field(
        ...,
        ge=1,
        le=20,
        description="Number of bedrooms / BHK (e.g. 3)",
        examples=[3],
    )
    bathrooms: int = Field(
        ...,
        ge=1,
        le=20,
        description="Number of bathrooms (e.g. 2)",
        examples=[2],
    )
    location: str = Field(
        ...,
        min_length=1,
        description="Location / neighbourhood name in Bengaluru (e.g. 'Whitefield')",
        examples=["Whitefield"],
    )
    balcony: float = Field(
        default=1.0,
        ge=0,
        description="Number of balconies (optional, default=1)",
        examples=[2],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "area": 1500,
                    "bedrooms": 3,
                    "bathrooms": 2,
                    "location": "Whitefield",
                    "balcony": 2,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for a successful prediction."""
    predicted_price: float = Field(
        description="Predicted price in Lakhs (₹)"
    )
    status: str = Field(
        default="success",
        description="Request status"
    )
    currency: str = Field(
        default="INR (Lakhs)",
        description="Currency unit of the prediction"
    )
    input_received: dict = Field(
        description="Echo of the input that was used for prediction"
    )


class HealthResponse(BaseModel):
    """Response schema for the root health check."""
    status: str
    message: str
    version: str
    docs: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Response schema for model metadata."""
    model_name: str
    hyperparameters: dict
    test_r2: float
    test_mse: float
    n_features: int
    numeric_features: list
    status: str


# ═══════════════════════════════════════════════════════════════════════════
# 5  API Endpoints
# ═══════════════════════════════════════════════════════════════════════════

# ── 5a. Root — Health Check ──────────────────────────────────────────────
@app.get(
    "/",
    response_model=HealthResponse,
    summary="Health Check",
    description="Verify that the API is running and return basic status info.",
    tags=["Status"],
)
def root():
    """
    Root endpoint — confirms the API is alive.

    Returns API status, version, and a link to the interactive docs.
    This is the first thing you should hit to verify the server started.
    """
    return {
        "status": "active",
        "message": "House Price Prediction API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "timestamp": datetime.now().isoformat(),
    }


# ── 5b. Model Info ──────────────────────────────────────────────────────
@app.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Model Information",
    description="Return metadata about the deployed ML model.",
    tags=["Status"],
)
def model_info():
    """
    Returns metadata about the currently deployed model.

    Useful for debugging, documentation, and API versioning.
    Reads from model_metadata.json created in Phase 7.
    """
    if not _model_metadata:
        raise HTTPException(
            status_code=503,
            detail="Model metadata not found. Run Phase 7 first.",
        )

    return {
        "model_name":       _model_metadata.get("model_name", "unknown"),
        "hyperparameters":  _model_metadata.get("hyperparameters", {}),
        "test_r2":          _model_metadata.get("test_r2", 0),
        "test_mse":         _model_metadata.get("test_mse", 0),
        "n_features":       _model_metadata.get("n_features", 0),
        "numeric_features": _model_metadata.get("numeric_features", []),
        "status":           "model_loaded",
    }


# ── 5c. Predict — The Core Endpoint ─────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict House Price",
    description=(
        "Submit house details and receive an estimated price prediction.\n\n"
        "**How it works internally**:\n"
        "1. FastAPI validates your JSON input via the HouseInput schema\n"
        "2. The input is converted to a Python dict\n"
        "3. `predict_price()` from Phase 7 handles:\n"
        "   - One-hot encoding the location\n"
        "   - Scaling numeric features using the saved StandardScaler\n"
        "   - Running the ElasticNet model\n"
        "4. The predicted price (in Lakhs ₹) is returned\n"
    ),
    tags=["Prediction"],
)
def predict(house: HouseInput):
    """
    Predict house price for the given input.

    The complete flow:
        JSON body → Pydantic validation → dict → predict_price() → response

    predict_price() internally:
        1. Loads final_model.pkl, scaler.pkl, feature_columns.pkl
        2. Builds a 262-column feature row (mostly zeros for one-hot)
        3. Sets numeric values + activates the correct location column
        4. Scales numeric features using the saved StandardScaler
        5. Runs model.predict() and returns the result
    """
    try:
        # Convert Pydantic model → dict (matching predict_price's expected keys)
        input_dict = {
            "area":      house.area,
            "bedrooms":  house.bedrooms,
            "bathrooms": house.bathrooms,
            "location":  house.location,
            "balcony":   house.balcony,
        }

        # Call the Phase 7 prediction pipeline
        predicted_price = predict_price(input_dict)

        return {
            "predicted_price": predicted_price,
            "status":          "success",
            "currency":        "INR (Lakhs)",
            "input_received":  input_dict,
        }

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model artefacts not found: {e}. "
                "Ensure Phase 7 has been run to generate "
                "final_model.pkl, scaler.pkl, and feature_columns.pkl."
            ),
        )
    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required field: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )
