"""
FastAPI Backend Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import warnings

warnings.filterwarnings("ignore")

from backend.app.api.endpoints import router
from backend.app.services.ml_registry import _load_or_train_all_models

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

app.include_router(router)

@app.on_event("startup")
def startup():
    _load_or_train_all_models()
