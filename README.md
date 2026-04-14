# House Price Prediction using Regularized Regression

> **CSE275 Project** — End-to-end ML pipeline for predicting Bengaluru house prices with Ridge (L2), Lasso (L1), and ElasticNet regularization, deployed as a full-stack web application.

---

## Problem Statement

> Build a regression model to estimate house prices from location, area, and amenities.
> Optimization objective: Minimize regularized loss (L2).
> Success criterion: High R² score with stable coefficients.

---

## Results

| Model | R² Score | MSE | Regularization | Status |
|-------|----------|-----|----------------|--------|
| Linear Regression | 0.7084 | 3,076 | None | Baseline |
| Ridge (alpha=1) | 0.7151 | 3,006 | L2 | Tuned |
| Lasso (alpha=0.01) | 0.7106 | 3,053 | L1 | Tuned (+8.04%) |
| **ElasticNet (alpha=0.001, l1=0.8)** | **0.7158** | **2,998** | **L1+L2** | **Best (+27.96%)** |

- **Best Model**: ElasticNet — highest R² with stable, sparse coefficients
- **Dataset**: 9,200+ Bengaluru property listings across 178 locations
- **Features**: Area (sqft), Bedrooms (BHK), Bathrooms, Balcony, Location (one-hot encoded)

---

## Project Structure

```
CSE275-Project/
├── backend/
│   ├── main.py                        # FastAPI server (6 endpoints)
│   ├── __init__.py
│   └── ml_pipeline/
│       ├── 01_data_loading.py         # Phase 1: Load & inspect raw CSV
│       ├── 02_eda.py                  # Phase 2: Exploratory Data Analysis
│       ├── 03_data_cleaning.py        # Phase 3: Cleaning & preprocessing
│       ├── 04_baseline_model.py       # Phase 4: Linear Regression baseline
│       ├── 05_advanced_models.py      # Phase 5: Ridge, Lasso, ElasticNet
│       ├── 06_optimization.py         # Phase 6: GridSearchCV + 5-fold CV
│       ├── 07_finalize_and_save_model.py  # Phase 7: Model selection & serialization
│       ├── run_pipeline.py            # Master script — runs all 7 phases
│       └── outputs/
│           ├── eda/                   # EDA plots
│           ├── baseline/              # Baseline model results
│           ├── advanced_models/       # Regularized model comparisons
│           ├── optimization/          # Tuning results & CV analysis
│           └── processed/             # Final model, scaler, feature columns
├── frontend/
│   └── app/
│       ├── page.js                    # Main page
│       ├── globals.css                # Design system
│       └── components/
│           ├── PredictionForm.js       # Core prediction interface
│           ├── ModelSelector.js        # 4-model picker with R² scores
│           ├── ComparisonTable.js      # Side-by-side model comparison
│           ├── ModelCharts.js          # Interactive Chart.js visualizations
│           ├── ExplainabilityChart.js  # Feature contribution breakdown
│           ├── SensitivitySliders.js   # Price sensitivity analysis
│           ├── PredictionHistory.js    # localStorage prediction log
│           ├── LocationHeatmap.js      # Color-coded location price map
│           ├── InputWarnings.js        # Smart data-driven validation
│           ├── PDFExport.js            # Professional PDF report generator
│           ├── InsightsPanel.js        # Bedroom-price behavior explainer
│           └── icons.js               # SVG icon library
├── data/
│   └── Bengaluru_House_Data.csv       # Raw dataset
├── docs/
│   ├── project_report.html            # Full project report
│   └── House Price Prediction — Project Report.pdf
├── requirements.txt
└── README.md
```

---

## ML Pipeline (7 Phases)

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `01_data_loading.py` | Load raw CSV (13,320 records), inspect shape, dtypes, null values |
| 2 | `02_eda.py` | Price distribution, correlation matrix, scatter plots, outlier analysis |
| 3 | `03_data_cleaning.py` | Outlier removal, feature engineering, one-hot encoding, StandardScaler |
| 4 | `04_baseline_model.py` | 80/20 split, Linear Regression baseline, R²/MSE/MAE evaluation |
| 5 | `05_advanced_models.py` | Ridge (L2), Lasso (L1), ElasticNet with default alpha comparison |
| 6 | `06_optimization.py` | GridSearchCV with 5-fold CV for optimal alpha/l1_ratio tuning |
| 7 | `07_finalize_and_save_model.py` | Select best model, retrain on full data, serialize for deployment |

---

## Web Application Features

### Backend (FastAPI — 6 Endpoints)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict price with selected model |
| `/compare` | POST | Run all 4 models on same input |
| `/models` | GET | List all models with R², MSE, coefficients |
| `/explain` | POST | Feature contribution breakdown (waterfall) |
| `/locations/stats` | GET | Average price per location (heatmap data) |
| `/data/stats` | GET | Data distribution stats (validation) |

### Frontend (Next.js — 11 Components)

| Feature | Description |
|---------|-------------|
| **Model Selector** | Pick from 4 models with R² scores, best model highlighted |
| **Compare All** | One-click comparison across all models |
| **Price Breakdown** | Waterfall chart showing each feature's contribution |
| **Sensitivity Analysis** | Real-time price deltas for area/bedroom changes |
| **Prediction History** | LocalStorage-backed log with reuse functionality |
| **Location Heatmap** | 178 locations color-coded by price with search/sort |
| **Input Warnings** | Smart alerts for unusual inputs (cramped sqft/room ratio) |
| **PDF Report Export** | Professional A4 report with prediction details |
| **Interactive Charts** | R² bars, price comparison, coefficient radar, feature doughnut |
| **Insights Panel** | Explains why more bedrooms can decrease price |

---

## Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/z-lovejeet/house-prediction.git
cd house-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run ML Pipeline (Optional — models already included)
```bash
cd backend/ml_pipeline
python run_pipeline.py
```

### 3. Start Backend
```bash
source .venv/bin/activate
uvicorn backend.main:app --port 8000
```

### 4. Start Frontend
```bash
cd frontend
npm install
npm run dev
```

### 5. Open Application
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML** | scikit-learn (Ridge, Lasso, ElasticNet, GridSearchCV) |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Next.js 16, React, Chart.js |
| **Data** | pandas, NumPy, matplotlib, seaborn |
| **Export** | jsPDF (PDF report generation) |
| **Language** | Python 3.9+, JavaScript (ES6+) |

---

## Key Design Decisions

1. **ElasticNet chosen over Ridge** — Though Ridge had similar R² (0.7151 vs 0.7158), ElasticNet provides automatic feature selection (251/262 features used), producing a more interpretable model.

2. **L2 regularization emphasis** — ElasticNet with l1_ratio=0.8 is 80% L1 + 20% L2. Ridge (pure L2) is also deployed as an option. Both minimize regularized loss as required.

3. **Negative BHK coefficient** — The model correctly learns that more bedrooms in the same area indicates cheaper, cramped properties (coefficient: -15.24 Lakhs/bedroom). This is explained in the Insights Panel.

4. **Full deployment** — The frontend goes beyond the assignment requirements to demonstrate production ML deployment skills.

---

## License

This project is for academic purposes (CSE275 coursework).
