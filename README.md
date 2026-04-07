# 🏠 House Price Prediction using Regularized Regression Models

> **CSE275 Project** — A complete, modular ML pipeline for predicting Bengaluru house prices using Linear, Ridge, Lasso, and ElasticNet regression models.

---

## 📋 Project Overview

This project implements an end-to-end machine learning pipeline that:
- Loads and inspects raw Bengaluru housing data (13,320 records)
- Performs exploratory data analysis with visualizations
- Cleans data with outlier removal, feature engineering, and one-hot encoding
- Trains a Linear Regression baseline model
- Trains Ridge (L2), Lasso (L1), and ElasticNet regularized models
- Compares all models with R², MSE, and MAE metrics
- Generates publication-quality plots and a detailed HTML report

### 🏆 Best Model: Ridge Regression (R² = 0.7151)

---

## 📁 Project Structure

```
CSE275-Project/
├── backend/
│   └── ml_pipeline/
│       ├── 01_data_loading.py        # Phase 1: Load & inspect raw CSV
│       ├── 02_eda.py                 # Phase 2: Exploratory Data Analysis
│       ├── 03_data_cleaning.py       # Phase 3: Cleaning & preprocessing
│       ├── 04_baseline_model.py      # Phase 4: Linear Regression baseline
│       ├── 05_advanced_models.py     # Phase 5: Ridge, Lasso, ElasticNet
│       ├── run_pipeline.py           # Master script — runs all phases
│       ├── __init__.py
│       └── outputs/                  # Generated outputs (gitignored)
│           ├── eda/                  # EDA plots
│           ├── processed/            # Cleaned X, y, scaler
│           ├── baseline/             # Baseline model plots & coefficients
│           └── advanced_models/      # Regularized model comparisons
├── data/
│   └── Bengaluru_House_Data.csv      # Raw dataset (gitignored)
├── docs/
│   └── project_report.html           # Full project report with Mermaid flowcharts
├── frontend/                          # (Planned — future deployment phase)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/CSE275-House-Price-Prediction.git
cd CSE275-House-Price-Prediction
```

### 2. Set Up Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 3. Download the Dataset
Download `Bengaluru_House_Data.csv` from [Kaggle](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data) and place it in the `data/` directory.

### 4. Run the Full Pipeline
```bash
cd backend/ml_pipeline
python run_pipeline.py
```

This will sequentially execute all 5 phases and generate outputs in the `outputs/` directory.

### 5. View the Project Report
Open `docs/project_report.html` in any browser to see the full documentation with flowcharts and embedded analysis.

---

## 📊 Pipeline Phases

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `01_data_loading.py` | Load raw CSV, inspect shape/dtypes/nulls |
| 2 | `02_eda.py` | Price distribution, correlations, scatter plots |
| 3 | `03_data_cleaning.py` | Outlier removal, feature engineering, scaling |
| 4 | `04_baseline_model.py` | Train-test split, Linear Regression, evaluation |
| 5 | `05_advanced_models.py` | Ridge, Lasso, ElasticNet with full comparison |

---

## 📈 Model Comparison Results

| Model | R² Score | MSE | MAE |
|-------|----------|-----|-----|
| Linear Regression | 0.7084 | 3,075.94 | 25.52 |
| **Ridge (α=1.0)** 🏆 | **0.7151** | **3,005.50** | **25.40** |
| Lasso (α=1.0) | 0.6577 | 3,611.19 | 28.63 |
| ElasticNet (α=1.0, ρ=0.5) | 0.5594 | 4,648.49 | 30.80 |

---

## 🛠 Tech Stack

- **Python 3.9+**
- **pandas** — Data manipulation
- **NumPy** — Numerical operations
- **scikit-learn** — ML models and metrics
- **matplotlib** — Visualizations
- **seaborn** — Statistical plots

---

## 📝 License

This project is for academic purposes (CSE275 coursework).
