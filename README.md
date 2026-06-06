# Marketing Attribution & Budget Optimization

Marketing Mix Modeling (MMM) and multi-touch attribution analysis on multi-region eCommerce advertising data.

## Overview

This project analyzes marketing channel effectiveness across ~100 eCommerce brands using the [Conjura Multi-Region MMM Dataset](https://figshare.com/articles/dataset/Multi-Region_Marketing_Mix_Modeling_MMM_Dataset_for_Several_eCommerce_Brands/25314841). It combines:

- **Macro-level MMM**: Multivariate regression with Ridge/Lasso regularization to quantify channel-level ROI
- **Micro-level attribution**: Simulated user journey data with first-touch, last-touch, linear, time-decay, Shapley Value, and Markov chain attribution models
- **Budget optimization**: Constrained optimization to recommend optimal spend allocation

## Dataset

| Property | Value |
|----------|-------|
| Source | figshare (via GitHub mirror) |
| Records | 132,759 rows x 50 columns |
| Brands | ~100 eCommerce brands |
| Territories | 19 (including "All Territories") |
| Date Range | 2019-07 to 2024-06 |
| Channels | Google (5 sub-channels), Meta (3), Tiktok, plus organic |

## Architecture

```
marketing-attribution-mmm/
├── scripts/
│   ├── preprocess.py              # Polars ETL: missing values, CTR/CPM/ROAS, adstock lags
│   ├── mmm_model.py               # OLS + Ridge + Lasso, VIF, Durbin-Watson, residuals
│   ├── generate_touchpoints.py    # Simulated user journey generation (50k users)
│   ├── multi_touch_attribution.py # 6 attribution models comparison
│   └── budget_optimizer.py        # scipy.optimize budget reallocation
├── notebooks/
│   └── 01_eda.ipynb               # Exploratory data analysis
├── dashboard/
│   └── app.py                     # Streamlit interactive dashboard
├── tests/
├── data/
│   ├── raw/                       # Conjura MMM dataset
│   └── processed/                 # Cleaned Parquet + model outputs
├── reports/
│   └── images/                    # Generated charts
├── config.py                      # Centralized configuration
├── Makefile                       # Workflow orchestration
└── requirements.txt
```

## Quick Start

```bash
# Setup
make setup

# Run full pipeline
make all

# Launch dashboard
make dashboard

# Run tests
make test

# Local quality gates
make verify
```

## Key Results

### MMM (Marketing Mix Modeling)
- **Best model**: Ridge regression with adstocked spend features
- **Top performing channels** (by coefficient magnitude):
  - Google Paid Search
  - Meta Facebook
  - Google Shopping
- **Diagnostics**: VIF < 5 for all channels (no severe multicollinearity)

### Attribution Model Comparison
| Channel | First-Touch | Last-Touch | Linear | Shapley | Markov |
|---------|------------|-----------|--------|---------|--------|
| Google Paid Search | 17.8% | 16.8% | 17.6% | 16.6% | 19.4% |
| Meta Facebook | 14.6% | 16.0% | 14.3% | 14.0% | 15.1% |
| Google Shopping | 14.2% | 13.1% | 13.6% | 12.4% | 14.8% |

### Budget Optimization
- Reallocating existing budget across channels can improve predicted revenue by **~130%**
- Under a +20% budget increase scenario, optimal allocation shifts more spend to Google Paid Search and Meta Facebook

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data Processing | Polars, DuckDB |
| Modeling | statsmodels, scikit-learn, scipy |
| Visualization | Plotly, Matplotlib, Streamlit |
| Testing | pytest, ruff |

## License

MIT
