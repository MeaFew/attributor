# Marketing Attribution & Budget Optimization

Marketing Mix Modeling (MMM) and multi-touch attribution analysis on multi-region eCommerce advertising data.

## Overview

This project analyzes marketing channel effectiveness across ~100 eCommerce brands using the [Conjura Multi-Region MMM Dataset](https://figshare.com/articles/dataset/Multi-Region_Marketing_Mix_Modeling_MMM_Dataset_for_Several_eCommerce_Brands/25314841). It combines:

- **Macro-level MMM**: Multivariate regression with Ridge/Lasso regularization to quantify channel-level ROI
- **Micro-level attribution**: Simulated user journey data with first-touch, last-touch, linear, time-decay, Shapley Value, and Markov chain attribution models
- **Budget optimization**: Constrained optimization to recommend optimal spend allocation

## Architecture

```
marketing-attribution-mmm/
├── scripts/
│   ├── preprocess.py              # Data cleaning & feature engineering
│   ├── mmm_model.py               # Marketing Mix Modeling
│   ├── generate_touchpoints.py    # Simulated user journey generation
│   ├── multi_touch_attribution.py # Attribution model comparison
│   └── budget_optimizer.py        # Budget allocation optimization
├── notebooks/
│   └── 01_eda.ipynb               # Exploratory data analysis
├── dashboard/
│   └── app.py                     # Streamlit interactive dashboard
├── tests/
├── data/
│   ├── raw/                       # Conjura MMM dataset
│   └── processed/                 # Cleaned Parquet files
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

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data Processing | Polars, DuckDB |
| Modeling | statsmodels, scikit-learn, scipy |
| Visualization | Plotly, Matplotlib, Streamlit |
| Testing | pytest, ruff |

## License

MIT
