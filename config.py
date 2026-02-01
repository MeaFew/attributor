"""Marketing Attribution & Budget Optimization — Centralized Configuration."""
from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
IMAGES_DIR = REPORTS_DIR / "images"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data files
RAW_CSV_PATH = RAW_DATA_DIR / "conjura_mmm_data.csv"
DATA_DICT_PATH = RAW_DATA_DIR / "conjura_mmm_data_dictionary.tsv"
CLEANED_PARQUET_PATH = PROCESSED_DATA_DIR / "mmm_cleaned.parquet"

# Simulated touchpoint data
SIMULATED_TOUCHPOINTS_PATH = PROCESSED_DATA_DIR / "simulated_touchpoints.parquet"
SIMULATED_JOURNEYS_PATH = PROCESSED_DATA_DIR / "simulated_journeys.parquet"

# DuckDB
DUCKDB_PATH = PROCESSED_DATA_DIR / "analytics.duckdb"

# Output directories
MODEL_OUTPUT_DIR = PROCESSED_DATA_DIR / "models"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, IMAGES_DIR, MODEL_OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Analysis constants
# ---------------------------------------------------------------------------
# Spend channels (from figshare data)
SPEND_CHANNELS = [
    "google_paid_search_spend",
    "google_shopping_spend",
    "google_pmax_spend",
    "google_display_spend",
    "google_video_spend",
    "meta_facebook_spend",
    "meta_instagram_spend",
    "meta_other_spend",
    "tiktok_spend",
]

CLICK_CHANNELS = [
    "google_paid_search_clicks",
    "google_shopping_clicks",
    "google_pmax_clicks",
    "google_display_clicks",
    "google_video_clicks",
    "meta_facebook_clicks",
    "meta_instagram_clicks",
    "meta_other_clicks",
    "tiktok_clicks",
]

IMPRESSION_CHANNELS = [
    "google_paid_search_impressions",
    "google_shopping_impressions",
    "google_pmax_impressions",
    "google_display_impressions",
    "google_video_impressions",
    "meta_facebook_impressions",
    "meta_instagram_impressions",
    "meta_other_impressions",
    "tiktok_impressions",
]

# Organic / non-paid channels (clicks only)
ORGANIC_CHANNELS = [
    "direct_clicks",
    "branded_search_clicks",
    "organic_search_clicks",
    "email_clicks",
    "referral_clicks",
    "all_other_clicks",
]

# Target variables
TARGET_NEW_CUSTOMERS = "first_purchases"
TARGET_NEW_REVENUE = "first_purchases_original_price"
TARGET_ALL_CUSTOMERS = "all_purchases"
TARGET_ALL_REVENUE = "all_purchases_original_price"

# Simulation parameters for touchpoint data
SIMULATION_PARAMS = {
    "n_users": 50_000,
    "max_touchpoints_per_user": 8,
    "conversion_rate": 0.035,
    "channels": [
        "google_paid_search",
        "google_shopping",
        "google_pmax",
        "google_display",
        "google_video",
        "meta_facebook",
        "meta_instagram",
        "meta_other",
        "tiktok",
        "direct",
        "branded_search",
        "organic_search",
        "email",
        "referral",
    ],
    "channel_weights": {
        "google_paid_search": 0.18,
        "google_shopping": 0.12,
        "google_pmax": 0.10,
        "google_display": 0.08,
        "google_video": 0.06,
        "meta_facebook": 0.15,
        "meta_instagram": 0.10,
        "meta_other": 0.03,
        "tiktok": 0.08,
        "direct": 0.03,
        "branded_search": 0.02,
        "organic_search": 0.02,
        "email": 0.02,
        "referral": 0.01,
    },
    "date_range_days": 365,
}
