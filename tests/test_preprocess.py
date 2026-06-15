"""Unit tests for data preprocessing."""

import sys
from pathlib import Path

import polars as pl

repo_root = Path(__file__).parents[1].resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from config import CLEANED_PARQUET_PATH


def test_cleaned_data_exists():
    assert CLEANED_PARQUET_PATH.exists(), "Cleaned data not found. Run scripts/preprocess.py first."


def test_cleaned_data_schema():
    df = pl.read_parquet(CLEANED_PARQUET_PATH)
    assert df.height > 100_000
    assert "total_spend" in df.columns
    assert "year" in df.columns
    assert "month" in df.columns
    assert "google_paid_search_adstock" in df.columns


def test_no_null_total_spend():
    df = pl.read_parquet(CLEANED_PARQUET_PATH)
    assert df["total_spend"].null_count() == 0


def test_date_range():
    df = pl.read_parquet(CLEANED_PARQUET_PATH)
    min_date = df["date_day"].min()
    max_date = df["date_day"].max()
    assert min_date.year >= 2019
    assert max_date.year <= 2025
