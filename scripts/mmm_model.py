"""Marketing Mix Modeling (MMM) with OLS, Ridge, and Lasso."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
repo_root = Path(__file__).parents[1].resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from config import (
    CLEANED_PARQUET_PATH,
    SPEND_CHANNELS,
    TARGET_NEW_REVENUE,
    MODEL_OUTPUT_DIR,
    IMAGES_DIR,
)


def prepare_features(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build feature matrix for MMM."""
    # Base features: adstocked spend
    feature_cols = [c.replace("_spend", "_adstock") for c in SPEND_CHANNELS]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Temporal features
    df = df.with_columns(
        pl.col("date_day").cast(pl.Int64).alias("trend"),  # days since epoch as proxy
    )
    # Normalize trend to start at 0
    min_trend = df["trend"].min()
    df = df.with_columns((pl.col("trend") - min_trend).alias("trend"))

    # Seasonality: sine/cosine of month
    df = df.with_columns(
        (2 * np.pi * pl.col("month") / 12).sin().alias("month_sin"),
        (2 * np.pi * pl.col("month") / 12).cos().alias("month_cos"),
        pl.col("is_weekend").cast(pl.Float64).alias("is_weekend"),
    )

    temporal_cols = ["trend", "month_sin", "month_cos", "is_weekend"]
    all_feature_cols = feature_cols + temporal_cols

    # Build matrix
    X = df.select(all_feature_cols).to_numpy()
    y = df.select(TARGET_NEW_REVENUE).to_numpy().ravel()

    return X, y, all_feature_cols


def fit_ols(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    """Fit OLS with statsmodels for diagnostics."""
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_const).fit()

    # VIF
    vif_data = []
    for i, name in enumerate(["const"] + feature_names):
        if i == 0:
            continue
        vif = variance_inflation_factor(X_const, i)
        vif_data.append({"feature": name, "vif": round(float(vif), 2)})

    # Durbin-Watson
    dw = durbin_watson(model.resid)

    # Coefficients (skip const for channel-level interpretation)
    coefs = {}
    for name, coef, pval in zip(feature_names, model.params[1:], model.pvalues[1:]):
            coefs[name] = {
                "coef": round(float(coef), 4),
                "pvalue": round(float(pval), 4),
                "significant": bool(pval < 0.05),
            }

    return {
        "model": "OLS",
        "r2": round(float(model.rsquared), 4),
        "adj_r2": round(float(model.rsquared_adj), 4),
        "aic": round(float(model.aic), 2),
        "bic": round(float(model.bic), 2),
        "durbin_watson": round(float(dw), 4),
        "vif": vif_data,
        "coefficients": coefs,
        "residuals": model.resid.tolist(),
    }


def fit_ridge(X: np.ndarray, y: np.ndarray, feature_names: list[str], alpha: float = 1.0) -> dict:
    """Fit Ridge regression."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)

    # Convert scaled coefs back to original scale
    coefs_original = model.coef_ / scaler.scale_
    intercept_original = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)

    coefs = {}
    for name, coef in zip(feature_names, coefs_original):
        coefs[name] = {"coef": round(float(coef), 4)}

    return {
        "model": f"Ridge(alpha={alpha})",
        "r2": round(float(r2_score(y, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y, y_pred)), 2),
        "coefficients": coefs,
        "intercept": round(float(intercept_original), 2),
    }


def fit_lasso(X: np.ndarray, y: np.ndarray, feature_names: list[str], alpha: float = 0.1) -> dict:
    """Fit Lasso regression."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)

    coefs_original = model.coef_ / scaler.scale_
    intercept_original = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)

    coefs = {}
    for name, coef in zip(feature_names, coefs_original):
        coefs[name] = {"coef": round(float(coef), 4)}

    return {
        "model": f"Lasso(alpha={alpha})",
        "r2": round(float(r2_score(y, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y, y_pred)), 2),
        "coefficients": coefs,
        "intercept": round(float(intercept_original), 2),
    }


def plot_model_comparison(ols_result: dict, ridge_result: dict, lasso_result: dict,
                          feature_names: list[str], output_dir: Path) -> None:
    """Plot coefficient comparison across models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Only plot spend channel coefficients
    spend_features = [f.replace("_spend", "_adstock") for f in SPEND_CHANNELS]
    spend_features = [f for f in spend_features if f in feature_names]

    x = np.arange(len(spend_features))
    width = 0.25

    ols_vals = [ols_result["coefficients"].get(f, {}).get("coef", 0) for f in spend_features]
    ridge_vals = [ridge_result["coefficients"].get(f, {}).get("coef", 0) for f in spend_features]
    lasso_vals = [lasso_result["coefficients"].get(f, {}).get("coef", 0) for f in spend_features]

    # Clean labels
    labels = [f.replace("_adstock", "").replace("google_", "G_").replace("meta_", "M_") for f in spend_features]

    ax.bar(x - width, ols_vals, width, label="OLS", alpha=0.8)
    ax.bar(x, ridge_vals, width, label="Ridge", alpha=0.8)
    ax.bar(x + width, lasso_vals, width, label="Lasso", alpha=0.8)

    ax.set_xlabel("Channel")
    ax.set_ylabel("Coefficient (Revenue per Spend Unit)")
    ax.set_title("MMM Coefficient Comparison Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5)
    fig.tight_layout()

    out = output_dir / "mmm_coefficient_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved coefficient comparison to {out}")
    plt.close(fig)


def plot_residuals(ols_result: dict, output_dir: Path) -> None:
    """Plot residual diagnostics."""
    residuals = np.array(ols_result["residuals"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Residual histogram
    axes[0].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(0, color="red", linestyle="--")

    # Q-Q plot
    sm.qqplot(residuals, line="45", fit=True, ax=axes[1])
    axes[1].set_title("Q-Q Plot")

    fig.tight_layout()
    out = output_dir / "mmm_residual_diagnostics.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved residual diagnostics to {out}")
    plt.close(fig)


def run_mmm(df: pl.DataFrame, brand_id: str | None = None, territory: str | None = None) -> dict:
    """Run MMM pipeline on selected brand/territory or full dataset."""
    if brand_id:
        df = df.filter(pl.col("organisation_id") == brand_id)
    if territory:
        df = df.filter(pl.col("territory_name") == territory)

    if df.height == 0:
        raise ValueError("No data after filtering")

    print(f"Running MMM on {df.height:,} rows (brand={brand_id}, territory={territory})")

    X, y, feature_names = prepare_features(df)

    ols = fit_ols(X, y, feature_names)
    ridge = fit_ridge(X, y, feature_names, alpha=1.0)
    lasso = fit_lasso(X, y, feature_names, alpha=0.1)

    # Summary
    summary = {
        "sample_size": df.height,
        "brand_id": brand_id,
        "territory": territory,
        "date_range": {
            "min": str(df["date_day"].min()),
            "max": str(df["date_day"].max()),
        },
        "models": {
            "ols": ols,
            "ridge": ridge,
            "lasso": lasso,
        },
    }

    # Save JSON
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = MODEL_OUTPUT_DIR / "mmm_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved results to {json_path}")

    # Plots
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plot_model_comparison(ols, ridge, lasso, feature_names, IMAGES_DIR)
    plot_residuals(ols, IMAGES_DIR)

    return summary


def select_best_brand(df: pl.DataFrame) -> tuple[str, str]:
    """Select brand+territory with most complete data."""
    summary = (
        df.group_by(["organisation_id", "territory_name"])
        .agg([
            pl.len().alias("n_rows"),
            pl.col("total_spend").sum().alias("total_spend"),
        ])
        .sort("total_spend", descending=True)
    )
    best = summary.row(0, named=True)
    return best["organisation_id"], best["territory_name"]


def run_cross_brand_elasticity(df: pl.DataFrame) -> pl.DataFrame:
    """Run MMM per brand-territory and aggregate elasticities."""
    print("Running cross-brand MMM elasticity analysis...")
    groups = df.group_by(["organisation_id", "territory_name"]).agg(pl.count())
    results = []

    for row in groups.iter_rows(named=True):
        brand, territory = row["organisation_id"], row["territory_name"]
        sub = df.filter(
            (pl.col("organisation_id") == brand) &
            (pl.col("territory_name") == territory)
        )
        if sub.height < 100:
            continue
        try:
            X, y, feature_names = prepare_features(sub)
            ridge = fit_ridge(X, y, feature_names, alpha=1.0)
            for feat, info in ridge["coefficients"].items():
                if "adstock" in feat:
                    results.append({
                        "brand": brand,
                        "territory": territory,
                        "channel": feat.replace("_adstock", ""),
                        "elasticity": info["coef"],
                    })
        except Exception as e:
            print(f"  Skip {brand}/{territory}: {e}")
            continue

    result_df = pl.DataFrame(results)
    out = MODEL_OUTPUT_DIR / "cross_brand_elasticities.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(out)
    print(f"  Saved {len(results)} elasticity records to {out}")
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Marketing Mix Modeling")
    parser.add_argument("--brand", type=str, default=None, help="Brand ID")
    parser.add_argument("--territory", type=str, default=None, help="Territory name")
    parser.add_argument("--cross-brand", action="store_true", help="Run cross-brand elasticity")
    args = parser.parse_args()

    df = pl.read_parquet(CLEANED_PARQUET_PATH)

    if args.cross_brand:
        run_cross_brand_elasticity(df)
    else:
        if not args.brand or not args.territory:
            args.brand, args.territory = select_best_brand(df)
            print(f"Auto-selected brand={args.brand}, territory={args.territory}")
        run_mmm(df, brand_id=args.brand, territory=args.territory)
