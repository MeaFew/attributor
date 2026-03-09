"""Budget allocation optimization based on MMM elasticities."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

repo_root = Path(__file__).parents[1].resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from config import (
    CLEANED_PARQUET_PATH,
    MODEL_OUTPUT_DIR,
    SPEND_CHANNELS,
)


def load_mmm_results() -> dict:
    """Load MMM results to extract channel elasticities."""
    path = MODEL_OUTPUT_DIR / "mmm_results.json"
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: MMM results not found at {path}")
        print("Run 'python scripts/mmm_model.py' first to generate results.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse MMM results: {e}")
        print("The file may be corrupted. Re-run 'python scripts/mmm_model.py'.")
        sys.exit(1)
    return data


def extract_params(mmm_data: dict) -> tuple[dict[str, float], float]:
    """Extract per-channel elasticity and Ridge intercept from MMM results."""
    ridge = mmm_data["models"]["ridge"]
    coefs = ridge["coefficients"]
    intercept = ridge.get("intercept", 0.0)

    elasticities = {}
    for ch in SPEND_CHANNELS:
        adstock_key = ch.replace("_spend", "_adstock")
        if adstock_key in coefs:
            elasticities[ch] = coefs[adstock_key]["coef"]
    return elasticities, intercept


def optimize_budget(
    current_spend: dict[str, float],
    elasticities: dict[str, float],
    intercept: float = 0.0,
    total_budget: float | None = None,
    min_spend_ratio: float = 0.1,
    max_spend_ratio: float = 3.0,
) -> dict:
    """Optimize budget allocation using scipy.optimize.

    Objective: maximize total predicted revenue = sum(elasticity_i * spend_i) + intercept
    Constraint: sum(spend_i) = total_budget (if provided)
    Bounds: min_spend_ratio * current <= spend <= max_spend_ratio * current

    Uses the linear formula matching the Ridge model: revenue = sum(coef_i * x_i) + intercept.
    """
    channels = list(current_spend.keys())
    current = np.array([current_spend[c] for c in channels])
    elastic = np.array([elasticities.get(c, 0.0) for c in channels])

    # If no total_budget, keep total constant
    if total_budget is None:
        total_budget = current.sum()

    # Objective: negative revenue using linear formula from Ridge model
    def objective(x):
        return -(np.sum(elastic * x) + intercept)

    # Constraint: sum(x) = total_budget
    def budget_constraint(x):
        return np.sum(x) - total_budget

    # Bounds: each channel can vary between min and max ratio of current
    bounds = []
    for i, c in enumerate(channels):
        min_spend = max(0, current[i] * min_spend_ratio)
        max_spend = current[i] * max_spend_ratio
        bounds.append((min_spend, max_spend))

    constraints = [{"type": "eq", "fun": budget_constraint}]

    result = minimize(
        objective,
        current,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    optimal = result.x
    predicted_revenue_current = float(np.sum(elastic * current) + intercept)
    predicted_revenue_optimal = float(np.sum(elastic * optimal) + intercept)

    warnings = []
    if predicted_revenue_current < 0:
        warnings.append(
            f"Negative predicted baseline revenue (${predicted_revenue_current:,.0f}). "
            "The linear model does not fit this brand well — "
            "optimization results may be unreliable."
        )

    return {
        "channels": channels,
        "current_spend": {c: round(float(v), 2) for c, v in zip(channels, current)},
        "optimal_spend": {c: round(float(v), 2) for c, v in zip(channels, optimal)},
        "current_revenue": round(predicted_revenue_current, 2),
        "optimal_revenue": round(predicted_revenue_optimal, 2),
        "improvement_pct": round(
            float(
                (predicted_revenue_optimal - predicted_revenue_current)
                / abs(predicted_revenue_current)
                * 100
            ),
            2,
        )
        if predicted_revenue_current != 0
        else 0,
        "total_budget": round(float(total_budget), 2),
        "warnings": warnings,
    }


def scenario_analysis(
    current_spend: dict[str, float],
    elasticities: dict[str, float],
    intercept: float = 0.0,
) -> dict:
    """Run multiple budget scenarios."""
    total = sum(current_spend.values())
    scenarios = {}

    # Scenario 1: same budget, reallocate
    scenarios["reallocate"] = optimize_budget(
        current_spend, elasticities, intercept, total_budget=total
    )

    # Scenario 2: +10% budget
    scenarios["increase_10pct"] = optimize_budget(
        current_spend, elasticities, intercept, total_budget=total * 1.1
    )

    # Scenario 3: +20% budget
    scenarios["increase_20pct"] = optimize_budget(
        current_spend, elasticities, intercept, total_budget=total * 1.2
    )

    # Scenario 4: -10% budget
    scenarios["decrease_10pct"] = optimize_budget(
        current_spend, elasticities, intercept, total_budget=total * 0.9
    )

    return scenarios


def main() -> None:
    """Run budget optimization."""
    mmm = load_mmm_results()
    elasticities, intercept = extract_params(mmm)

    # Use average daily spend from the MMM training data as current spend baseline
    try:
        df = pl.read_parquet(CLEANED_PARQUET_PATH)
        current_spend = {}
        for ch in SPEND_CHANNELS:
            if ch not in df.columns:
                current_spend[ch] = 0.0
                continue
            avg = float(df[ch].mean())
            if avg is not None and avg > 0:
                current_spend[ch] = avg
            else:
                # Channel unused for this brand; compute 10th percentile
                # of non-zero spend across ALL brands as fallback
                non_zero = df[ch].filter(pl.col(ch) > 0)
                if non_zero.height > 0:
                    fallback = float(non_zero.quantile(0.1))
                    current_spend[ch] = fallback
                else:
                    current_spend[ch] = 0.0
    except Exception as e:
        print(f"Warning: Could not load cleaned data: {e}")
        print("Using zero baseline for all channels.")
        current_spend = {ch: 0.0 for ch in SPEND_CHANNELS}

    print("Running budget optimization...")
    scenarios = scenario_analysis(current_spend, elasticities, intercept)

    # Save results
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_OUTPUT_DIR / "budget_optimization.json"
    with open(out, "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"  Saved scenarios to {out}")

    # Print summary
    for name, result in scenarios.items():
        print(f"\nScenario: {name}")
        print(f"  Total budget: ${result['total_budget']:,.0f}")
        print(
            f"  Predicted revenue: ${result['current_revenue']:,.0f} -> ${result['optimal_revenue']:,.0f}"
        )
        print(f"  Improvement: {result['improvement_pct']:.1f}%")
        print("  Top reallocation:")
        changes = {
            c: result["optimal_spend"][c] - result["current_spend"][c] for c in result["channels"]
        }
        for ch, delta in sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
            print(
                f"    {ch}: ${result['current_spend'][ch]:,.0f} -> ${result['optimal_spend'][ch]:,.0f} ({delta:+,.0f})"
            )


if __name__ == "__main__":
    main()
