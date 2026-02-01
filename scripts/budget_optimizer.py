"""Budget allocation optimization based on MMM elasticities."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import sys
repo_root = Path(__file__).parents[1].resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from config import (
    MODEL_OUTPUT_DIR,
    SPEND_CHANNELS,
    IMAGES_DIR,
)


def load_mmm_results() -> dict:
    """Load MMM results to extract channel elasticities."""
    path = MODEL_OUTPUT_DIR / "mmm_results.json"
    with open(path) as f:
        data = json.load(f)
    return data


def extract_elasticities(mmm_data: dict) -> dict[str, float]:
    """Extract per-channel elasticity (revenue per spend unit) from Ridge model."""
    ridge = mmm_data["models"]["ridge"]["coefficients"]
    elasticities = {}
    for ch in SPEND_CHANNELS:
        adstock_key = ch.replace("_spend", "_adstock")
        if adstock_key in ridge:
            elasticities[ch] = ridge[adstock_key]["coef"]
    return elasticities


def optimize_budget(
    current_spend: dict[str, float],
    elasticities: dict[str, float],
    total_budget: float | None = None,
    min_spend_ratio: float = 0.1,
    max_spend_ratio: float = 3.0,
) -> dict:
    """Optimize budget allocation using scipy.optimize.

    Objective: maximize total predicted revenue = sum(elasticity_i * spend_i)
    Constraint: sum(spend_i) = total_budget (if provided)
    Bounds: min_spend_ratio * current <= spend <= max_spend_ratio * current
    """
    channels = list(current_spend.keys())
    n = len(channels)
    current = np.array([current_spend[c] for c in channels])
    elastic = np.array([elasticities.get(c, 0.0) for c in channels])

    # If no total_budget, keep total constant
    if total_budget is None:
        total_budget = current.sum()

    # Objective: negative revenue (minimize negative = maximize revenue)
    def objective(x):
        return -np.sum(elastic * x)

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
    predicted_revenue_current = np.sum(elastic * current)
    predicted_revenue_optimal = np.sum(elastic * optimal)

    return {
        "channels": channels,
        "current_spend": {c: round(float(v), 2) for c, v in zip(channels, current)},
        "optimal_spend": {c: round(float(v), 2) for c, v in zip(channels, optimal)},
        "current_revenue": round(float(predicted_revenue_current), 2),
        "optimal_revenue": round(float(predicted_revenue_optimal), 2),
        "improvement_pct": round(
            float((predicted_revenue_optimal - predicted_revenue_current)
                  / abs(predicted_revenue_current) * 100), 2
        ) if predicted_revenue_current != 0 else 0,
        "total_budget": round(float(total_budget), 2),
    }


def scenario_analysis(
    current_spend: dict[str, float],
    elasticities: dict[str, float],
) -> dict:
    """Run multiple budget scenarios."""
    total = sum(current_spend.values())
    scenarios = {}

    # Scenario 1: same budget, reallocate
    scenarios["reallocate"] = optimize_budget(
        current_spend, elasticities, total_budget=total
    )

    # Scenario 2: +10% budget
    scenarios["increase_10pct"] = optimize_budget(
        current_spend, elasticities, total_budget=total * 1.1
    )

    # Scenario 3: +20% budget
    scenarios["increase_20pct"] = optimize_budget(
        current_spend, elasticities, total_budget=total * 1.2
    )

    # Scenario 4: -10% budget
    scenarios["decrease_10pct"] = optimize_budget(
        current_spend, elasticities, total_budget=total * 0.9
    )

    return scenarios


def main() -> None:
    """Run budget optimization."""
    mmm = load_mmm_results()
    elasticities = extract_elasticities(mmm)

    # Use average daily spend from the MMM training data as current spend baseline
    # In practice this would come from real budget data
    sample_size = mmm.get("sample_size", 365)
    current_spend = {ch: 1000.0 for ch in SPEND_CHANNELS}  # placeholder
    # Override with realistic values if available
    for ch in SPEND_CHANNELS:
        key = ch.replace("_spend", "_adstock")
        if key in elasticities:
            # Use a reasonable baseline: inverse of elasticity as proxy for current investment
            e = elasticities[ch]
            current_spend[ch] = max(100, abs(5000 / (e + 1e-6)))

    print("Running budget optimization...")
    scenarios = scenario_analysis(current_spend, elasticities)

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
        print(f"  Predicted revenue: ${result['current_revenue']:,.0f} → ${result['optimal_revenue']:,.0f}")
        print(f"  Improvement: {result['improvement_pct']:.1f}%")
        print("  Top reallocation:")
        changes = {
            c: result["optimal_spend"][c] - result["current_spend"][c]
            for c in result["channels"]
        }
        for ch, delta in sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
            print(f"    {ch}: ${result['current_spend'][ch]:,.0f} → ${result['optimal_spend'][ch]:,.0f} ({delta:+,.0f})")


if __name__ == "__main__":
    main()
