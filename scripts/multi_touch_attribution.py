"""Multi-touch attribution models comparison."""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

repo_root = Path(__file__).parents[1].resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from config import (
    IMAGES_DIR,
    MODEL_OUTPUT_DIR,
    SIMULATED_JOURNEYS_PATH,
    SIMULATED_TOUCHPOINTS_PATH,
)


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load simulated touchpoint and journey data."""
    tp = pl.read_parquet(SIMULATED_TOUCHPOINTS_PATH)
    j = pl.read_parquet(SIMULATED_JOURNEYS_PATH)
    return tp, j


# ---------------------------------------------------------------------------
# Rule-based attribution models
# ---------------------------------------------------------------------------


def first_touch_attribution(tp: pl.DataFrame, journeys: pl.DataFrame) -> dict[str, float]:
    """Attribute 100% to the first touchpoint."""
    # Get conversion value per user from journeys
    conv_values = journeys.select(["user_id", "conversion_value"])
    first = tp.filter(pl.col("touchpoint_number") == 1).drop("conversion_value")
    first = first.join(conv_values, on="user_id")
    result = (
        first.group_by("channel")
        .agg(pl.sum("conversion_value").alias("attributed"))
        .sort("attributed", descending=True)
    )
    return {row["channel"]: row["attributed"] for row in result.iter_rows(named=True)}


def last_touch_attribution(tp: pl.DataFrame) -> dict[str, float]:
    """Attribute 100% to the last touchpoint (conversion touchpoint)."""
    last = tp.filter(pl.col("is_conversion") == 1)
    result = (
        last.group_by("channel")
        .agg(pl.sum("conversion_value").alias("attributed"))
        .sort("attributed", descending=True)
    )
    return {row["channel"]: row["attributed"] for row in result.iter_rows(named=True)}


def linear_attribution(tp: pl.DataFrame) -> dict[str, float]:
    """Attribute equally across all touchpoints in the journey."""
    # Count touchpoints per user
    counts = tp.group_by("user_id").agg(pl.len().alias("n_touches"))
    tp = tp.join(counts, on="user_id")
    tp = tp.with_columns(pl.col("conversion_value") / pl.col("n_touches"))

    result = (
        tp.group_by("channel")
        .agg(pl.sum("conversion_value").alias("attributed"))
        .sort("attributed", descending=True)
    )
    return {row["channel"]: row["attributed"] for row in result.iter_rows(named=True)}


def time_decay_attribution(tp: pl.DataFrame, journeys: pl.DataFrame, half_life_days: float = 7.0) -> dict[str, float]:
    """Attribute more weight to touchpoints closer to conversion.

    Each touchpoint receives a share of the journey's total conversion_value,
    proportional to its time-decay weight relative to all touchpoints in the
    same journey.
    """
    # Drop per-touchpoint conversion_value (only set on conversion touchpoint)
    tp = tp.drop("conversion_value")

    # Get conversion timestamp per user
    conv = tp.filter(pl.col("is_conversion") == 1).select(["user_id", "timestamp"])
    tp = tp.join(conv, on="user_id", suffix="_conv")

    # Compute days to conversion
    tp = tp.with_columns(
        (
            pl.col("timestamp_conv").str.to_datetime(strict=False)
            - pl.col("timestamp").str.to_datetime(strict=False)
        )
        .dt.total_days()
        .alias("days_to_conv")
    )

    # Decay weight: 2^(-days/half_life)
    tp = tp.with_columns((2.0 ** (-pl.col("days_to_conv") / half_life_days)).alias("weight"))

    # Normalize weights per user
    weights = tp.group_by("user_id").agg(pl.sum("weight").alias("total_weight"))
    tp = tp.join(weights, on="user_id")

    # Join total conversion_value from the full journey table
    conv_values = journeys.select(["user_id", "conversion_value"])
    tp = tp.join(conv_values, on="user_id")

    # Attribute: weight/total_weight * user's total conversion_value
    tp = tp.with_columns(pl.col("weight") / pl.col("total_weight") * pl.col("conversion_value"))

    result = (
        tp.group_by("channel")
        .agg(pl.sum("weight").alias("attributed"))
        .sort("attributed", descending=True)
    )
    return {row["channel"]: row["attributed"] for row in result.iter_rows(named=True)}


# ---------------------------------------------------------------------------
# Shapley Value attribution
# ---------------------------------------------------------------------------


def shapley_attribution(journeys: pl.DataFrame) -> dict[str, float]:
    """Compute Shapley Value for each channel based on all subset combinations.

    Uses the standard definition: v(S) = total conversion value of paths whose
    channel set is a SUBSET of S. Larger S means more paths are included.
    """
    # Build exact conversion value per channel set
    v_exact = defaultdict(float)
    for row in journeys.filter(pl.col("converted") == 1).iter_rows(named=True):
        channels = frozenset(row["path"].split(" > "))
        v_exact[channels] += row["conversion_value"]

    all_channels = set()
    for cs in v_exact:
        all_channels.update(cs)
    all_channels = sorted(all_channels)
    n = len(all_channels)

    if n == 0:
        return {}

    # Enumerate all subsets and compute v(S) = sum of v_exact for all subsets of S
    all_subsets = []
    for r in range(0, n + 1):
        all_subsets.extend(combinations(all_channels, r))

    v = {}
    for subset in all_subsets:
        subset_set = frozenset(subset)
        total = 0.0
        for exact_set, val in v_exact.items():
            if exact_set.issubset(subset_set):
                total += val
        v[subset_set] = total

    # Compute Shapley values
    shapley = {ch: 0.0 for ch in all_channels}
    for ch in all_channels:
        for subset in all_subsets:
            if ch in subset:
                continue
            s = len(subset)
            subset_set = frozenset(subset)
            subset_with_ch = frozenset(subset + (ch,))
            weight = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
            marginal = v.get(subset_with_ch, 0) - v.get(subset_set, 0)
            shapley[ch] += marginal * weight

    # Check for negative Shapley values and warn
    negative_channels = [(k, v) for k, v in shapley.items() if v < 0]
    if negative_channels:
        names = ", ".join(f"{k} ({v:.2f})" for k, v in negative_channels)
        print(f"  Warning: {len(negative_channels)} channel(s) had negative Shapley values "
              f"and were clipped to 0: {names}")
    # Ensure non-negative
    shapley = {k: max(0.0, v) for k, v in shapley.items()}

    # Normalize to total conversion value
    total_conv = sum(v_exact.values())
    total_shapley = sum(shapley.values())
    if total_shapley > 0:
        shapley = {k: round(v / total_shapley * total_conv, 2) for k, v in shapley.items()}

    return dict(sorted(shapley.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Removal Effect attribution
# ---------------------------------------------------------------------------


def removal_effect_attribution(journeys: pl.DataFrame) -> dict[str, float]:
    """Channel removal effect analysis.

    Computes attribution by measuring the drop in overall conversion rate when
    a channel is removed from all user journeys. Channels whose removal causes
    the largest conversion drop get more credit.

    This is a removal effect analysis, NOT a full Markov chain model
    (no transition probability matrix). The name reflects the underlying
    intuition: measure each channel's contribution by observing what happens
    when it is removed.
    """
    total_users = journeys.height
    total_conv = journeys.filter(pl.col("converted") == 1).height
    baseline_rate = total_conv / total_users if total_users > 0 else 0

    if baseline_rate == 0:
        return {}

    all_channels = set()
    for row in journeys.iter_rows(named=True):
        all_channels.update(row["path"].split(" > "))

    removal_effects = {}
    for ch in sorted(all_channels):
        # Users who never touched this channel
        without_ch = journeys.filter(~pl.col("path").str.contains(rf"\b{re.escape(ch)}\b"))
        conv_without = without_ch.filter(pl.col("converted") == 1).height
        rate_without = conv_without / without_ch.height if without_ch.height > 0 else 0
        effect = (baseline_rate - rate_without) / baseline_rate
        removal_effects[ch] = max(0, effect)

    # Attribute conversions proportional to removal effect
    total_effect = sum(removal_effects.values())
    if total_effect > 0:
        total_conv_value = journeys.filter(pl.col("converted") == 1)["conversion_value"].sum()
        result = {
            ch: round(effect / total_effect * total_conv_value, 2)
            for ch, effect in removal_effects.items()
        }
    else:
        result = {ch: 0.0 for ch in removal_effects}

    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Run all models and compare
# ---------------------------------------------------------------------------


def run_all_models() -> dict:
    """Run all attribution models and return comparison."""
    tp, journeys = load_data()

    print("Running attribution models...")

    results = {
        "first_touch": first_touch_attribution(tp, journeys),
        "last_touch": last_touch_attribution(tp),
        "linear": linear_attribution(tp),
        "time_decay": time_decay_attribution(tp, journeys),
        "shapley": shapley_attribution(journeys),
        "removal_effect": removal_effect_attribution(journeys),
    }

    # Normalize to percentages
    for model_name, values in results.items():
        total = sum(values.values())
        if total > 0:
            results[model_name] = {k: round(v / total * 100, 2) for k, v in values.items()}

    # Save JSON
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_OUTPUT_DIR / "attribution_comparison.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved comparison to {out}")

    # Print summary
    print("\nAttribution model comparison (% of total conversions):")
    all_channels = set()
    for v in results.values():
        all_channels.update(v.keys())

    header = f"{'Channel':<20}" + "".join(f"{m:<12}" for m in results)
    print(header)
    print("-" * len(header))
    for ch in sorted(all_channels):
        row = f"{ch:<20}" + "".join(f"{results[m].get(ch, 0):>10.1f}%  " for m in results)
        print(row)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    run_all_models()
