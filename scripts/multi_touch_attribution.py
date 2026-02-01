"""Multi-touch attribution models comparison."""
import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

import sys
repo_root = Path(__file__).parents[1].resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from config import (
    SIMULATED_TOUCHPOINTS_PATH,
    SIMULATED_JOURNEYS_PATH,
    MODEL_OUTPUT_DIR,
    IMAGES_DIR,
)


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load simulated touchpoint and journey data."""
    tp = pl.read_parquet(SIMULATED_TOUCHPOINTS_PATH)
    j = pl.read_parquet(SIMULATED_JOURNEYS_PATH)
    return tp, j


# ---------------------------------------------------------------------------
# Rule-based attribution models
# ---------------------------------------------------------------------------

def first_touch_attribution(tp: pl.DataFrame) -> dict[str, float]:
    """Attribute 100% to the first touchpoint."""
    first = tp.filter(pl.col("touchpoint_number") == 1)
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


def time_decay_attribution(tp: pl.DataFrame, half_life_days: float = 7.0) -> dict[str, float]:
    """Attribute more weight to touchpoints closer to conversion."""
    # Get conversion timestamp per user
    conv = tp.filter(pl.col("is_conversion") == 1).select(["user_id", "timestamp"])
    tp = tp.join(conv, on="user_id", suffix="_conv")

    # Compute days to conversion
    tp = tp.with_columns(
        (pl.col("timestamp_conv").str.to_datetime() - pl.col("timestamp").str.to_datetime())
        .dt.total_days()
        .alias("days_to_conv")
    )

    # Decay weight: 2^(-days/half_life)
    tp = tp.with_columns(
        (2.0 ** (-pl.col("days_to_conv") / half_life_days)).alias("weight")
    )

    # Normalize weights per user
    weights = tp.group_by("user_id").agg(pl.sum("weight").alias("total_weight"))
    tp = tp.join(weights, on="user_id")
    tp = tp.with_columns(
        pl.col("weight") / pl.col("total_weight") * pl.col("conversion_value")
    )

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
    """Compute Shapley Value for each channel based on all subset combinations."""
    # Build conversion count per channel set
    channel_sets = defaultdict(float)
    for row in journeys.filter(pl.col("converted") == 1).iter_rows(named=True):
        channels = set(row["path"].split(" > "))
        for r in range(1, len(channels) + 1):
            for subset in combinations(channels, r):
                channel_sets[frozenset(subset)] += row["conversion_value"]

    all_channels = set()
    for cs in channel_sets:
        all_channels.update(cs)
    all_channels = sorted(all_channels)
    n = len(all_channels)

    if n == 0:
        return {}

    shapley = {ch: 0.0 for ch in all_channels}

    for ch in all_channels:
        for subset in channel_sets:
            if ch not in subset:
                continue
            subset_without_ch = frozenset(subset - {ch})
            v_with = channel_sets.get(subset, 0)
            v_without = channel_sets.get(subset_without_ch, 0)
            marginal = v_with - v_without
            s = len(subset)
            weight = math.factorial(s - 1) * math.factorial(n - s) / math.factorial(n)
            shapley[ch] += marginal * weight

    # Normalize
    total = sum(shapley.values())
    if total > 0:
        shapley = {k: round(v / total * sum(channel_sets.values()), 2) for k, v in shapley.items()}

    return dict(sorted(shapley.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Markov Chain attribution
# ---------------------------------------------------------------------------

def markov_attribution(journeys: pl.DataFrame) -> dict[str, float]:
    """Markov chain attribution with removal effect."""
    # Build transition counts: start -> channel1 -> channel2 -> ... -> conversion/null
    transitions = defaultdict(lambda: defaultdict(float))
    conversion_count = 0.0
    null_count = 0.0

    for row in journeys.iter_rows(named=True):
        channels = row["path"].split(" > ")
        conv_value = row["conversion_value"] if row["converted"] else 0

        # Start state
        prev = "start"
        for ch in channels:
            transitions[prev][ch] += 1
            prev = ch

        if row["converted"]:
            transitions[prev]["conversion"] += 1
            conversion_count += conv_value
        else:
            transitions[prev]["null"] += 1
            null_count += 1

    all_states = set(transitions.keys()) | {"conversion", "null"}

    # Compute transition probabilities
    trans_prob = {}
    for state in transitions:
        total = sum(transitions[state].values())
        trans_prob[state] = {k: v / total for k, v in transitions[state].items()}

    # Compute conversion probability from start
    def conv_prob_from(start: str, removed: set[str] | None = None) -> float:
        """Monte Carlo simulation of conversion probability."""
        rng = np.random.default_rng(42)
        n_sim = 10_000
        conv = 0
        for _ in range(n_sim):
            state = start
            for _ in range(20):  # max path length
                if state == "conversion":
                    conv += 1
                    break
                if state == "null":
                    break
                if state not in trans_prob:
                    break
                probs = trans_prob[state]
                if removed:
                    probs = {k: v for k, v in probs.items() if k not in removed}
                    total = sum(probs.values())
                    if total == 0:
                        break
                    probs = {k: v / total for k, v in probs.items()}
                states = list(probs.keys())
                p = list(probs.values())
                state = rng.choice(states, p=p)
        return conv / n_sim

    baseline = conv_prob_from("start")

    # Compute removal effect for each channel
    all_channels = set()
    for state in transitions:
        if state not in ("start", "conversion", "null"):
            all_channels.add(state)

    removal_effects = {}
    for ch in sorted(all_channels):
        without = conv_prob_from("start", removed={ch})
        effect = (baseline - without) / baseline if baseline > 0 else 0
        removal_effects[ch] = effect

    # Attribute conversions proportional to removal effect
    total_effect = sum(removal_effects.values())
    if total_effect > 0:
        markov = {ch: round(effect / total_effect * conversion_count, 2)
                  for ch, effect in removal_effects.items()}
    else:
        markov = {ch: 0.0 for ch in removal_effects}

    return dict(sorted(markov.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Run all models and compare
# ---------------------------------------------------------------------------

def run_all_models() -> dict:
    """Run all attribution models and return comparison."""
    tp, journeys = load_data()

    print("Running attribution models...")

    results = {
        "first_touch": first_touch_attribution(tp),
        "last_touch": last_touch_attribution(tp),
        "linear": linear_attribution(tp),
        "time_decay": time_decay_attribution(tp),
        "shapley": shapley_attribution(journeys),
        "markov": markov_attribution(journeys),
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
