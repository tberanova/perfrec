"""
Scoring utilities for reranking perfume recommendations using soft contextual filtering.

This module provides functions that adjust base recommendation scores by applying
lightweight user preference signals, such as preferred seasons, occasions, or types,
based on perfume metadata (chart distributions). These adjustments allow
recommendations to reflect user context while keeping collaborative information.

Currently implemented:
    - soft_boost: Boosts scores based on overlap between perfume chart categories and user-selected filters.
"""


def soft_boost(perfume, active_filters: dict[str, set[str]], category_attr_map: dict[str, str], factor: float = 0.6) -> float:
    """
    Soft-boosting of perfume scores based on chart-category overlap with user filters.

    Args:
        perfume: ORM perfume object with chart attributes.
        active_filters (dict): e.g. {"season": {"summer", "fall"}, ...}
        category_attr_map (dict): Mapping of category names to perfume chart attributes.
        factor (float): Strength of boost multiplier per matching label.

    Returns:
        float: Score adjustment to apply to base recommendation score.
    """
    boost = 0.0

    for cat, attr in category_attr_map.items():
        if cat not in active_filters:
            continue

        chart = getattr(perfume, attr) or {}
        if not chart:
            continue

        for label, pct in chart.items():  # already 0–100
            if label.lower() in active_filters[cat]:
                boost += factor * (pct / 100.0)

    return boost
