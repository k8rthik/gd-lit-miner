"""Aggregation and trend metrics for knowledge-graph edges and entities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

_PER = 1000.0  # rates are expressed per 1,000 processed documents


def noisy_or(confidences: Iterable[float]) -> float:
    """Probability that at least one independent piece of evidence is correct."""
    remaining = 1.0
    for c in confidences:
        if not 0.0 <= c <= 1.0:
            raise ValueError(f"confidence out of range: {c}")
        remaining *= 1.0 - c
    return 1.0 - remaining


def ols_slope(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    var = sum((x - mean_x) ** 2 for x, _ in points)
    if var == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / var


@dataclass(frozen=True)
class Emergence:
    early_docs: int
    recent_docs: int
    early_rate: float  # supporting docs per 1,000 processed docs, earlier period
    recent_rate: float  # same, most recent ``recent_years`` years
    rate_ratio: float  # add-one smoothed recent/early ratio of rates
    slope: float  # OLS slope of the yearly rate (per 1,000 docs per year)


def emergence(year_counts: Mapping[int, int], year_totals: Mapping[int, int], recent_years: int) -> Emergence:
    """Compare how often an association is reported recently vs earlier, normalised by
    how many documents were processed in each year (the corpus is year-stratified)."""
    if recent_years <= 0:
        raise ValueError("recent_years must be positive")
    years = sorted(y for y, total in year_totals.items() if total > 0)
    if not years:
        return Emergence(0, 0, 0.0, 0.0, 1.0, 0.0)
    cutoff = years[-1] - recent_years + 1
    recent = [y for y in years if y >= cutoff]
    early = [y for y in years if y < cutoff]

    def window(ys: list[int]) -> tuple[int, int]:
        return sum(year_counts.get(y, 0) for y in ys), sum(year_totals[y] for y in ys)

    early_docs, early_total = window(early)
    recent_docs, recent_total = window(recent)
    early_rate = _PER * early_docs / early_total if early_total else 0.0
    recent_rate = _PER * recent_docs / recent_total if recent_total else 0.0
    smoothed_early = (early_docs + 1) / (early_total + 1) if early else None
    smoothed_recent = (recent_docs + 1) / (recent_total + 1)
    ratio = smoothed_recent / smoothed_early if smoothed_early else 1.0
    slope = ols_slope([(y, _PER * year_counts.get(y, 0) / year_totals[y]) for y in years])
    return Emergence(early_docs, recent_docs, early_rate, recent_rate, ratio, slope)
