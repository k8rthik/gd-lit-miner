"""Precision / recall / F1 helpers."""

from __future__ import annotations

from collections.abc import Hashable, Set
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def prf_from_counts(tp: int, fp: int, fn: int) -> PRF:
    if min(tp, fp, fn) < 0:
        raise ValueError("counts must be non-negative")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF(precision, recall, f1, tp, fp, fn)


def set_prf(gold: Set[Hashable], predicted: Set[Hashable]) -> PRF:
    tp = len(gold & predicted)
    return prf_from_counts(tp, len(predicted - gold), len(gold - predicted))
