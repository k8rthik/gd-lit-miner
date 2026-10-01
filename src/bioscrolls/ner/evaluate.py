"""Exact-match span-level NER evaluation (the standard strict CoNLL-style metric)."""

from __future__ import annotations

from collections.abc import Sequence

from bioscrolls.config import ENTITY_TYPES
from bioscrolls.evaluation.metrics import set_prf
from bioscrolls.models import Span


def _keyed(spans_per_sentence: Sequence[Sequence[Span]], label: str | None = None) -> set:
    return {
        (i, s.start, s.end, s.label)
        for i, spans in enumerate(spans_per_sentence)
        for s in spans
        if label is None or s.label == label
    }


def span_metrics(gold: Sequence[Sequence[Span]], predicted: Sequence[Sequence[Span]]) -> dict:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must cover the same sentences")
    return {
        "micro": set_prf(_keyed(gold), _keyed(predicted)),
        "per_type": {t: set_prf(_keyed(gold, t), _keyed(predicted, t)) for t in ENTITY_TYPES},
    }
