"""Relation-extraction evaluation at sentence-candidate and document level."""

from __future__ import annotations

from collections.abc import Sequence

from bioscrolls import config
from bioscrolls.corpora.biored import AnnotatedDoc
from bioscrolls.evaluation.metrics import PRF, prf_from_counts, set_prf
from bioscrolls.relation.examples import RelationExample

NONE = config.NO_RELATION
POSITIVE_LABELS = tuple(label for label in config.RELATION_LABELS if label != NONE)

Prediction = tuple[str, float]


def sentence_level_metrics(gold: Sequence[str], predicted: Sequence[str]) -> dict:
    """Binary (any relation vs none), typed micro, per-label and macro-F1 scores."""
    if len(gold) != len(predicted):
        raise ValueError(f"{len(gold)} gold labels but {len(predicted)} predictions")
    pairs = list(zip(gold, predicted, strict=True))
    binary = prf_from_counts(
        tp=sum(g != NONE and p != NONE for g, p in pairs),
        fp=sum(g == NONE and p != NONE for g, p in pairs),
        fn=sum(g != NONE and p == NONE for g, p in pairs),
    )
    typed = prf_from_counts(
        tp=sum(g == p != NONE for g, p in pairs),
        fp=sum(p != NONE and p != g for g, p in pairs),
        fn=sum(g != NONE and p != g for g, p in pairs),
    )
    per_label = {
        label: prf_from_counts(
            tp=sum(g == p == label for g, p in pairs),
            fp=sum(p == label and g != label for g, p in pairs),
            fn=sum(g == label and p != label for g, p in pairs),
        )
        for label in POSITIVE_LABELS
    }
    macro = sum(m.f1 for m in per_label.values()) / len(per_label)
    return {"binary": binary, "typed": typed, "per_label": per_label, "macro_f1": macro}


def cooccurrence_baseline(examples: Sequence[RelationExample]) -> list[Prediction]:
    """Every co-occurring in-scope pair is predicted as a generic association."""
    return [("Association", 1.0) for _ in examples]


def _doc_predictions(
    examples: Sequence[RelationExample], predictions: Sequence[Prediction]
) -> dict[tuple[str, frozenset[str]], Prediction]:
    best: dict[tuple[str, frozenset[str]], Prediction] = {}
    for example, (label, confidence) in zip(examples, predictions, strict=True):
        if label == NONE:
            continue
        key = (example.pmid, frozenset({example.head_key, example.tail_key}))
        if key not in best or confidence > best[key][1]:
            best[key] = (label, confidence)
    return best


def document_level_metrics(
    examples: Sequence[RelationExample],
    predictions: Sequence[Prediction],
    docs: Sequence[AnnotatedDoc],
) -> dict[str, PRF | float]:
    """Compare aggregated predictions to *all* gold relations, including those that
    are only expressed across sentences (which a sentence model cannot recover)."""
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions differ in length")
    gold_typed = {(d.pmid, frozenset({r.id1, r.id2}), r.label) for d in docs for r in d.relations}
    gold_pairs = {(p, k) for p, k, _ in gold_typed}
    best = _doc_predictions(examples, predictions)
    pred_typed = {(p, k, label) for (p, k), (label, _) in best.items()}
    candidate_pairs = {(e.pmid, frozenset({e.head_key, e.tail_key})) for e in examples}
    ceiling = len(gold_pairs & candidate_pairs) / len(gold_pairs) if gold_pairs else 0.0
    return {
        "binary": set_prf(gold_pairs, set(best)),
        "typed": set_prf(gold_typed, pred_typed),
        "sentence_ceiling_recall": ceiling,
    }
