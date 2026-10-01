"""End-to-end evaluation of the deployed pipeline on BioRED.

Unlike the component metrics (which use gold entities), this runs predicted
NER -> dictionary normalisation -> relation classification on raw BioRED test
abstracts and compares the resulting identifier pairs with gold relations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from bioscrolls import config
from bioscrolls.corpora.biored import AnnotatedDoc
from bioscrolls.evaluation.metrics import set_prf
from bioscrolls.extraction import DocExtraction
from bioscrolls.models import Document
from bioscrolls.normalize.abbreviations import find_abbreviations
from bioscrolls.normalize.normalizer import Normalizer

_PREFIXES = ("NCBIGene:", "MESH:")


def strip_prefix(entity_id: str) -> str:
    for prefix in _PREFIXES:
        if entity_id.startswith(prefix):
            return entity_id[len(prefix) :]
    return entity_id


def as_documents(docs: Sequence[AnnotatedDoc]) -> list[Document]:
    return [Document(pmid=d.pmid, title=d.title, abstract=d.abstract, year=None) for d in docs]


def normalization_accuracy(docs: Sequence[AnnotatedDoc], normalizer_factory: Callable[[list], Normalizer]) -> dict:
    """Given gold spans and types, how often does normalisation return a gold id?"""
    requests, items = [], []
    for doc in docs:
        abbreviations = find_abbreviations(doc.text)
        for entity in doc.entities:
            if entity.ids:
                requests.append((entity.text, entity.label, abbreviations))
                items.append((entity, abbreviations))
    normalizer = normalizer_factory(requests)
    per_type: dict[str, dict[str, int]] = {t: {"n": 0, "correct": 0, "matched": 0} for t in config.ENTITY_TYPES}
    for entity, abbreviations in items:
        resolution = normalizer.resolve(entity.text, entity.label, abbreviations)
        stats = per_type[entity.label]
        stats["n"] += 1
        stats["matched"] += int(resolution.matched)
        stats["correct"] += int(strip_prefix(resolution.entity_id) in entity.ids)
    total = {k: sum(s[k] for s in per_type.values()) for k in ("n", "correct", "matched")}

    def summarise(s: dict[str, int]) -> dict[str, float]:
        n = s["n"] or 1
        return {"n": s["n"], "accuracy": s["correct"] / n, "vocabulary_hit_rate": s["matched"] / n}

    return {"overall": summarise(total), "per_type": {t: summarise(s) for t, s in per_type.items()}}


def pipeline_relation_metrics(
    docs: Sequence[AnnotatedDoc], extractions: Sequence[DocExtraction], min_confidence: float
) -> dict:
    """Document-level (pmid, unordered id pair) P/R/F1, model vs co-occurrence of predicted entities."""
    gold = {(d.pmid, frozenset({r.id1, r.id2})) for d in docs for r in d.relations}
    predicted, cooccurring = set(), set()
    for extraction in extractions:
        for rel in extraction.relations:
            key = (rel.pmid, frozenset({strip_prefix(rel.head.entity_id), strip_prefix(rel.tail.entity_id)}))
            cooccurring.add(key)
            if rel.label != config.NO_RELATION and rel.confidence >= min_confidence:
                predicted.add(key)
    return {"model": set_prf(gold, predicted), "cooccurrence": set_prf(gold, cooccurring)}
