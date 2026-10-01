"""End-to-end extraction: sentences -> NER -> normalisation -> relation classification."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from bioscrolls import config
from bioscrolls.models import Document, Mention, RelationPrediction, Span
from bioscrolls.normalize.abbreviations import find_abbreviations
from bioscrolls.normalize.normalizer import Normalizer, Resolution
from bioscrolls.relation.candidates import KeyedMention, candidate_pairs, mark_entities
from bioscrolls.text.language import NON_ENGLISH, guess_language
from bioscrolls.text.segment import document_sentences, join_title_abstract

log = logging.getLogger(__name__)

SCOPE_FULL = "title+abstract"
SCOPE_TITLE = "title"
SCOPE_NONE = "none"

Probabilities = tuple[tuple[str, float], ...]


class Tagger(Protocol):
    def tag(self, texts: Sequence[str]) -> list[list[Span]]: ...


class Classifier(Protocol):
    def classify(self, marked_texts: Sequence[str]) -> list[tuple[str, float, Probabilities]]: ...


NormalizerFactory = Callable[[list[tuple[str, str, Mapping[str, str]]]], Normalizer]


@dataclass(frozen=True)
class DocExtraction:
    pmid: str
    scope: str
    mentions: tuple[Mention, ...]
    relations: tuple[RelationPrediction, ...]
    resolutions: tuple[tuple[str, Resolution], ...]  # (entity type, resolution)


@dataclass(frozen=True)
class _Sentence:
    pmid: str
    index: int
    start: int
    text: str


@dataclass(frozen=True)
class _Candidate:
    sentence: _Sentence
    head: Mention
    tail: Mention
    marked: str


def nlp_scope(doc: Document) -> str:
    """Which English text we can honestly run the English-only models on."""
    if doc.abstract and guess_language(doc.abstract) != NON_ENGLISH:
        return SCOPE_FULL
    if doc.title and guess_language(doc.title) != NON_ENGLISH:
        return SCOPE_TITLE
    return SCOPE_NONE


def _sentences(doc: Document, scope: str) -> list[_Sentence]:
    if scope == SCOPE_NONE:
        return []
    abstract = doc.abstract if scope == SCOPE_FULL else ""
    text = join_title_abstract(doc.title, abstract)
    spans = document_sentences(doc.title, abstract)
    return [_Sentence(doc.pmid, i, s, text[s:e]) for i, (s, e) in enumerate(spans)]


def relation_confidence(probabilities: Probabilities) -> float:
    """P(any relation) = 1 - P(None); robust to which positive label wins."""
    p_none = dict(probabilities).get(config.NO_RELATION, 0.0)
    return min(1.0, max(0.0, 1.0 - p_none))


def _abbreviations(docs: Sequence[Document], corpus: Mapping[str, str] | None) -> dict[str, dict[str, str]]:
    return {d.pmid: {**(corpus or {}), **find_abbreviations(d.text)} for d in docs}


def _link_sentence(
    sentence: _Sentence, spans: Sequence[Span], normalizer: Normalizer, abbreviations: Mapping[str, str]
) -> tuple[list[Mention], list[tuple[str, Resolution]], list[_Candidate]]:
    mentions, resolutions, keyed = [], [], []
    for span in spans:
        resolution = normalizer.resolve(span.text, span.label, abbreviations)
        global_span = Span(span.start + sentence.start, span.end + sentence.start, span.label, span.text)
        mention = Mention(sentence.pmid, sentence.index, global_span, resolution.entity_id, resolution.name)
        mentions.append(mention)
        resolutions.append((span.label, resolution))
        keyed.append((KeyedMention(span.start, span.end, span.label, resolution.entity_id), mention))
    by_position = {(k.start, k.key): m for k, m in keyed}
    candidates = [
        _Candidate(
            sentence, by_position[(h.start, h.key)], by_position[(t.start, t.key)], mark_entities(sentence.text, h, t)
        )
        for h, t in candidate_pairs([k for k, _ in keyed])
        if h.end <= t.start or t.end <= h.start
    ]
    return mentions, resolutions, candidates


def extract_documents(
    docs: Sequence[Document],
    tagger: Tagger,
    classifier: Classifier,
    normalizer_factory: NormalizerFactory,
    corpus_abbreviations: Mapping[str, str] | None = None,
) -> list[DocExtraction]:
    scopes = {d.pmid: nlp_scope(d) for d in docs}
    sentences = [s for d in docs for s in _sentences(d, scopes[d.pmid])]
    spans_per_sentence = tagger.tag([s.text for s in sentences]) if sentences else []
    abbreviations = _abbreviations(docs, corpus_abbreviations)
    normalizer = normalizer_factory(
        [
            (sp.text, sp.label, abbreviations[s.pmid])
            for s, spans in zip(sentences, spans_per_sentence, strict=True)
            for sp in spans
        ]
    )

    mentions = {d.pmid: [] for d in docs}
    resolutions = {d.pmid: [] for d in docs}
    candidates: list[_Candidate] = []
    for sentence, spans in zip(sentences, spans_per_sentence, strict=True):
        found, resolved, pairs = _link_sentence(sentence, spans, normalizer, abbreviations[sentence.pmid])
        mentions[sentence.pmid].extend(found)
        resolutions[sentence.pmid].extend(resolved)
        candidates.extend(pairs)

    predictions = classifier.classify([c.marked for c in candidates]) if candidates else []
    relations = {d.pmid: [] for d in docs}
    for cand, (label, _, probs) in zip(candidates, predictions, strict=True):
        relations[cand.sentence.pmid].append(
            RelationPrediction(
                cand.sentence.pmid,
                cand.sentence.index,
                cand.head,
                cand.tail,
                label,
                relation_confidence(probs),
                probs,
                cand.marked,
            )
        )
    return [
        DocExtraction(
            d.pmid, scopes[d.pmid], tuple(mentions[d.pmid]), tuple(relations[d.pmid]), tuple(resolutions[d.pmid])
        )
        for d in docs
    ]


def storage_rows(extraction: DocExtraction) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """Rows for :meth:`Store.save_extraction`: (mentions, relations, entities)."""
    mention_rows = [
        (m.pmid, m.sentence_index, m.span.start, m.span.end, m.span.text, m.span.label, m.entity_id)
        for m in extraction.mentions
    ]
    relation_rows = [
        (r.pmid, r.sentence_index, r.head.entity_id, r.tail.entity_id, r.label, r.confidence, r.evidence)
        for r in extraction.relations
    ]
    entity_rows = [(res.entity_id, label, res.name, res.matched) for label, res in extraction.resolutions]
    return mention_rows, relation_rows, entity_rows
