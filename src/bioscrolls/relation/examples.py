"""Sentence-level relation examples derived from BioRED's document-level labels.

A candidate is every in-scope entity pair co-occurring in a sentence. Its label is
the document-level BioRED relation for that identifier pair (or ``None``). This is
the standard sentence-level reduction; pairs only related across sentences are
not representable and are counted as misses in document-level evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from bioscrolls import config
from bioscrolls.corpora.biored import AnnotatedDoc
from bioscrolls.relation.candidates import KeyedMention, candidate_pairs, mark_entities
from bioscrolls.text.segment import document_sentences


@dataclass(frozen=True)
class RelationExample:
    pmid: str
    sentence_index: int
    text: str
    head_key: str
    tail_key: str
    head_type: str
    tail_type: str
    label: str


def _sentence_mentions(doc: AnnotatedDoc, start: int, end: int) -> list[KeyedMention]:
    return [
        KeyedMention(e.start - start, e.end - start, e.label, key)
        for e in doc.entities
        if e.start >= start and e.end <= end
        for key in e.ids
    ]


def relation_examples(docs: Iterable[AnnotatedDoc]) -> list[RelationExample]:
    examples: list[RelationExample] = []
    for doc in docs:
        text = doc.text
        for index, (start, end) in enumerate(document_sentences(doc.title, doc.abstract)):
            sentence = text[start:end]
            for head, tail in candidate_pairs(_sentence_mentions(doc, start, end)):
                if head.start < tail.end and tail.start < head.end:
                    continue
                label = doc.relation_between(head.key, tail.key) or config.NO_RELATION
                examples.append(
                    RelationExample(
                        pmid=doc.pmid,
                        sentence_index=index,
                        text=mark_entities(sentence, head, tail),
                        head_key=head.key,
                        tail_key=tail.key,
                        head_type=head.label,
                        tail_type=tail.label,
                        label=label,
                    )
                )
    return examples
