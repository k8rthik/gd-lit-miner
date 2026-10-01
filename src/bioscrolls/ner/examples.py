"""Turn annotated documents into sentence-level NER examples."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from bioscrolls.corpora.biored import AnnotatedDoc
from bioscrolls.models import Span
from bioscrolls.ner.encoding import remove_overlaps
from bioscrolls.text.segment import document_sentences


@dataclass(frozen=True)
class NerExample:
    pmid: str
    sentence_index: int
    text: str
    spans: tuple[Span, ...]


def ner_examples(docs: Iterable[AnnotatedDoc]) -> list[NerExample]:
    examples: list[NerExample] = []
    for doc in docs:
        text = doc.text
        for index, (start, end) in enumerate(document_sentences(doc.title, doc.abstract)):
            inside = [
                Span(e.start - start, e.end - start, e.label, e.text)
                for e in doc.entities
                if e.start >= start and e.end <= end
            ]
            examples.append(NerExample(doc.pmid, index, text[start:end], tuple(remove_overlaps(inside))))
    return examples
