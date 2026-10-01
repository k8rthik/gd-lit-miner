"""Sentence segmentation over the PubTator-style ``title + " " + abstract`` text."""

from __future__ import annotations

from bioscrolls.text.sentences import split_sentences


def join_title_abstract(title: str, abstract: str) -> str:
    return f"{title} {abstract}"


def document_sentences(title: str, abstract: str) -> list[tuple[int, int]]:
    """Split title and abstract separately (titles often lack a final period)."""
    offset = len(title) + 1
    title_spans = split_sentences(title)
    abstract_spans = [(s + offset, e + offset) for s, e in split_sentences(abstract)]
    return title_spans + abstract_spans
