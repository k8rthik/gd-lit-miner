"""Immutable domain records shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from bioscrolls.config import ENTITY_TYPES, RELATION_LABELS


class ValidationError(ValueError):
    """Raised when a domain record is constructed with invalid data."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


@dataclass(frozen=True)
class Document:
    """A PubMed record reduced to what the pipeline needs.

    ``title``/``abstract`` hold the English text used for NLP (PubMed supplies an
    English title and usually an English abstract for non-English articles).
    ``other_abstracts`` keeps non-English abstracts as ``(language, text)`` pairs.
    """

    pmid: str
    title: str
    abstract: str
    year: int | None
    languages: tuple[str, ...] = ("eng",)
    journal: str = ""
    vernacular_title: str = ""
    other_abstracts: tuple[tuple[str, str], ...] = ()
    query_label: str = ""

    def __post_init__(self) -> None:
        _require(self.pmid.isdigit(), f"pmid must be numeric, got {self.pmid!r}")
        _require(self.year is None or 1800 <= self.year <= 2100, f"bad year {self.year}")
        _require(len(self.languages) > 0, "languages must not be empty")

    @property
    def is_english(self) -> bool:
        return "eng" in self.languages

    @property
    def text(self) -> str:
        """Title and abstract joined the way PubTator does (title, space, abstract)."""
        return f"{self.title} {self.abstract}".strip()


@dataclass(frozen=True)
class Span:
    """A typed character span inside some text."""

    start: int
    end: int
    label: str
    text: str = ""

    def __post_init__(self) -> None:
        _require(0 <= self.start < self.end, f"invalid span [{self.start}, {self.end})")
        _require(self.label in ENTITY_TYPES, f"unknown entity type {self.label!r}")


@dataclass(frozen=True)
class Mention:
    """An entity mention in a document, optionally normalized to an identifier."""

    pmid: str
    sentence_index: int
    span: Span
    entity_id: str
    entity_name: str


@dataclass(frozen=True)
class RelationPrediction:
    """A classified relation between two entity mentions in one sentence."""

    pmid: str
    sentence_index: int
    head: Mention
    tail: Mention
    label: str
    confidence: float
    probabilities: tuple[tuple[str, float], ...] = field(default=())
    evidence: str = ""

    def __post_init__(self) -> None:
        _require(self.label in RELATION_LABELS, f"unknown relation label {self.label!r}")
        _require(0.0 <= self.confidence <= 1.0, f"confidence out of range: {self.confidence}")
