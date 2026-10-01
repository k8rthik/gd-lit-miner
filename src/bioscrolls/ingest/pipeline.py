"""Plan PubMed queries for the neurological corpus and run ingestion."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from bioscrolls import config
from bioscrolls.ingest.eutils import SearchResult
from bioscrolls.models import Document

log = logging.getLogger(__name__)

YEARLY = "yearly"
NON_ENGLISH = "non_english"


class SearchClient(Protocol):
    def esearch(self, term: str, retmax: int, retstart: int = 0) -> SearchResult: ...

    def efetch(self, pmids: Sequence[str]) -> list[Document]: ...


@dataclass(frozen=True)
class QuerySpec:
    label: str
    term: str
    retmax: int
    year: int | None
    kind: str


@dataclass(frozen=True)
class IngestResult:
    documents: tuple[Document, ...]
    # (label, year, kind, total PubMed hits, ids retrieved)
    search_counts: tuple[tuple[str, int | None, str, int, int], ...]


def plan_queries(
    queries: Mapping[str, str],
    start: int,
    end: int,
    per_year: int,
    non_english: int,
) -> list[QuerySpec]:
    """Stratify by year so the corpus spans the whole period (needed for trends).
    ``non_english`` extra non-English-language articles are requested per topic
    and year, so that they are spread over time rather than all recent."""
    if not queries:
        raise ValueError("at least one query is required")
    if start > end:
        raise ValueError(f"start year {start} is after end year {end}")
    if per_year <= 0:
        raise ValueError("per_year must be positive")
    if non_english < 0:
        raise ValueError("non_english must be >= 0")
    specs: list[QuerySpec] = []
    for label, base in queries.items():
        for year in range(start, end + 1):
            term = f"{base} AND {config.ABSTRACT_FILTER} AND {year}[dp]"
            specs.append(QuerySpec(label, term, per_year, year, YEARLY))
            if non_english:
                specs.append(QuerySpec(label, f"{term} {config.NON_ENGLISH_FILTER}", non_english, year, NON_ENGLISH))
    return specs


def run_ingest(client: SearchClient, specs: Sequence[QuerySpec]) -> IngestResult:
    """Search every spec, de-duplicate PMIDs (first query label wins), fetch records."""
    label_by_pmid: dict[str, str] = {}
    counts = []
    for spec in specs:
        result = client.esearch(spec.term, retmax=spec.retmax)
        counts.append((spec.label, spec.year, spec.kind, result.count, len(result.ids)))
        for pmid in result.ids:
            label_by_pmid.setdefault(pmid, spec.label)
        log.info("%s %s %s: %d/%d ids", spec.label, spec.kind, spec.year, len(result.ids), result.count)
    fetched = client.efetch(list(label_by_pmid))
    documents = tuple(replace(doc, query_label=label_by_pmid.get(doc.pmid, "")) for doc in fetched)
    return IngestResult(documents=documents, search_counts=tuple(counts))
