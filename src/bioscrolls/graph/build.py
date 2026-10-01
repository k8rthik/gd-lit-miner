"""Aggregate sentence-level predictions into a time-indexed knowledge graph.

* Evidence unit = one document. For each (document, head, tail) we keep the
  most confident sentence-level prediction; its label types the evidence.
* Edge = (head entity, tail entity, relation label) with document counts,
  mean/max confidence, noisy-OR combined confidence and first/last year.
* ``edge_years`` / ``entity_years`` hold per-year document counts, and
  ``year_docs`` the number of processed documents per year (trend denominators).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from bioscrolls import config
from bioscrolls.graph.metrics import noisy_or
from bioscrolls.graph.store import Store


@dataclass(frozen=True)
class Evidence:
    pmid: str
    year: int | None
    head: str
    tail: str
    label: str
    confidence: float
    n_sentences: int


@dataclass(frozen=True)
class EdgeRecord:
    head: str
    tail: str
    label: str
    n_docs: int
    n_sentences: int
    mean_confidence: float
    max_confidence: float
    combined_confidence: float
    first_year: int | None
    last_year: int | None


def document_evidence(rows: Iterable[tuple], min_confidence: float) -> list[Evidence]:
    """rows: (pmid, year, head, tail, label, confidence) for every classified candidate."""
    best: dict[tuple[str, str, str], tuple] = {}
    sentences: dict[tuple[str, str, str], int] = defaultdict(int)
    for pmid, year, head, tail, label, confidence in rows:
        if label == config.NO_RELATION or confidence < min_confidence:
            continue
        key = (pmid, head, tail)
        sentences[key] += 1
        if key not in best or confidence > best[key][5]:
            best[key] = (pmid, year, head, tail, label, confidence)
    return [Evidence(*row, n_sentences=sentences[key]) for key, row in sorted(best.items())]


def aggregate_edges(evidence: Iterable[Evidence]) -> list[EdgeRecord]:
    groups: dict[tuple[str, str, str], list[Evidence]] = defaultdict(list)
    for item in evidence:
        groups[(item.head, item.tail, item.label)].append(item)
    records = []
    for (head, tail, label), items in groups.items():
        confidences = [i.confidence for i in items]
        years = [i.year for i in items if i.year is not None]
        records.append(
            EdgeRecord(
                head,
                tail,
                label,
                len(items),
                sum(i.n_sentences for i in items),
                sum(confidences) / len(confidences),
                max(confidences),
                noisy_or(confidences),
                min(years) if years else None,
                max(years) if years else None,
            )
        )
    return sorted(records, key=lambda r: (-r.n_docs, -r.combined_confidence, r.head, r.tail))


def edge_year_rows(evidence: Iterable[Evidence]) -> list[tuple]:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for item in evidence:
        if item.year is not None:
            groups[(item.head, item.tail, item.label, item.year)].append(item.confidence)
    return [(*key, len(c), sum(c) / len(c)) for key, c in sorted(groups.items())]


def build_graph(store: Store, min_confidence: float = config.MIN_RELATION_CONFIDENCE) -> dict[str, int]:
    """Rebuild all graph tables from stored extractions. Returns table sizes."""
    rows = store.conn.execute(
        """SELECT r.pmid, d.year, r.head_id, r.tail_id, r.label, r.confidence
           FROM relations r JOIN documents d ON d.pmid = r.pmid"""
    ).fetchall()
    evidence = document_evidence(rows, min_confidence)
    store.clear_graph()
    store.conn.executemany(
        "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [tuple(vars(e).values()) for e in aggregate_edges(evidence)],
    )
    store.conn.executemany("INSERT INTO edge_years VALUES (?, ?, ?, ?, ?, ?)", edge_year_rows(evidence))
    store.conn.executescript(
        """
        INSERT INTO entity_stats
            SELECT m.entity_id, COUNT(DISTINCT m.pmid), COUNT(*), MIN(d.year), MAX(d.year)
            FROM mentions m JOIN documents d ON d.pmid = m.pmid GROUP BY m.entity_id;
        INSERT INTO entity_years
            SELECT m.entity_id, d.year, COUNT(DISTINCT m.pmid)
            FROM mentions m JOIN documents d ON d.pmid = m.pmid
            WHERE d.year IS NOT NULL GROUP BY m.entity_id, d.year;
        INSERT INTO year_docs
            SELECT year, COUNT(*) FROM documents
            WHERE extracted = 1 AND nlp_scope != 'none' AND year IS NOT NULL GROUP BY year;
        """
    )
    store.set_meta("graph_min_confidence", str(min_confidence))
    return {t: store.count(t) for t in ("edges", "edge_years", "entity_stats", "entity_years", "year_docs")}
