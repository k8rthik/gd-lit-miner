"""Read-side queries over the knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass

from bioscrolls import config
from bioscrolls.graph.store import Store


@dataclass(frozen=True)
class EntityRow:
    entity_id: str
    type: str
    name: str
    matched: bool
    n_docs: int
    n_mentions: int


@dataclass(frozen=True)
class NeighborRow:
    entity_id: str
    neighbor_id: str
    neighbor_name: str
    neighbor_type: str
    label: str
    n_docs: int
    mean_confidence: float
    first_year: int | None
    last_year: int | None


_ENTITY_SELECT = """
    SELECT e.entity_id, e.type, e.name, e.matched, COALESCE(s.n_docs, 0), COALESCE(s.n_mentions, 0)
    FROM entities e LEFT JOIN entity_stats s ON s.entity_id = e.entity_id
"""


def _validate_years(since: int | None, until: int | None) -> None:
    for year in (since, until):
        if year is not None and not 1800 <= year <= 2100:
            raise ValueError(f"implausible year {year}")
    if since is not None and until is not None and since > until:
        raise ValueError("--since must not be after --until")


def find_entities(store: Store, term: str, limit: int = 10) -> list[EntityRow]:
    """Exact id or case-insensitive name match first; substring match as a fallback."""
    term = term.strip()
    if not term:
        raise ValueError("entity term must be non-empty")
    exact = store.conn.execute(
        _ENTITY_SELECT + " WHERE e.entity_id = ? OR lower(e.name) = lower(?) ORDER BY 5 DESC LIMIT ?",
        (term, term, limit),
    ).fetchall()
    rows = (
        exact
        or store.conn.execute(
            _ENTITY_SELECT + " WHERE lower(e.name) LIKE lower(?) ORDER BY 5 DESC LIMIT ?", (f"%{term}%", limit)
        ).fetchall()
    )
    return [EntityRow(r[0], r[1], r[2], bool(r[3]), r[4], r[5]) for r in rows]


def neighbors(
    store: Store,
    entity_id: str,
    since: int | None = None,
    until: int | None = None,
    label: str | None = None,
    min_docs: int = 1,
    limit: int = config.DEFAULT_QUERY_LIMIT,
) -> list[NeighborRow]:
    """Edges touching ``entity_id``, re-aggregated over ``[since, until]`` when given."""
    _validate_years(since, until)
    if label is not None and label not in config.RELATION_LABELS:
        raise ValueError(f"unknown relation label {label!r}")
    sql = """
        WITH windowed AS (
            SELECT head_id, tail_id, label, SUM(n_docs) AS n_docs,
                   SUM(n_docs * mean_confidence) / SUM(n_docs) AS mean_conf,
                   MIN(year) AS first_year, MAX(year) AS last_year
            FROM edge_years
            WHERE (head_id = :eid OR tail_id = :eid)
              AND year >= COALESCE(:since, 0) AND year <= COALESCE(:until, 9999)
              AND (:label IS NULL OR label = :label)
            GROUP BY head_id, tail_id, label
        )
        SELECT CASE WHEN w.head_id = :eid THEN w.tail_id ELSE w.head_id END AS other,
               e.name, e.type, w.label, w.n_docs, w.mean_conf, w.first_year, w.last_year
        FROM windowed w JOIN entities e
          ON e.entity_id = CASE WHEN w.head_id = :eid THEN w.tail_id ELSE w.head_id END
        WHERE w.n_docs >= :min_docs
        ORDER BY w.n_docs DESC, w.mean_conf DESC
        LIMIT :limit
    """
    params = {"eid": entity_id, "since": since, "until": until, "label": label, "min_docs": min_docs, "limit": limit}
    return [NeighborRow(entity_id, *row) for row in store.conn.execute(sql, params)]


def evidence_sentences(store: Store, a: str, b: str, since: int | None = None, limit: int = 5) -> list[tuple]:
    """Highest-confidence supporting sentences for a pair: (pmid, year, label, confidence, sentence)."""
    _validate_years(since, None)
    return store.conn.execute(
        """SELECT r.pmid, d.year, r.label, r.confidence, r.sentence
           FROM relations r JOIN documents d ON d.pmid = r.pmid
           WHERE ((r.head_id = ? AND r.tail_id = ?) OR (r.head_id = ? AND r.tail_id = ?))
             AND r.label != ? AND (? IS NULL OR d.year >= ?)
           ORDER BY r.confidence DESC LIMIT ?""",
        (a, b, b, a, config.NO_RELATION, since, since, limit),
    ).fetchall()


def top_entities(
    store: Store, entity_type: str | None = None, since: int | None = None, limit: int = config.DEFAULT_QUERY_LIMIT
) -> list[tuple[str, str, str, int]]:
    """Entity frequency: (entity_id, name, type, n_docs) over documents from ``since`` on."""
    _validate_years(since, None)
    if entity_type is not None and entity_type not in config.ENTITY_TYPES:
        raise ValueError(f"unknown entity type {entity_type!r}")
    return store.conn.execute(
        """SELECT y.entity_id, e.name, e.type, SUM(y.n_docs) AS n
           FROM entity_years y JOIN entities e ON e.entity_id = y.entity_id
           WHERE y.year >= COALESCE(?, 0) AND (? IS NULL OR e.type = ?)
           GROUP BY y.entity_id ORDER BY n DESC, e.name LIMIT ?""",
        (since, entity_type, entity_type, limit),
    ).fetchall()
