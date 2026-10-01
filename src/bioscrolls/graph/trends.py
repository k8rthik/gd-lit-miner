"""Per-year trend series and emerging-association ranking."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from bioscrolls import config
from bioscrolls.graph.metrics import Emergence, emergence
from bioscrolls.graph.store import Store


@dataclass(frozen=True)
class EdgeTrend:
    head_id: str
    head_name: str
    tail_id: str
    tail_name: str
    label: str
    series: tuple[tuple[int, int], ...]  # (year, supporting docs)
    emergence: Emergence


def year_totals(store: Store) -> dict[int, int]:
    return dict(store.conn.execute("SELECT year, n_docs FROM year_docs").fetchall())


def edge_trends(
    store: Store,
    entity_id: str | None = None,
    recent_years: int = config.EMERGING_RECENT_YEARS,
    min_docs: int = config.EMERGING_MIN_DOCS,
) -> list[EdgeTrend]:
    """All edges (optionally touching ``entity_id``) with >= ``min_docs`` supporting
    documents, ranked by how much more often they are reported recently."""
    if min_docs < 1:
        raise ValueError("min_docs must be >= 1")
    totals = year_totals(store)
    rows = store.conn.execute(
        """SELECT y.head_id, h.name, y.tail_id, t.name, y.label, y.year, y.n_docs
           FROM edge_years y
           JOIN entities h ON h.entity_id = y.head_id JOIN entities t ON t.entity_id = y.tail_id
           WHERE (? IS NULL OR y.head_id = ? OR y.tail_id = ?)""",
        (entity_id, entity_id, entity_id),
    ).fetchall()
    grouped: dict[tuple, dict[int, int]] = defaultdict(dict)
    for head, head_name, tail, tail_name, label, year, n in rows:
        grouped[(head, head_name, tail, tail_name, label)][year] = n
    trends = [
        EdgeTrend(*key, series=tuple(sorted(counts.items())), emergence=emergence(counts, totals, recent_years))
        for key, counts in grouped.items()
        if sum(counts.values()) >= min_docs
    ]
    return sorted(trends, key=lambda t: (-t.emergence.rate_ratio, -t.emergence.recent_docs, t.head_name))


def entity_series(store: Store, entity_id: str) -> list[tuple[int, int, float]]:
    """(year, docs mentioning the entity, docs per 1,000 processed docs)."""
    totals = year_totals(store)
    rows = store.conn.execute(
        "SELECT year, n_docs FROM entity_years WHERE entity_id = ? ORDER BY year", (entity_id,)
    ).fetchall()
    return [(y, n, 1000.0 * n / totals[y] if totals.get(y) else 0.0) for y, n in rows]
