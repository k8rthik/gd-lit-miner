"""Export the knowledge graph to GraphML (Gephi/Cytoscape) or node-link JSON (D3)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import networkx as nx

from bioscrolls.graph.store import Store

FORMATS = ("graphml", "json")


def to_networkx(store: Store, min_docs: int = 1, since: int | None = None) -> nx.MultiDiGraph:
    """Directed multigraph: head -> tail, one edge per relation label."""
    if min_docs < 1:
        raise ValueError("min_docs must be >= 1")
    years: dict[tuple[str, str, str], dict[int, int]] = defaultdict(dict)
    for head, tail, label, year, n in store.conn.execute(
        "SELECT head_id, tail_id, label, year, n_docs FROM edge_years WHERE year >= COALESCE(?, 0)", (since,)
    ):
        years[(head, tail, label)][year] = n
    graph = nx.MultiDiGraph()
    edges = store.conn.execute(
        "SELECT head_id, tail_id, label, mean_confidence, combined_confidence FROM edges"
    ).fetchall()
    for head, tail, label, mean_conf, combined in edges:
        series = years.get((head, tail, label), {})
        n_docs = sum(series.values())
        if n_docs < min_docs:
            continue
        graph.add_edge(
            head,
            tail,
            key=label,
            label=label,
            n_docs=n_docs,
            mean_confidence=round(mean_conf, 4),
            combined_confidence=round(combined, 4),
            first_year=min(series),
            last_year=max(series),
            years=json.dumps({str(y): c for y, c in sorted(series.items())}),
        )
    stats = {
        row[0]: row[1:]
        for row in store.conn.execute(
            "SELECT e.entity_id, e.name, e.type, e.matched, COALESCE(s.n_docs, 0) "
            "FROM entities e LEFT JOIN entity_stats s ON s.entity_id = e.entity_id"
        )
    }
    for node in graph.nodes:
        name, etype, matched, n_docs = stats.get(node, (node, "Unknown", 0, 0))
        graph.nodes[node].update(name=name, type=etype, normalized=bool(matched), n_docs=n_docs)
    return graph


def export_graph(store: Store, path: Path, fmt: str, min_docs: int = 1, since: int | None = None) -> tuple[int, int]:
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}")
    graph = to_networkx(store, min_docs=min_docs, since=since)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "graphml":
        nx.write_graphml(graph, path)
    else:
        data = nx.node_link_data(graph, edges="links")
        for link in data["links"]:
            link["years"] = json.loads(link["years"])
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return graph.number_of_nodes(), graph.number_of_edges()
