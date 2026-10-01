import pytest

from bioscrolls.graph.build import aggregate_edges, build_graph, document_evidence, edge_year_rows
from bioscrolls.graph.store import open_store
from bioscrolls.models import Document

ROWS = [
    ("1", 2010, "g", "d", "Association", 0.9),
    ("1", 2010, "g", "d", "Positive_Correlation", 0.6),
    ("2", 2015, "g", "d", "Association", 0.5),
    ("3", 2016, "g", "d", "None", 0.2),
    ("3", 2016, "c", "d", "Negative_Correlation", 0.4),  # below threshold
]


def test_document_evidence_keeps_best_sentence_per_doc():
    ev = document_evidence(ROWS, min_confidence=0.5)
    assert [(e.pmid, e.label, e.confidence, e.n_sentences) for e in ev] == [
        ("1", "Association", 0.9, 2),
        ("2", "Association", 0.5, 1),
    ]


def test_aggregate_edges_and_years():
    ev = document_evidence(ROWS, min_confidence=0.5)
    (edge,) = aggregate_edges(ev)
    assert (edge.head, edge.tail, edge.label, edge.n_docs) == ("g", "d", "Association", 2)
    assert edge.mean_confidence == pytest.approx(0.7)
    assert edge.combined_confidence == pytest.approx(1 - 0.1 * 0.5)
    assert (edge.first_year, edge.last_year) == (2010, 2015)
    assert edge_year_rows(ev) == [("g", "d", "Association", 2010, 1, 0.9), ("g", "d", "Association", 2015, 1, 0.5)]


def test_build_graph_end_to_end(tmp_path):
    with open_store(tmp_path / "g.db") as store:
        store.upsert_documents([Document("1", "t", "a", 2010), Document("2", "t", "a", 2015)])
        store.save_extraction(
            "1",
            "title+abstract",
            [("1", 0, 0, 1, "G", "Gene", "g"), ("1", 0, 2, 3, "D", "Disease", "d")],
            [("1", 0, "g", "d", "Association", 0.8, "s")],
            [("g", "Gene", "G", True), ("d", "Disease", "D", True)],
        )
        store.save_extraction("2", "title", [("2", 0, 0, 1, "G", "Gene", "g")], [], [])
        sizes = build_graph(store, min_confidence=0.5)
        assert sizes == {"edges": 1, "edge_years": 1, "entity_stats": 2, "entity_years": 3, "year_docs": 2}
        assert store.get_meta("graph_min_confidence") == "0.5"
        # Rebuilding is idempotent.
        assert build_graph(store, min_confidence=0.5) == sizes
