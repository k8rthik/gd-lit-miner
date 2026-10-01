import json

import networkx as nx
import pytest
from graph_fixture import populate

from bioscrolls.graph.export import export_graph
from bioscrolls.graph.query import evidence_sentences, find_entities, neighbors, top_entities
from bioscrolls.graph.store import open_store
from bioscrolls.graph.trends import edge_trends, entity_series


@pytest.fixture
def db(tmp_path):
    return populate(tmp_path / "kg.db")


def test_find_entities_exact_and_fuzzy(db):
    with open_store(db) as store:
        assert find_entities(store, "apoe")[0].entity_id == "NCBIGene:348"
        assert find_entities(store, "MESH:D000544")[0].name == "Alzheimer Disease"
        assert find_entities(store, "alzheimer")[0].n_docs == 10
        assert find_entities(store, "nothing-like-this") == []
        with pytest.raises(ValueError):
            find_entities(store, "  ")


def test_neighbors_with_time_window(db):
    with open_store(db) as store:
        all_time = neighbors(store, "MESH:D000544")
        assert [(r.neighbor_name, r.n_docs) for r in all_time] == [("APOE", 10), ("Donepezil", 3)]
        recent = neighbors(store, "MESH:D000544", since=2018)
        assert [(r.neighbor_name, r.n_docs) for r in recent] == [("APOE", 2), ("Donepezil", 2)]
        assert neighbors(store, "MESH:D000544", label="Negative_Correlation")[0].neighbor_name == "Donepezil"
        assert neighbors(store, "MESH:D000544", min_docs=5)[0].first_year == 2010
        with pytest.raises(ValueError):
            neighbors(store, "x", since=2020, until=2010)
        with pytest.raises(ValueError):
            neighbors(store, "x", label="Bogus")
        with pytest.raises(ValueError):
            neighbors(store, "x", since=99)


def test_evidence_and_frequency(db):
    with open_store(db) as store:
        ev = evidence_sentences(store, "MESH:D000544", "NCBIGene:348", since=2015, limit=2)
        assert len(ev) == 2 and all(row[1] >= 2015 for row in ev)
        top = top_entities(store, limit=2)
        assert top[0][3] == 10
        assert top_entities(store, entity_type="Chemical")[0][1] == "Donepezil"
        with pytest.raises(ValueError):
            top_entities(store, entity_type="Protein")


def test_trends_rank_emerging_edges(db):
    with open_store(db) as store:
        trends = edge_trends(store, recent_years=3, min_docs=2)
        assert trends[0].tail_name == "Alzheimer Disease" and trends[0].head_name == "Donepezil"
        assert trends[0].emergence.rate_ratio > trends[1].emergence.rate_ratio
        assert edge_trends(store, entity_id="NCBIGene:348", recent_years=3)[0].series[0] == (2010, 1)
        assert entity_series(store, "MESH:D000077265")[-1] == (2019, 1, 1000.0)
        with pytest.raises(ValueError):
            edge_trends(store, min_docs=0)


def test_export_graphml_and_json(db, tmp_path):
    with open_store(db) as store:
        nodes, edges = export_graph(store, tmp_path / "g.graphml", "graphml")
        assert (nodes, edges) == (3, 2)
        graph = nx.read_graphml(tmp_path / "g.graphml")
        assert graph.nodes["NCBIGene:348"]["name"] == "APOE"
        nodes, edges = export_graph(store, tmp_path / "g.json", "json", min_docs=5)
        assert (nodes, edges) == (2, 1)
        data = json.loads((tmp_path / "g.json").read_text())
        assert data["links"][0]["years"]["2010"] == 1
        with pytest.raises(ValueError):
            export_graph(store, tmp_path / "g.csv", "csv")
