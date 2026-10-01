import pytest

from bioscrolls.graph.store import open_store
from bioscrolls.models import Document


def test_roundtrip_documents_and_extractions(tmp_path):
    db = tmp_path / "x.db"
    docs = [
        Document("1", "T", "A", 2020, languages=("fre",), other_abstracts=(("fre", "x"),), query_label="pd"),
        Document("2", "T2", "A2", 2021),
    ]
    with open_store(db) as store:
        assert store.upsert_documents(docs) == 2
        store.record_search_counts([("pd", 2020, "yearly", 10, 2)])
    with open_store(db) as store:
        loaded = store.documents(only_unextracted=True)
        assert [d.pmid for d in loaded] == ["1", "2"]
        assert loaded[0].languages == ("fre",)
        store.save_extraction(
            "1",
            "title",
            [("1", 0, 0, 1, "T", "Gene", "g")],
            [("1", 0, "g", "d", "Association", 0.9, "s")],
            [("g", "Gene", "G", True)],
        )
        assert [d.pmid for d in store.documents(only_unextracted=True)] == ["2"]
        assert store.count("mentions") == 1
        # Re-saving replaces rather than duplicates.
        store.save_extraction("1", "title", [("1", 0, 0, 1, "T", "Gene", "g")], [], [("g", "Gene", "G", True)])
        assert store.count("mentions") == 1 and store.count("relations") == 0
        assert len(store.documents(limit=1)) == 1
        store.set_meta("k", "v")
        assert store.get_meta("k") == "v" and store.get_meta("missing") is None
        with pytest.raises(ValueError):
            store.count("sqlite_master")


def test_rollback_on_error(tmp_path):
    db = tmp_path / "y.db"
    with pytest.raises(RuntimeError):
        with open_store(db) as store:
            store.upsert_documents([Document("5", "t", "a", 2000)])
            raise RuntimeError("boom")
    with open_store(db) as store:
        assert store.count("documents") == 0
