import json

import pytest
from click.testing import CliRunner
from graph_fixture import populate

from bioscrolls.cli import main


@pytest.fixture
def db(tmp_path):
    return populate(tmp_path / "kg.db")


def run(*args):
    result = CliRunner().invoke(main, list(map(str, args)))
    return result


def test_query_with_since_and_evidence(db):
    result = run("query", "--db", db, "--entity", "Alzheimer Disease", "--since", 2018, "--evidence", 1)
    assert result.exit_code == 0, result.output
    assert "APOE" in result.output and "Donepezil" in result.output
    assert "PMID" in result.output


def test_query_unknown_entity_fails(db):
    result = run("query", "--db", db, "--entity", "zzzz")
    assert result.exit_code != 0
    assert "no entity matching" in result.output


def test_query_rejects_bad_year(db):
    assert run("query", "--db", db, "--entity", "APOE", "--since", 1).exit_code != 0


def test_missing_db(tmp_path):
    result = run("stats", "--db", tmp_path / "none.db")
    assert result.exit_code != 0 and "does not exist" in result.output


def test_top_trends_stats_export(db, tmp_path):
    assert "APOE" in run("top", "--db", db).output
    trends = run("trends", "--db", db, "--recent-years", 3, "--entity", "Alzheimer")
    assert trends.exit_code == 0, trends.output
    assert "Donepezil" in trends.output
    stats = run("stats", "--db", db)
    assert "documents: 10" in stats.output and "normalised" in stats.output
    out = tmp_path / "g.json"
    exported = run("export", "--db", db, "--format", "json", "--out", out)
    assert exported.exit_code == 0, exported.output
    assert json.loads(out.read_text())["nodes"]


def test_ingest_uses_client(monkeypatch, tmp_path):
    from bioscrolls.ingest import eutils
    from bioscrolls.ingest.eutils import SearchResult
    from bioscrolls.models import Document

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def esearch(self, term, retmax, retstart=0):
            return SearchResult(("1", "2"), 99)

        def efetch(self, pmids):
            return [Document(p, "t", "a", 2020, languages=("eng",) if p == "1" else ("ger",)) for p in pmids]

    monkeypatch.setattr(eutils, "EutilsClient", FakeClient)
    db = tmp_path / "i.db"
    result = run("ingest", "--db", db, "--start", 2020, "--end", 2020, "--per-year", 2, "--topic", "als")
    assert result.exit_code == 0, result.output
    assert "stored 2 documents (1 non-English" in result.output
