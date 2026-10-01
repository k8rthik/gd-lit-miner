import json
import re

import pytest
from click.testing import CliRunner
from graph_fixture import populate

from bioscrolls import config
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


def test_build_graph_command(db):
    result = run("build-graph", "--db", db, "--min-confidence", 0.95)
    assert result.exit_code == 0, result.output
    assert re.search(r"\bedges:\s+0\b", result.output)


def test_extract_command_with_fake_pipeline(monkeypatch, tmp_path):
    from test_extraction import DictTagger, FixedClassifier, factory

    from bioscrolls import runtime
    from bioscrolls.graph.store import open_store
    from bioscrolls.models import Document

    fake = runtime.Pipeline(DictTagger(), FixedClassifier(), factory, "cpu")
    monkeypatch.setattr(runtime, "load_pipeline", lambda: fake)
    db = tmp_path / "e.db"
    abstract = "Alzheimer's disease (AD) is common. APOE is linked to AD in many cohorts of patients."
    with open_store(db) as store:
        store.upsert_documents([Document("1", "APOE in AD", abstract, 2020), Document("2", "AD", abstract, 2021)])
    result = run("extract", "--db", db, "--batch-docs", 1)
    assert result.exit_code == 0, result.output
    assert "2/2" in result.output
    with open_store(db) as store:
        assert store.count("relations") > 0
        assert store.documents(only_unextracted=True) == []


def test_download_commands(monkeypatch, tmp_path):
    from bioscrolls.corpora import biored
    from bioscrolls.normalize import lexicon

    monkeypatch.setattr(biored, "download_biored", lambda: tmp_path / "BioRED")
    monkeypatch.setattr(lexicon, "download_lexicons", lambda: [tmp_path / "hgnc.txt"])
    assert "BioRED" in run("download-corpus").output
    assert "hgnc.txt" in run("download-lexicons").output


def test_train_commands_pass_epoch_override(monkeypatch):
    from bioscrolls.ner import train as ner_train
    from bioscrolls.relation import train as re_train

    seen = {}

    def fake_ner(cfg):
        seen["ner"] = cfg.epochs
        return {"test": {"micro": {"f1": 0.5}}}

    def fake_re(cfg):
        seen["re"] = cfg.epochs
        return {"test": {"model": {"sentence": {"typed": {"f1": 0.4}}}}}

    monkeypatch.setattr(ner_train, "train_ner", fake_ner)
    monkeypatch.setattr(re_train, "train_re", fake_re)
    assert run("train-ner", "--epochs", 2).exit_code == 0
    assert run("train-re").exit_code == 0
    assert seen == {"ner": 2, "re": config.RE_TRAIN.epochs}


def test_evaluate_pipeline_command(monkeypatch, tmp_path):
    from test_extraction import DictTagger, FixedClassifier, factory

    from bioscrolls import runtime
    from bioscrolls.corpora import biored

    gold = biored.AnnotatedDoc(
        pmid="1",
        title="APOE in Alzheimer's disease",
        abstract="APOE is linked to Alzheimer's disease in these patients of the cohort.",
        entities=(
            biored.GoldEntity(0, 4, "APOE", "Gene", ("348",)),
            biored.GoldEntity(8, 27, "Alzheimer's disease", "Disease", ("D000544",)),
        ),
        relations=(biored.GoldRelation("348", "D000544", "Association"),),
    )
    monkeypatch.setattr(biored, "load_split", lambda split, corpus_dir=None: [gold])
    monkeypatch.setattr(
        runtime, "load_pipeline", lambda: runtime.Pipeline(DictTagger(), FixedClassifier(), factory, "cpu")
    )
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    result = run("evaluate-pipeline", "--split", "Test")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["relations_end_to_end"]["model"]["tp"] == 1
    assert payload["normalization_given_gold_spans"]["overall"]["accuracy"] == 1.0
    assert (tmp_path / "pipeline_metrics.json").exists()
