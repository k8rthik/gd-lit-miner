from bioscrolls.corpora.biored import AnnotatedDoc, GoldEntity, GoldRelation
from bioscrolls.evaluation.end_to_end import (
    as_documents,
    normalization_accuracy,
    pipeline_relation_metrics,
    strip_prefix,
)
from bioscrolls.extraction import DocExtraction
from bioscrolls.models import Mention, RelationPrediction, Span
from bioscrolls.normalize.lexicon import LexiconEntry
from bioscrolls.normalize.normalizer import Normalizer

DOC = AnnotatedDoc(
    pmid="1",
    title="APOE and Alzheimer disease",
    abstract="Text.",
    entities=(
        GoldEntity(0, 4, "APOE", "Gene", ("348",)),
        GoldEntity(9, 26, "Alzheimer disease", "Disease", ("D000544",)),
    ),
    relations=(GoldRelation("348", "D000544", "Association"), GoldRelation("348", "D999", "Association")),
)


def test_strip_prefix():
    assert strip_prefix("NCBIGene:348") == "348"
    assert strip_prefix("MESH:D000544") == "D000544"
    assert strip_prefix("gene-text:foo") == "gene-text:foo"


def test_as_documents():
    (doc,) = as_documents([DOC])
    assert doc.pmid == "1" and doc.title == DOC.title


def test_normalization_accuracy():
    sources = {
        "Gene": [LexiconEntry("APOE", "NCBIGene:348", "APOE", 0)],
        "Disease": [LexiconEntry("Alzheimer Disease", "MESH:D000544", "Alzheimer Disease", 0)],
        "Chemical": [],
    }
    result = normalization_accuracy(
        [DOC], lambda reqs: Normalizer.build({k: iter(v) for k, v in sources.items()}, reqs)
    )
    assert result["overall"]["accuracy"] == 1.0
    assert result["per_type"]["Gene"]["n"] == 1


def _mention(eid, label):
    return Mention("1", 0, Span(0, 4, label, "x"), eid, eid)


def test_pipeline_relation_metrics():
    rel = RelationPrediction(
        "1", 0, _mention("NCBIGene:348", "Gene"), _mention("MESH:D000544", "Disease"), "Association", 0.9
    )
    weak = RelationPrediction("1", 0, _mention("NCBIGene:348", "Gene"), _mention("MESH:D1", "Disease"), "None", 0.2)
    extraction = DocExtraction("1", "title+abstract", (), (rel, weak), ())
    m = pipeline_relation_metrics([DOC], [extraction], min_confidence=0.5)
    assert (m["model"].tp, m["model"].fp, m["model"].fn) == (1, 0, 1)
    assert (m["cooccurrence"].tp, m["cooccurrence"].fp, m["cooccurrence"].fn) == (1, 1, 1)
