import pytest

from bioscrolls.corpora.biored import AnnotatedDoc, GoldRelation
from bioscrolls.relation.evaluate import (
    cooccurrence_baseline,
    document_level_metrics,
    sentence_level_metrics,
)
from bioscrolls.relation.examples import RelationExample


def ex(pmid, h, t, label, idx=0):
    return RelationExample(pmid, idx, "x", h, t, "Gene", "Disease", label)


def test_sentence_level_binary_and_typed():
    gold = ["Association", "None", "Positive_Correlation", "Negative_Correlation"]
    pred = ["Association", "Association", "Association", "None"]
    m = sentence_level_metrics(gold, pred)
    # binary: tp=2 (idx0, idx2), fp=1 (idx1), fn=1 (idx3)
    assert (m["binary"].tp, m["binary"].fp, m["binary"].fn) == (2, 1, 1)
    # typed: tp=1, fp=2 (idx1 wrong, idx2 wrong type), fn=2 (idx2, idx3)
    assert (m["typed"].tp, m["typed"].fp, m["typed"].fn) == (1, 2, 2)
    assert m["per_label"]["Association"].tp == 1
    assert 0 <= m["macro_f1"] <= 1


def test_sentence_level_length_mismatch():
    with pytest.raises(ValueError):
        sentence_level_metrics(["None"], [])


def test_cooccurrence_baseline_predicts_association_everywhere():
    preds = cooccurrence_baseline([ex("1", "a", "b", "None"), ex("1", "a", "c", "Association")])
    assert preds == [("Association", 1.0), ("Association", 1.0)]


def test_document_level_counts_cross_sentence_gold_as_misses():
    docs = [
        AnnotatedDoc("1", "t", "a", (), (GoldRelation("a", "b", "Association"), GoldRelation("a", "z", "Association"))),
    ]
    examples = [ex("1", "a", "b", "Association", 0), ex("1", "a", "b", "Association", 1), ex("1", "a", "c", "None")]
    preds = [("None", 0.9), ("Positive_Correlation", 0.6), ("Association", 0.7)]
    m = document_level_metrics(examples, preds, docs)
    # predicted pairs {ab (Positive), ac}; gold {ab, az}
    assert (m["binary"].tp, m["binary"].fp, m["binary"].fn) == (1, 1, 1)
    assert m["typed"].tp == 0  # ab predicted Positive_Correlation, gold Association
    assert m["sentence_ceiling_recall"] == pytest.approx(0.5)
