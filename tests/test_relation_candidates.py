import pytest

from bioscrolls.corpora.biored import parse_pubtator
from bioscrolls.relation.candidates import KeyedMention, candidate_pairs, mark_entities
from bioscrolls.relation.examples import relation_examples


def km(start, end, label, key):
    return KeyedMention(start, end, label, key)


def test_candidate_pairs_cross_type_first_mention_and_ordering():
    mentions = [
        km(0, 4, "Disease", "D1"),
        km(10, 14, "Gene", "G1"),
        km(20, 24, "Gene", "G1"),  # repeat mention of G1 ignored
        km(30, 34, "Gene", "G2"),
        km(40, 44, "Chemical", "C1"),
    ]
    pairs = candidate_pairs(mentions)
    keys = [(h.key, t.key) for h, t in pairs]
    # Gene/Chemical are heads, Disease is always tail; Gene-Gene pairs excluded.
    assert keys == [("G1", "C1"), ("G1", "D1"), ("G2", "C1"), ("G2", "D1"), ("C1", "D1")]
    assert pairs[0][0].start == 10


def test_candidate_pairs_skip_same_key():
    assert candidate_pairs([km(0, 2, "Gene", "X"), km(3, 5, "Disease", "X")]) == []


def test_mark_entities_inserts_markers_in_order():
    text = "APOE raises AD risk"
    marked = mark_entities(text, km(0, 4, "Gene", "g"), km(12, 14, "Disease", "d"))
    assert marked == "[E1] APOE [/E1] raises [E2] AD [/E2] risk"
    # Tail before head in the text.
    marked = mark_entities(text, km(12, 14, "Gene", "g"), km(0, 4, "Disease", "d"))
    assert marked == "[E2] APOE [/E2] raises [E1] AD [/E1] risk"


def test_mark_entities_rejects_overlap():
    with pytest.raises(ValueError):
        mark_entities("abcdef", km(0, 4, "Gene", "g"), km(2, 5, "Disease", "d"))


def test_relation_examples_from_biored(fixtures_dir):
    docs = parse_pubtator((fixtures_dir / "biored_sample.PubTator").read_text())
    examples = relation_examples(docs[:1])
    assert examples, "expected candidate pairs"
    labels = {(e.head_key, e.tail_key): e.label for e in examples}
    # SCN5A (6331) and bradycardia (D001919) co-occur in the title and are related.
    assert labels[("6331", "D001919")] == "Association"
    # lidocaine/mexiletine vs ventricular tachycardia: Negative_Correlation.
    assert labels[("D008012", "D017180")] == "Negative_Correlation"
    assert any(e.label == "None" for e in examples)
    for e in examples:
        assert e.text.count("[E1]") == 1 and e.text.count("[E2]") == 1
