import pytest

from bioscrolls.corpora.biored import (
    CorpusFormatError,
    parse_pubtator,
)


@pytest.fixture
def docs(fixtures_dir):
    return parse_pubtator((fixtures_dir / "biored_sample.PubTator").read_text())


def test_parses_documents(docs):
    assert [d.pmid for d in docs] == ["15485686", "16046395"]
    first = docs[0]
    assert first.text.startswith("A novel SCN5A mutation")
    assert first.title_length == len(first.title)


def test_entities_keep_only_in_scope_types_and_offsets_match(docs):
    first = docs[0]
    labels = {e.label for e in first.entities}
    assert labels == {"Gene", "Disease", "Chemical"}
    for entity in first.entities:
        assert first.text[entity.start : entity.end] == entity.text
    scn5a = first.entities[0]
    assert (scn5a.text, scn5a.label, scn5a.ids) == ("SCN5A", "Gene", ("6331",))


def test_relations_are_mapped_and_filtered_to_in_scope_pairs(docs):
    rels = {(r.id1, r.id2): r.label for r in docs[0].relations}
    # Gene-disease relation kept.
    assert rels[("D001919", "6331")] == "Association"
    # Variant relations dropped (variants are out of scope).
    assert not any("SUB" in a or "SUB" in b for a, b in rels)
    # Chemical-disease kept.
    assert rels[("D001145", "D008801")] == "Negative_Correlation"


def test_relation_lookup_is_order_insensitive(docs):
    first = docs[0]
    assert first.relation_between("6331", "D001919") == "Association"
    assert first.relation_between("D001919", "6331") == "Association"
    assert first.relation_between("6331", "D999999") is None


def test_multi_id_entities_are_split():
    text = (
        "1|t|MODY genes.\n1|a|Text here.\n"
        "1\t0\t4\tMODY\tGeneOrGeneProduct\t3172,3651\n"
        "1\t0\t4\tMODY\tGeneOrGeneProduct\t-\n"
    )
    (doc,) = parse_pubtator(text)
    assert doc.entities[0].ids == ("3172", "3651")
    assert doc.entities[1].ids == ()


def test_offset_mismatch_raises():
    text = "1|t|Hello world\n1|a|x\n1\t0\t5\tWorld\tChemicalEntity\tD1\n"
    with pytest.raises(CorpusFormatError, match="offset"):
        parse_pubtator(text)


def test_malformed_line_raises():
    with pytest.raises(CorpusFormatError):
        parse_pubtator("1|t|T\n1|a|A\n1\tbroken\n")
