import pytest

from bioscrolls.corpora.biored import parse_pubtator
from bioscrolls.ner.examples import ner_examples


@pytest.fixture
def doc(fixtures_dir):
    return parse_pubtator((fixtures_dir / "biored_sample.PubTator").read_text())[0]


def test_examples_cover_sentences_with_relative_spans(doc):
    examples = ner_examples([doc])
    assert examples[0].text.startswith("A novel SCN5A mutation")
    for ex in examples:
        for span in ex.spans:
            assert ex.text[span.start : span.end] == span.text
    total = sum(len(ex.spans) for ex in examples)
    in_scope = len(doc.entities)
    assert 0 < total <= in_scope


def test_example_ids_are_stable(doc):
    examples = ner_examples([doc])
    assert examples[0].pmid == "15485686"
    assert [ex.sentence_index for ex in examples] == list(range(len(examples)))
