import pytest

from bioscrolls.config import NER_LABELS
from bioscrolls.models import Span
from bioscrolls.ner.encoding import (
    IGNORE_INDEX,
    LABEL_TO_ID,
    decode_tags,
    remove_overlaps,
    spans_to_token_tags,
)

# "APOE e4 raises AD risk" tokenised as: [CLS] AP ##OE e ##4 raises AD risk [SEP]
OFFSETS = [(0, 0), (0, 2), (2, 4), (5, 6), (6, 7), (8, 14), (15, 17), (18, 22), (0, 0)]


def tag_names(ids):
    return [NER_LABELS[i] if i != IGNORE_INDEX else None for i in ids]


def test_spans_to_token_tags():
    spans = [Span(0, 4, "Gene"), Span(15, 17, "Disease")]
    tags = spans_to_token_tags(OFFSETS, spans)
    assert tag_names(tags) == [None, "B-Gene", "I-Gene", "O", "O", "O", "B-Disease", "O", None]


def test_span_starting_mid_token_still_tagged():
    tags = spans_to_token_tags(OFFSETS, [Span(1, 4, "Gene")])
    assert tag_names(tags)[1:3] == ["B-Gene", "I-Gene"]


def test_decode_tags_roundtrip():
    tags = spans_to_token_tags(OFFSETS, [Span(0, 4, "Gene"), Span(15, 22, "Disease")])
    spans = decode_tags(OFFSETS, tags, "APOE e4 raises AD risk")
    assert [(s.start, s.end, s.label, s.text) for s in spans] == [
        (0, 4, "Gene", "APOE"),
        (15, 22, "Disease", "AD risk"),
    ]


def test_decode_handles_orphan_inside_and_type_switch():
    ids = [
        IGNORE_INDEX,
        LABEL_TO_ID["I-Gene"],
        LABEL_TO_ID["I-Gene"],
        LABEL_TO_ID["I-Disease"],
        LABEL_TO_ID["O"],
        LABEL_TO_ID["O"],
        LABEL_TO_ID["B-Chemical"],
        LABEL_TO_ID["B-Chemical"],
        IGNORE_INDEX,
    ]
    spans = decode_tags(OFFSETS, ids, "APOE e4 raises AD risk")
    assert [(s.start, s.end, s.label) for s in spans] == [
        (0, 4, "Gene"),
        (5, 6, "Disease"),
        (15, 17, "Chemical"),
        (18, 22, "Chemical"),
    ]


def test_decode_length_mismatch():
    with pytest.raises(ValueError):
        decode_tags(OFFSETS, [0], "x")


def test_remove_overlaps_prefers_longer_then_earlier():
    spans = [Span(0, 4, "Gene"), Span(0, 10, "Disease"), Span(12, 15, "Chemical"), Span(14, 18, "Gene")]
    kept = remove_overlaps(spans)
    assert [(s.start, s.end) for s in kept] == [(0, 10), (14, 18)]
