import re

from bioscrolls.extraction import (
    SCOPE_FULL,
    SCOPE_NONE,
    SCOPE_TITLE,
    extract_documents,
    nlp_scope,
    relation_confidence,
    storage_rows,
)
from bioscrolls.models import Document, Span
from bioscrolls.normalize.lexicon import LexiconEntry
from bioscrolls.normalize.normalizer import Normalizer

LEXICON = {
    "Gene": [LexiconEntry("APOE", "NCBIGene:348", "APOE", 0)],
    "Disease": [LexiconEntry("Alzheimer Disease", "MESH:D000544", "Alzheimer Disease", 0)],
    "Chemical": [LexiconEntry("Donepezil", "MESH:D000077265", "Donepezil", 0)],
}
VOCAB = {"APOE": "Gene", "Alzheimer's disease": "Disease", "AD": "Disease", "donepezil": "Chemical"}


class DictTagger:
    """Tags exact dictionary hits; stands in for the BioBERT tagger."""

    def tag(self, texts):
        out = []
        for text in texts:
            spans = []
            for surface, label in VOCAB.items():
                for m in re.finditer(rf"\b{re.escape(surface)}\b", text):
                    spans.append(Span(m.start(), m.end(), label, surface))
            out.append(sorted(spans, key=lambda s: s.start))
        return out


class FixedClassifier:
    def __init__(self):
        self.seen = []

    def classify(self, marked):
        self.seen.extend(marked)
        probs = (("None", 0.1), ("Association", 0.7), ("Positive_Correlation", 0.1), ("Negative_Correlation", 0.1))
        return [("Association", 0.7, probs) for _ in marked]


def factory(requests):
    return Normalizer.build({k: iter(v) for k, v in LEXICON.items()}, requests)


ENGLISH_ABSTRACT = (
    "Alzheimer's disease (AD) is the most common dementia. The APOE e4 allele is the strongest genetic risk factor for AD "
    "and donepezil is used in the treatment of patients with AD in the clinic."
)


def test_nlp_scope():
    assert nlp_scope(Document("1", "APOE and AD", ENGLISH_ABSTRACT, 2020)) == SCOPE_FULL
    french = (
        "Malgré le traitement dopaminergique qui améliore les symptômes moteurs, de nombreux défis restent à relever."
    )
    assert (
        nlp_scope(
            Document(
                "2", "Therapeutic perspectives in the treatment of Parkinson disease", french, 2020, languages=("fre",)
            )
        )
        == SCOPE_TITLE
    )
    assert nlp_scope(Document("3", "", "", 2020)) == SCOPE_NONE


def test_extract_documents_links_entities_and_classifies_pairs():
    doc = Document("10", "APOE in Alzheimer's disease", ENGLISH_ABSTRACT, 2021)
    clf = FixedClassifier()
    (result,) = extract_documents([doc], DictTagger(), clf, factory)
    ids = {m.entity_id for m in result.mentions}
    assert {"NCBIGene:348", "MESH:D000544", "MESH:D000077265"} <= ids
    # Short form "AD" resolved through the in-document abbreviation.
    ad = [m for m in result.mentions if m.span.text == "AD"]
    assert ad and all(m.entity_id == "MESH:D000544" for m in ad)
    pairs = {(r.head.entity_id, r.tail.entity_id) for r in result.relations}
    assert ("NCBIGene:348", "MESH:D000544") in pairs
    assert ("MESH:D000077265", "MESH:D000544") in pairs
    assert all(r.confidence == 0.9 for r in result.relations)
    assert all("[E1]" in r.evidence for r in result.relations)
    for m in result.mentions:
        assert doc.text[m.span.start : m.span.end] == m.span.text

    mentions, relations, entities = storage_rows(result)
    assert len(mentions) == len(result.mentions)
    assert relations[0][4] == "Association"
    assert ("NCBIGene:348", "Gene", "APOE", True) in entities


def test_extract_handles_documents_without_entities():
    doc = Document("11", "Nothing here", "Plain words without any entity of interest in this sentence at all.", 2021)
    clf = FixedClassifier()
    (result,) = extract_documents([doc], DictTagger(), clf, factory)
    assert result.mentions == () and result.relations == ()
    assert clf.seen == []


def test_relation_confidence_bounds():
    assert relation_confidence((("None", 0.25), ("Association", 0.75))) == 0.75
    assert relation_confidence(()) == 1.0
