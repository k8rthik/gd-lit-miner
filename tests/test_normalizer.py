from bioscrolls.normalize.lexicon import LexiconEntry
from bioscrolls.normalize.normalizer import Normalizer, candidate_keys, corpus_abbreviations

SOURCES = {
    "Gene": [
        LexiconEntry("APOE", "NCBIGene:348", "APOE", 0),
        LexiconEntry("synuclein alpha", "NCBIGene:6622", "SNCA", 0),
    ],
    "Disease": [
        LexiconEntry("Alzheimer Disease", "MESH:D000544", "Alzheimer Disease", 0),
        LexiconEntry("Seizures", "MESH:D012640", "Seizures", 0),
    ],
    "Chemical": [LexiconEntry("Levodopa", "MESH:D007980", "Levodopa", 0)],
}


def build(mentions):
    return Normalizer.build({k: iter(v) for k, v in SOURCES.items()}, mentions)


def test_resolves_direct_and_variant_forms():
    norm = build([("APOE", "Gene", {}), ("alpha-synuclein", "Gene", {}), ("seizure", "Disease", {})])
    assert norm.resolve("APOE", "Gene", {}).entity_id == "NCBIGene:348"
    r = norm.resolve("alpha-synuclein", "Gene", {})
    assert (r.entity_id, r.name, r.matched) == ("NCBIGene:6622", "SNCA", True)
    assert norm.resolve("seizure", "Disease", {}).entity_id == "MESH:D012640"


def test_resolves_abbreviation_via_long_form():
    abbrevs = {"AD": "Alzheimer's disease"}
    norm = build([("AD", "Disease", abbrevs)])
    r = norm.resolve("AD", "Disease", abbrevs)
    assert r.entity_id == "MESH:D000544"
    assert r.name == "Alzheimer Disease"


def test_strips_parenthetical_short_form_in_mention():
    norm = build([("levodopa (L-DOPA)", "Chemical", {})])
    assert norm.resolve("levodopa (L-DOPA)", "Chemical", {}).entity_id == "MESH:D007980"


def test_unmatched_mentions_get_stable_text_ids():
    norm = build([("Foo-1", "Gene", {})])
    r = norm.resolve("Foo-1", "Gene", {})
    assert r.entity_id == "gene-text:1 foo"
    assert r.name == "Foo-1"
    assert not r.matched
    # Type is respected: a gene surface form does not resolve as a disease.
    assert not norm.resolve("APOE", "Disease", {}).matched


def test_candidate_keys_order():
    keys = candidate_keys("AD", {"AD": "Alzheimer's diseases"})
    assert keys[:2] == ["alzheimer diseases", "alzheimer disease"]
    assert keys[-1] == "ad"


def test_corpus_abbreviations_majority_vote():
    texts = [
        "Alzheimer disease (AD) is common.",
        "Alzheimer's disease (AD) again.",
        "atopic dermatitis (AD) once.",
        "Alzheimer disease (AD) third.",
    ]
    assert corpus_abbreviations(texts)["AD"] == "Alzheimer disease"
