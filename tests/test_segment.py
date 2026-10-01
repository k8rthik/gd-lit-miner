from bioscrolls.text.segment import document_sentences, join_title_abstract


def test_title_is_its_own_sentence_and_offsets_are_global():
    title, abstract = "APOE in AD", "First one. Second one."
    text = join_title_abstract(title, abstract)
    spans = document_sentences(title, abstract)
    assert [text[s:e] for s, e in spans] == ["APOE in AD", "First one.", "Second one."]


def test_empty_abstract():
    assert document_sentences("Only title.", "") == [(0, 11)]
