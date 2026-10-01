from bioscrolls.text.language import ENGLISH, NON_ENGLISH, UNKNOWN, guess_language


def test_english_text():
    text = "Mutations in the APOE gene are associated with an increased risk of Alzheimer disease in this cohort."
    assert guess_language(text) == ENGLISH


def test_french_text():
    text = "Malgré le traitement dopaminergique qui améliore les symptômes moteurs, de nombreux défis restent à relever pour la maladie."
    assert guess_language(text) == NON_ENGLISH


def test_short_or_empty_text_is_unknown():
    assert guess_language("") == UNKNOWN
    assert guess_language("APOE e4") == UNKNOWN
