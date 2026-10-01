from bioscrolls.text.sentences import split_sentences


def texts(text):
    return [text[s:e] for s, e in split_sentences(text)]


def test_basic_split_with_offsets():
    text = "APOE is a gene. It is linked to AD!  Is it?"
    assert texts(text) == ["APOE is a gene.", "It is linked to AD!", "Is it?"]


def test_does_not_split_on_abbreviations_or_decimals():
    text = "Levels rose 2.5 fold (e.g. in mice) vs. controls, cf. Fig. 2. Smith et al. agreed. Next one."
    assert texts(text) == [
        "Levels rose 2.5 fold (e.g. in mice) vs. controls, cf. Fig. 2.",
        "Smith et al. agreed.",
        "Next one.",
    ]


def test_does_not_split_before_lowercase_or_digit():
    text = "The p. value was low. this continues. 3 patients died."
    assert texts(text) == ["The p. value was low. this continues. 3 patients died."]


def test_section_labels_start_new_sentence():
    text = "BACKGROUND: Something happened. METHODS: We did it."
    assert texts(text) == ["BACKGROUND: Something happened.", "METHODS: We did it."]


def test_empty_and_whitespace():
    assert split_sentences("") == []
    assert split_sentences("   ") == []
    assert texts("No terminal punctuation") == ["No terminal punctuation"]
