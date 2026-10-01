from bioscrolls.normalize.abbreviations import find_abbreviations


def test_finds_classic_definitions():
    text = (
        "Patients with Alzheimer's disease (AD) and amyotrophic lateral sclerosis (ALS) were studied. "
        "AD risk was linked to apolipoprotein E (APOE)."
    )
    abbrevs = find_abbreviations(text)
    assert abbrevs["AD"] == "Alzheimer's disease"
    assert abbrevs["ALS"] == "amyotrophic lateral sclerosis"
    assert abbrevs["APOE"] == "apolipoprotein E"


def test_ignores_non_abbreviation_parentheses():
    text = "The drug was given daily (n = 20) to mice (see methods) and levodopa (L-DOPA) worked."
    abbrevs = find_abbreviations(text)
    assert "n = 20" not in abbrevs
    assert "see methods" not in abbrevs
    assert abbrevs["L-DOPA"] == "levodopa"


def test_no_match_when_letters_absent():
    assert find_abbreviations("We studied cats (XYZ).") == {}
