import pytest

from bioscrolls.normalize.keys import lookup_keys, normalize_key


@pytest.mark.parametrize(
    "a,b",
    [
        ("Alzheimer's disease", "Alzheimer Disease"),
        ("alpha-synuclein", "synuclein alpha"),
        ("α-synuclein", "alpha synuclein"),
        ("L-DOPA", "l dopa"),
        ("  Parkinson  Disease ", "parkinson disease"),
        ("Tau", "TAU"),
    ],
)
def test_equivalent_surface_forms_share_a_key(a, b):
    assert normalize_key(a) == normalize_key(b)


def test_distinct_forms_differ():
    assert normalize_key("APOE") != normalize_key("APOE4")


def test_empty_key():
    assert normalize_key("--") == ""


def test_lookup_keys_add_singular_variant():
    assert lookup_keys("Lewy bodies") == ["bodies lewy", "body lewy"]
    assert lookup_keys("seizures") == ["seizures", "seizure"]
    assert lookup_keys("AD") == ["ad"]
