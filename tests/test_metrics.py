import pytest

from bioscrolls.evaluation.metrics import PRF, prf_from_counts, set_prf


def test_prf_from_counts():
    m = prf_from_counts(tp=8, fp=2, fn=8)
    assert m.precision == pytest.approx(0.8)
    assert m.recall == pytest.approx(0.5)
    assert m.f1 == pytest.approx(2 * 0.8 * 0.5 / 1.3)


def test_zero_division_is_zero():
    assert prf_from_counts(0, 0, 0) == PRF(0.0, 0.0, 0.0, 0, 0, 0)


def test_negative_counts_rejected():
    with pytest.raises(ValueError):
        prf_from_counts(-1, 0, 0)


def test_set_prf():
    m = set_prf(gold={1, 2, 3}, predicted={2, 3, 4, 5})
    assert (m.tp, m.fp, m.fn) == (2, 2, 1)
    assert m.as_dict()["f1"] == pytest.approx(m.f1)
