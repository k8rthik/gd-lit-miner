import pytest

from bioscrolls.graph.metrics import emergence, noisy_or, ols_slope


def test_noisy_or():
    assert noisy_or([]) == 0.0
    assert noisy_or([0.5, 0.5]) == pytest.approx(0.75)
    assert noisy_or([1.0, 0.2]) == 1.0
    with pytest.raises(ValueError):
        noisy_or([1.5])


def test_ols_slope():
    assert ols_slope([(2000, 1.0), (2001, 2.0), (2002, 3.0)]) == pytest.approx(1.0)
    assert ols_slope([(2000, 1.0)]) == 0.0
    assert ols_slope([]) == 0.0


def test_emergence_detects_rising_association():
    totals = {y: 100 for y in range(2010, 2020)}
    counts = {2017: 3, 2018: 4, 2019: 5}
    e = emergence(counts, totals, recent_years=3)
    assert e.recent_rate == pytest.approx(12 / 300 * 1000)
    assert e.early_rate == 0.0
    assert e.rate_ratio > 5
    assert e.slope > 0
    assert e.recent_docs == 12 and e.early_docs == 0


def test_emergence_flat_and_empty():
    totals = {y: 10 for y in range(2010, 2014)}
    flat = emergence({y: 1 for y in totals}, totals, recent_years=2)
    assert flat.rate_ratio == pytest.approx(1.0)
    empty = emergence({}, {}, recent_years=2)
    assert empty.recent_docs == 0 and empty.rate_ratio == 1.0
    with pytest.raises(ValueError):
        emergence({}, totals, recent_years=0)
