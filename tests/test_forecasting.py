import numpy as np

from src.forecasting.baseline import seasonal_naive, wape


def test_seasonal_naive_repeats_weekly_pattern():
    week = np.arange(7)
    pred = seasonal_naive(np.tile(week, 4), 10)
    assert list(pred[:7]) == list(week)
    assert len(pred) == 10


def test_wape_perfect_and_zero_denominator():
    assert wape([1, 2, 3], [1, 2, 3]) == 0
    assert np.isnan(wape([0, 0], [1, 1]))
