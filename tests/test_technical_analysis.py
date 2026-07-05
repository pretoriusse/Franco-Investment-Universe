"""
Unit tests for assets.technical_analysis indicator math.

These are pinned-behavior tests: they document what the current formulas
actually do, including the unguarded NaN/inf edge cases produced by
division-by-zero when a price series has no down-moves (or no movement at
all). If RSI/Z-score logic is ever consolidated with the duplicate
implementation in close_report.py, these tests should move/extend with it.
"""

import numpy as np
import pandas as pd
import pytest

from assets.technical_analysis import calculate_rsi, calculate_z_score


def test_z_score_known_values():
    series = pd.Series([1, 2, 3, 4, 5], dtype=float)
    result = calculate_z_score(series)
    expected = (series - series.mean()) / series.std()
    pd.testing.assert_series_equal(result, expected)


def test_z_score_constant_series_is_nan():
    series = pd.Series([5.0, 5.0, 5.0])
    result = calculate_z_score(series)
    assert result.isna().all()


def test_rsi_monotonically_increasing_series_saturates_at_100():
    series = pd.Series(range(1, 21), dtype=float)
    rsi = calculate_rsi(series, period=14)
    assert np.isnan(rsi.iloc[0])
    assert (rsi.iloc[1:] == 100).all()


def test_rsi_monotonically_decreasing_series_goes_to_zero():
    series = pd.Series(range(20, 0, -1), dtype=float)
    rsi = calculate_rsi(series, period=14)
    assert np.isnan(rsi.iloc[0])
    assert (rsi.iloc[1:] == 0).all()


def test_rsi_constant_series_is_nan_due_to_zero_division():
    series = pd.Series([10.0] * 10)
    rsi = calculate_rsi(series, period=14)
    assert rsi.isna().all()


def test_rsi_is_bounded_for_mixed_series():
    series = pd.Series([10, 11, 9, 12, 8, 13, 7, 14, 6, 15, 5, 16], dtype=float)
    rsi = calculate_rsi(series, period=14)
    valid = rsi.dropna()
    assert not valid.empty
    assert (valid >= 0).all() and (valid <= 100).all()
