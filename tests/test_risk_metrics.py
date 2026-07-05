"""
Pinned-behavior tests for close_report.calculate_risk_metrics — the pandas
port of the Excel volatility/momentum/drawdown sheet formulas.
"""

import numpy as np
import pandas as pd
import pytest

from close_report import RISK_METRIC_COLS, calculate_risk_metrics


@pytest.fixture
def hist():
    n = 60
    rng = np.random.default_rng(0)
    close = pd.Series(100 + rng.normal(0, 1, n)).cumsum().abs() + 50
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="B"),
            "close": close,
            "Adj Close": close * 1.1,  # total-return close
            "high": close + 1,
            "low": close - 1,
            "volume": rng.integers(1000, 5000, n).astype(float),
        }
    )


def test_risk_metrics_match_formulas(hist):
    df = calculate_risk_metrics(hist.copy())
    last = df.iloc[-1]
    tr = hist["Adj Close"]

    # daily total return: today vs previous available row
    assert last["daily_tr_return"] == pytest.approx(tr.iloc[-1] / tr.iloc[-2] - 1)
    # 24-day return: today vs the observation 24 rows back (Excel TAKE -25)
    assert last["return_24d"] == pytest.approx(tr.iloc[-1] / tr.iloc[-25] - 1)
    # 24-day annualised vol: sample std of last 24 daily returns * sqrt(252)
    daily = tr.pct_change()
    assert last["vol_24d"] == pytest.approx(daily.iloc[-24:].std() * np.sqrt(252))
    # 55-day drawdown vs the max over the last 55 rows including today
    assert last["drawdown_55d"] == pytest.approx(
        tr.iloc[-55:].max() and tr.iloc[-1] / tr.iloc[-55:].max() - 1
    )
    # risk-adjusted momentum scaling
    assert last["risk_adj_mom_24d"] == pytest.approx(
        last["return_24d"] / (last["vol_24d"] / np.sqrt(252 / 24))
    )
    # true range covers the overnight gap vs previous close
    prev_close = hist["close"].iloc[-2]
    expected_tr = max(
        hist["high"].iloc[-1] - hist["low"].iloc[-1],
        abs(hist["high"].iloc[-1] - prev_close),
        abs(hist["low"].iloc[-1] - prev_close),
    )
    assert last["true_range"] == pytest.approx(expected_tr)
    assert last["atr_14_pct"] == pytest.approx(df["true_range_pct"].iloc[-14:].mean())
    # volume ratio and MA trend
    assert last["volume_ratio_24d"] == pytest.approx(
        hist["volume"].iloc[-1] / hist["volume"].iloc[-24:].mean()
    )
    assert last["ma_trend_24_55"] == pytest.approx(
        tr.iloc[-24:].mean() / tr.iloc[-55:].mean() - 1
    )


def test_failed_calculations_become_zero_never_nan_or_inf(hist):
    # Short history: 55-day windows can't be filled → 0, not NaN
    df = calculate_risk_metrics(hist.head(10).copy())
    assert df.iloc[-1]["vol_55d"] == 0.0
    assert df.iloc[-1]["return_24d"] == 0.0

    # Zero volume and flat prices: division by zero → 0, not inf/NaN
    flat = hist.copy()
    flat["volume"] = 0.0
    flat["close"] = 100.0
    flat["Adj Close"] = 110.0
    flat["high"] = flat["low"] = 100.0
    df = calculate_risk_metrics(flat)
    metrics = df[RISK_METRIC_COLS]
    assert np.isfinite(metrics.to_numpy()).all()
    assert df.iloc[-1]["volume_ratio_24d"] == 0.0
    assert df.iloc[-1]["vol_ratio_24_55"] == 0.0
    # every metric survives round() — this is what the report layer does
    for col in RISK_METRIC_COLS:
        round(df.iloc[-1][col], 4)
