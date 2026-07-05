"""Typed value objects shared by the close / adjusted-close report pipelines.

These dataclasses replace the loosely-typed dictionaries and bare tuples that
used to be passed between the report stages, so the data flowing through
``close_report.py`` and ``adjusted_close_report.py`` is statically checked
under ``mypy --strict``.

Rendering note: the report templates access image fields by attribute
(``stockimg.bollinger`` etc.). Jinja falls back to an empty string for any
attribute a dataclass does not define, so the chart dataclasses below only
declare the fields their respective templates actually consume — matching the
behaviour of the plain dicts they replaced.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import pandas as pd

# A single ranked row as produced by ``DataFrame.to_dict(orient="records")``.
# Keys are ``Hashable`` to match pandas' own typing for record dicts.
StockRecord = dict[Hashable, Any]

# ``{metric_name: {"top_10": [...], "bottom_10": [...]}}`` as consumed by the
# summary template and the e-mail renderer.
TopBottomData = dict[str, dict[str, list[StockRecord]]]


@dataclass(frozen=True)
class ModelHyperParams:
    """Hyper-parameters for the per-ticker LSTM price model.

    Defaults mirror the production configuration; ``epochs`` is overridden to a
    small value when ``DEBUGGING`` is enabled.
    """

    lstm_units: int = 400
    dropout: float = 0.3
    epochs: int = 200


@dataclass(frozen=True)
class PredictionResult:
    """Output of a single ticker's LSTM roll-forward.

    Prices are in rand (inverse-scaled). ``*_path`` hold the full day-by-day
    prediction series used for the prediction chart.
    """

    next_week_price: float
    next_month_price: float
    next_week_path: list[float]
    next_month_path: list[float]


@dataclass(frozen=True)
class SentimentAdjustment:
    """LSTM prices after the news-sentiment bias has been applied.

    ``sentiment_score`` is the VADER compound score in ``[-1, +1]`` that drove
    the adjustment (``0.0`` when there is no news or the fetch failed).
    """

    next_week_price: float
    next_month_price: float
    sentiment_score: float


@dataclass(frozen=True)
class RSIComparison:
    """Price-relative strength of a stock vs its sector and market benchmarks.

    Each value is the stock's N-day price ratio divided by the benchmark's ratio
    over the same window (``1.0`` = moved in line with the benchmark). Any leg
    that cannot be computed defaults to ``0.0``.
    """

    rsi_1m_sector: float = 0.0
    rsi_3m_sector: float = 0.0
    rsi_6m_sector: float = 0.0
    rsi_1m_market: float = 0.0
    rsi_3m_market: float = 0.0
    rsi_6m_market: float = 0.0


@dataclass(frozen=True)
class StockChartImages:
    """Base64-encoded PNG charts embedded in the detailed / web reports.

    ``volume_prediction`` is populated for equities only; it is ``None`` for
    commodities, which have no volume chart.
    """

    code: str
    name: str
    adj_prediction: str
    bollinger: str
    overbought_oversold: str
    volume_prediction: str | None = None


@dataclass(frozen=True)
class RankedStockImage:
    """Base64-encoded PNG charts for a ticker in a summary top/bottom-10 list.

    Fields are optional because a chart file may be missing on disk, in which
    case the encoder returns ``None`` and the template renders nothing.
    """

    name: str
    ticker: str
    prediction: str | None
    bollinger: str | None
    overbought_oversold: str | None


@dataclass
class FetchResult:
    """Everything ``fetch_data`` produces for the downstream report layer.

    ``total_value_next_*`` are the projected portfolio values (predicted price ×
    shares held, summed across the universe) at the 7- and 30-day horizons.
    """

    stocks: pd.DataFrame
    images: list[StockChartImages]
    total_value_next_week: float
    total_value_next_month: float
