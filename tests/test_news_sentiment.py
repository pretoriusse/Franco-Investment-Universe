"""
Unit tests for assets.news_sentiment.

fetch_news_for_ticker / run_daily_sentiment_pipeline talk to yfinance and the
database respectively, so they're exercised here only through monkeypatched
boundaries. The scoring/aggregation math itself is pure and tested directly.
"""
from datetime import datetime, timedelta

import pytest

from assets.news_sentiment import (
    _score_text,
    aggregate_daily_sentiment,
    fetch_news_for_ticker,
    get_current_sentiment,
)


def test_score_text_empty_string_is_neutral():
    assert _score_text("") == 0.0
    assert _score_text(None) == 0.0


def test_score_text_financial_lexicon_boosts_positive_terms():
    plain = _score_text("Company reports results")
    boosted = _score_text("Company reports record profits and a rally")
    assert boosted > plain
    assert boosted > 0


def test_score_text_financial_lexicon_boosts_negative_terms():
    score = _score_text("Company faces bankruptcy and fraud investigation")
    assert score < -0.5


def test_aggregate_daily_sentiment_empty_list():
    result = aggregate_daily_sentiment([])
    assert result == {
        "sentiment_score": 0.0,
        "article_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "neutral_count": 0,
    }


def test_aggregate_daily_sentiment_counts_and_weighting():
    now = datetime.utcnow()
    articles = [
        {"title": "a", "score": 0.8, "published_at": now},  # fresh, positive
        {"title": "b", "score": -0.6, "published_at": now - timedelta(hours=72)},  # stale, negative
        {"title": "c", "score": 0.0, "published_at": now},  # neutral
    ]
    result = aggregate_daily_sentiment(articles)

    assert result["article_count"] == 3
    assert result["positive_count"] == 1
    assert result["negative_count"] == 1
    assert result["neutral_count"] == 1
    # The fresh positive article is weighted more heavily than the 3-day-old
    # negative one, so the aggregate should land on the positive side.
    assert result["sentiment_score"] > 0


def test_fetch_news_for_ticker_filters_by_lookback_window(monkeypatch):
    now_ts = datetime.utcnow().timestamp()
    fresh = {"title": "Fresh news", "publisher": "X", "providerPublishTime": now_ts}
    stale = {
        "title": "Stale news",
        "publisher": "X",
        "providerPublishTime": now_ts - timedelta(days=10).total_seconds(),
    }

    class FakeTicker:
        def __init__(self, _ticker):
            self.news = [fresh, stale]

    monkeypatch.setattr("assets.news_sentiment.yf.Ticker", FakeTicker)

    articles = fetch_news_for_ticker("ABG.JO", lookback_days=3)

    assert len(articles) == 1
    assert articles[0]["title"] == "Fresh news"


def test_fetch_news_for_ticker_swallows_exceptions(monkeypatch):
    class BrokenTicker:
        def __init__(self, _ticker):
            raise RuntimeError("network error")

    monkeypatch.setattr("assets.news_sentiment.yf.Ticker", BrokenTicker)

    assert fetch_news_for_ticker("ABG.JO") == []


def test_get_current_sentiment_uses_fetch_and_aggregate(monkeypatch):
    fake_articles = [{"title": "x", "score": 0.5, "published_at": datetime.utcnow()}]
    monkeypatch.setattr(
        "assets.news_sentiment.fetch_news_for_ticker", lambda ticker, lookback_days=3: fake_articles
    )

    score = get_current_sentiment("ABG.JO")

    assert score == pytest.approx(0.5)


def test_get_current_sentiment_returns_zero_with_no_articles(monkeypatch):
    monkeypatch.setattr(
        "assets.news_sentiment.fetch_news_for_ticker", lambda ticker, lookback_days=3: []
    )
    assert get_current_sentiment("ABG.JO") == 0.0
