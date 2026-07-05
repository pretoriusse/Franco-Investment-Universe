"""
Daily news-sentiment pipeline for JSE tickers.

Sources  : yfinance built-in news feed (no extra API key required).
Sentiment: VADER compound score in [-1, +1].
           Financial-domain lexicon boosts applied on top of the
           default VADER dictionary for words common in JSE reporting.

Outputs  : per-ticker daily aggregate written to `news_sentiment` table
           via database_queries.insert_sentiment_batch().

This module also exposes the shared sentiment primitives (`Article`,
`score_text`, `aggregate_daily_sentiment`, the configured analyzer) that the
macro/thematic pipeline in `assets.macro_sentiment` reuses.

Usage (from data_downloader.py):
    from assets.news_sentiment import run_daily_sentiment_pipeline
    run_daily_sentiment_pipeline(ticker_list)

Usage (ad-hoc, to fetch the current score for one ticker):
    from assets.news_sentiment import get_current_sentiment
    score = get_current_sentiment("ABG.JO")   # float in [-1, +1]
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, TypedDict

import numpy as np
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


class Article(TypedDict):
    """A single normalized, already-scored news item."""

    title: str
    summary: str
    publisher: str
    published_at: datetime  # timezone-aware (UTC)
    score: float


class SentimentAgg(TypedDict):
    """A daily aggregate sentiment record for one ticker or theme."""

    sentiment_score: float
    article_count: int
    positive_count: int
    negative_count: int
    neutral_count: int


# ---------------------------------------------------------------------------
# Financial-domain lexicon boosts
# Positive and negative words that carry extra weight in JSE/SA reporting.
# ---------------------------------------------------------------------------
_FINANCIAL_LEXICON: dict[str, float] = {
    # positive
    "dividend": 1.5,
    "dividends": 1.5,
    "profit": 1.2,
    "profits": 1.2,
    "earnings": 1.2,
    "upgrade": 2.0,
    "buyback": 1.5,
    "acquisition": 1.2,
    "merger": 1.0,
    "record": 1.0,
    "beat": 1.5,
    "rally": 2.0,
    "surplus": 1.2,
    "growth": 1.0,
    "expand": 0.8,
    # negative
    "downgrade": -2.0,
    "fraud": -3.0,
    "loss": -1.5,
    "losses": -1.5,
    "retrenchment": -1.5,
    "retrench": -1.5,
    "retrenchments": -1.5,
    "bankruptcy": -3.0,
    "default": -2.5,
    "sanction": -2.0,
    "sanctions": -2.0,
    "miss": -1.5,
    "slump": -2.0,
    "writedown": -1.8,
    "write-down": -1.8,
    "impairment": -1.5,
    "investigation": -1.2,
    "probe": -1.0,
    "recession": -1.5,
    "deficit": -1.0,
}

_analyzer = SentimentIntensityAnalyzer()
_analyzer.lexicon.update(_FINANCIAL_LEXICON)


# ---------------------------------------------------------------------------
# Core helpers (shared with assets.macro_sentiment)
# ---------------------------------------------------------------------------


def score_text(text: str) -> float:
    """Return the VADER compound score for a piece of text, in [-1, +1]."""
    if not text:
        return 0.0
    return float(_analyzer.polarity_scores(text)["compound"])


def _parse_published(item: dict[str, Any]) -> datetime | None:
    """Extract a timezone-aware publish datetime from a yfinance news item.

    Handles both the legacy flat schema (``providerPublishTime`` epoch) and
    the current nested schema (``content.pubDate`` ISO-8601 string). Returns
    ``None`` if no usable timestamp is present.
    """
    content = item.get("content") if isinstance(item.get("content"), dict) else item

    # Current schema: ISO-8601 string under content.pubDate / displayTime.
    iso = content.get("pubDate") or content.get("displayTime")
    if isinstance(iso, str) and iso:
        try:
            return datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except ValueError:
            pass

    # Legacy schema: epoch seconds under providerPublishTime.
    epoch = item.get("providerPublishTime") or content.get("providerPublishTime")
    if epoch:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc)

    return None


def _extract_title_summary(item: dict[str, Any]) -> tuple[str, str, str]:
    """Return (title, summary, publisher) from either yfinance news schema."""
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    title = str(content.get("title") or item.get("title") or "")
    summary = str(content.get("summary") or content.get("description") or "")
    provider = content.get("provider")
    if isinstance(provider, dict):
        publisher = str(provider.get("displayName") or "")
    else:
        publisher = str(item.get("publisher") or "")
    return title, summary, publisher


def fetch_news_for_ticker(ticker: str, lookback_days: int = 3) -> list[Article]:
    """
    Fetch recent news articles from yfinance for *ticker*.

    Only articles published within the last *lookback_days* are kept. If
    yfinance returns no news (common for .JO tickers) the list is empty; no
    exception is raised. Tolerant of both the legacy and current yfinance
    news schemas.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    articles: list[Article] = []

    try:
        raw_news = yf.Ticker(ticker).news or []
        for item in raw_news:
            if not isinstance(item, dict):
                continue
            published_at = _parse_published(item)
            if published_at is None or published_at < cutoff:
                continue
            title, summary, publisher = _extract_title_summary(item)
            if not title:
                continue
            articles.append(
                Article(
                    title=title,
                    summary=summary,
                    publisher=publisher,
                    published_at=published_at,
                    score=score_text(f"{title}. {summary}"),
                )
            )
    except Exception as exc:
        logger.warning(f"[news_sentiment] Could not fetch news for {ticker}: {exc}")

    return articles


def aggregate_daily_sentiment(articles: list[Article]) -> SentimentAgg:
    """
    Collapse a list of article dicts into a single daily sentiment record.

    Uses a recency-weighted mean so that articles published hours ago carry
    more weight than articles from three days ago.
    """
    if not articles:
        return SentimentAgg(
            sentiment_score=0.0,
            article_count=0,
            positive_count=0,
            negative_count=0,
            neutral_count=0,
        )

    now = datetime.now(timezone.utc)
    scores: list[float] = []
    weights: list[float] = []
    pos = neg = neu = 0

    for a in articles:
        s = a["score"]
        age_hours = max((now - a["published_at"]).total_seconds() / 3600, 0.1)
        w = 1.0 / (1.0 + age_hours / 24)  # exponential-ish decay over 24 h
        scores.append(s)
        weights.append(w)
        if s > 0.05:
            pos += 1
        elif s < -0.05:
            neg += 1
        else:
            neu += 1

    weighted_score = float(np.average(scores, weights=weights))
    return SentimentAgg(
        sentiment_score=round(weighted_score, 4),
        article_count=len(articles),
        positive_count=pos,
        negative_count=neg,
        neutral_count=neu,
    )


def get_current_sentiment(ticker: str, lookback_days: int = 3) -> float:
    """
    Convenience function: fetch and score news for *ticker* right now.
    Returns a float in [-1, +1].  Returns 0.0 if no articles are found.
    """
    articles = fetch_news_for_ticker(ticker, lookback_days=lookback_days)
    return aggregate_daily_sentiment(articles)["sentiment_score"]


# ---------------------------------------------------------------------------
# Pipeline entry point (called from data_downloader.py)
# ---------------------------------------------------------------------------


def run_daily_sentiment_pipeline(tickers: list[str]) -> None:
    """
    Fetch and store daily sentiment for every ticker in *tickers*.

    Designed to be called once per day (e.g. at 17:10 from data_downloader.py)
    after market close, so the sentiment covers the full trading day's news.
    """
    # Import here to avoid circular imports at module level
    from assets import database_queries as db_queries

    today = date.today()
    batch: list[dict[str, Any]] = []

    for ticker in tickers:
        try:
            articles = fetch_news_for_ticker(ticker)
            agg = aggregate_daily_sentiment(articles)
            batch.append({"ticker": ticker, "date": today, **agg})
            logger.info(
                f"[sentiment] {ticker}: score={agg['sentiment_score']:+.3f}  "
                f"articles={agg['article_count']} "
                f"(+{agg['positive_count']} -{agg['negative_count']} ~{agg['neutral_count']})"
            )
        except Exception as exc:
            logger.error(f"[sentiment] Pipeline error for {ticker}: {exc}")
            batch.append(
                {
                    "ticker": ticker,
                    "date": today,
                    "sentiment_score": 0.0,
                    "article_count": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 0,
                }
            )

    if batch:
        db_queries.insert_sentiment_batch(batch)
        logger.info(f"[sentiment] Stored sentiment for {len(batch)} tickers.")
