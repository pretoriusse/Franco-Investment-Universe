"""
Daily macro / thematic news-sentiment pipeline.

Unlike `assets.news_sentiment` (which scores news that names a specific
ticker), this module ingests *broad* financial and geopolitical news from
multiple sources, classifies each article into one or more macro **themes**
(commodities, sectors, currency, global-risk, and an overall MARKET theme),
and stores one recency-weighted daily sentiment score per theme.

The goal is to give the price models the macro *drivers* as input features
(e.g. an oil shock from a Strait-of-Hormuz closure, a rand sell-off, a global
risk-off event) so they can learn cross-asset relationships such as
"oil up -> fuel producers up / refiners squeezed".

Sources
-------
* RSS feeds        - free, no key (feedparser).
* GDELT DOC 2.0    - free, no key; global event coverage.
* NewsAPI.org      - optional; only used when NEWSAPI_KEY is configured.

Sentiment scoring reuses the VADER analyzer and recency-weighted aggregator
from `assets.news_sentiment`.

Usage (from data_downloader.py):
    from assets.macro_sentiment import run_daily_macro_sentiment_pipeline
    run_daily_macro_sentiment_pipeline()
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import feedparser
import requests

from assets.const import NEWSAPI_KEY
from assets.news_sentiment import Article, aggregate_daily_sentiment, score_text

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 15  # seconds
_HEADERS = {"User-Agent": "FrancoInvestmentUniverse/1.0 (sentiment pipeline)"}
_DEFAULT_LOOKBACK_DAYS = 3


# ---------------------------------------------------------------------------
# Theme taxonomy
# ---------------------------------------------------------------------------
# An article is assigned to every theme whose keywords it matches; the MARKET
# theme always receives all articles. Keywords are matched case-insensitively
# as substrings of "<title>. <summary>".
MARKET_THEME = "MARKET"

THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    # --- commodities -------------------------------------------------------
    "OIL": (
        "oil",
        "crude",
        "brent",
        "wti",
        "opec",
        "petroleum",
        "refinery",
        "refiner",
        "fuel",
        "diesel",
        "gasoline",
        "strait of hormuz",
        "suez",
        "pipeline",
    ),
    "GOLD": ("gold", "bullion"),
    "PLATINUM": ("platinum", "palladium", "pgm", "rhodium"),
    "COAL": ("coal", "thermal coal", "coking coal"),
    "COPPER": ("copper",),
    "IRON_ORE": ("iron ore", "iron-ore"),
    # --- currency / rates --------------------------------------------------
    "RAND": (
        "rand",
        " zar",
        "south african currency",
        "reserve bank",
        "sarb",
    ),
    # --- sectors -----------------------------------------------------------
    "FINANCIALS": (
        "bank",
        "banks",
        "banking",
        "insurer",
        "insurance",
        "lender",
        "interest rate",
        "repo rate",
        "financial services",
        "credit",
    ),
    "MINING": (
        "mining",
        "miner",
        "mine ",
        "resources sector",
        "commodities",
        "smelter",
    ),
    "RETAIL": (
        "retail",
        "retailer",
        "consumer spending",
        "consumer confidence",
        "same-store sales",
    ),
    "ENERGY": (
        "eskom",
        "electricity",
        "load shedding",
        "loadshedding",
        "power cut",
        "renewable",
        "grid",
        "blackout",
    ),
    "TELCO": ("telecom", "telecoms", "mobile operator", "spectrum", "data prices"),
    "PROPERTY": ("property", "real estate", "reit", "landlord"),
    "INDUSTRIAL": ("manufacturing", "industrial production", "factory", "pmi"),
    # --- global macro / geopolitics ---------------------------------------
    "GLOBAL_RISK": (
        "war",
        "conflict",
        "sanction",
        "sanctions",
        "tariff",
        "tariffs",
        "geopolitical",
        "recession",
        "inflation",
        "federal reserve",
        "rate hike",
        "rate cut",
        "crisis",
        "default",
    ),
}

# All themes that can be produced/stored, including the catch-all MARKET.
ALL_THEMES: tuple[str, ...] = (MARKET_THEME, *THEME_KEYWORDS.keys())

# Broad queries issued to the keyword-based APIs (GDELT, NewsAPI). Kept small
# to limit request volume; fine-grained theme assignment happens locally.
_BROAD_QUERIES: tuple[str, ...] = (
    "oil OR crude OR OPEC OR brent",
    "gold OR platinum OR palladium OR copper OR coal price",
    "South Africa economy OR rand currency OR Reserve Bank",
    "geopolitical OR conflict OR sanctions OR tariffs OR inflation markets",
)

# RSS feeds mixing South-African and global financial/commodity coverage.
# Dead or malformed feeds are skipped at runtime (logged, never fatal).
_RSS_FEEDS: tuple[str, ...] = (
    "https://www.moneyweb.co.za/feed/",
    "https://mg.co.za/section/business/feed/",
    "https://www.investing.com/rss/news_25.rss",  # commodities
    "https://www.investing.com/rss/news_301.rss",  # economy
    "https://oilprice.com/rss/main",
    "https://www.miningweekly.com/page/rss",
)


# ---------------------------------------------------------------------------
# Article construction
# ---------------------------------------------------------------------------


def _make_article(
    title: str, summary: str, publisher: str, published_at: datetime
) -> Article:
    """Build a scored :class:`Article` from raw fields."""
    return Article(
        title=title,
        summary=summary,
        publisher=publisher,
        published_at=published_at,
        score=score_text(f"{title}. {summary}"),
    )


# ---------------------------------------------------------------------------
# Source adapters - each returns a list of scored Articles, never raises.
# ---------------------------------------------------------------------------


def _fetch_rss(lookback_days: int) -> list[Article]:
    """Fetch and score articles from the configured RSS feeds."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    articles: list[Article] = []

    for url in _RSS_FEEDS:
        try:
            parsed = feedparser.parse(url, request_headers=_HEADERS)
            source = str(getattr(parsed.feed, "title", "")) or url
            for entry in parsed.entries:
                published_at = _parse_struct_time(
                    getattr(entry, "published_parsed", None)
                )
                if published_at is None:
                    published_at = datetime.now(timezone.utc)
                if published_at < cutoff:
                    continue
                title = str(getattr(entry, "title", "") or "")
                summary = str(getattr(entry, "summary", "") or "")
                if not title:
                    continue
                articles.append(_make_article(title, summary, source, published_at))
        except Exception as exc:
            logger.warning(f"[macro_sentiment] RSS feed failed ({url}): {exc}")

    return articles


def _fetch_gdelt(lookback_days: int) -> list[Article]:
    """Fetch and score articles from the GDELT DOC 2.0 API."""
    articles: list[Article] = []
    timespan = f"{max(lookback_days, 1) * 24}h"

    for query in _BROAD_QUERIES:
        try:
            resp = requests.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={
                    "query": f"{query} sourcelang:english",
                    "mode": "ArtList",
                    "format": "json",
                    "maxrecords": "75",
                    "timespan": timespan,
                },
                headers=_HEADERS,
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            payload: Any = resp.json()
            for art in payload.get("articles", []) or []:
                title = str(art.get("title") or "")
                if not title:
                    continue
                published_at = _parse_gdelt_date(str(art.get("seendate") or ""))
                articles.append(
                    _make_article(title, "", str(art.get("domain") or ""), published_at)
                )
            time.sleep(1.0)  # be polite to the public endpoint
        except Exception as exc:
            logger.warning(f"[macro_sentiment] GDELT query failed ({query!r}): {exc}")

    return articles


def _fetch_newsapi(lookback_days: int) -> list[Article]:
    """Fetch and score articles from NewsAPI.org (skipped without a key)."""
    if not NEWSAPI_KEY:
        logger.info("[macro_sentiment] NEWSAPI_KEY not set - skipping NewsAPI source.")
        return []

    articles: list[Article] = []
    from_date = (date.today() - timedelta(days=lookback_days)).isoformat()

    for query in _BROAD_QUERIES:
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "from": from_date,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": "100",
                },
                headers={**_HEADERS, "X-Api-Key": NEWSAPI_KEY},
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            payload: Any = resp.json()
            for art in payload.get("articles", []) or []:
                title = str(art.get("title") or "")
                if not title:
                    continue
                summary = str(art.get("description") or "")
                published_at = _parse_iso_date(str(art.get("publishedAt") or ""))
                source_obj = art.get("source") or {}
                publisher = (
                    str(source_obj.get("name") or "")
                    if isinstance(source_obj, dict)
                    else ""
                )
                articles.append(_make_article(title, summary, publisher, published_at))
        except Exception as exc:
            logger.warning(f"[macro_sentiment] NewsAPI query failed ({query!r}): {exc}")

    return articles


# ---------------------------------------------------------------------------
# Date parsing helpers
# ---------------------------------------------------------------------------


def _parse_struct_time(value: time.struct_time | None) -> datetime | None:
    """Convert a feedparser ``*_parsed`` struct_time (UTC) to a datetime."""
    if value is None:
        return None
    try:
        return datetime(*value[:6], tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _parse_gdelt_date(value: str) -> datetime:
    """Parse a GDELT ``seendate`` (``YYYYMMDDTHHMMSSZ``); fall back to now."""
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def _parse_iso_date(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; fall back to now (UTC)."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Classification + aggregation
# ---------------------------------------------------------------------------


def _classify(article: Article) -> set[str]:
    """Return the set of themes an article belongs to (always incl. MARKET)."""
    text = f"{article['title']}. {article['summary']}".lower()
    themes = {MARKET_THEME}
    for theme, keywords in THEME_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            themes.add(theme)
    return themes


def _dedupe(articles: list[Article]) -> list[Article]:
    """Drop duplicate articles sharing a normalized title (cross-source)."""
    seen: set[str] = set()
    unique: list[Article] = []
    for a in articles:
        key = a["title"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(a)
    return unique


def collect_macro_articles(
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> list[Article]:
    """Gather, dedupe and score macro articles from all available sources."""
    articles: list[Article] = []
    articles.extend(_fetch_rss(lookback_days))
    articles.extend(_fetch_gdelt(lookback_days))
    articles.extend(_fetch_newsapi(lookback_days))
    unique = _dedupe(articles)
    logger.info(
        f"[macro_sentiment] Collected {len(unique)} unique articles from all sources."
    )
    return unique


def aggregate_by_theme(articles: list[Article]) -> dict[str, Any]:
    """Aggregate scored articles into one daily record per matched theme.

    Returns a mapping ``theme -> SentimentAgg``. Themes with no matching
    article on the day are omitted (callers may treat them as neutral 0.0).
    """
    buckets: dict[str, list[Article]] = {}
    for article in articles:
        for theme in _classify(article):
            buckets.setdefault(theme, []).append(article)

    return {theme: aggregate_daily_sentiment(group) for theme, group in buckets.items()}


# ---------------------------------------------------------------------------
# Ticker -> theme mapping (consumed by the training feature builder)
# ---------------------------------------------------------------------------

# Commodity-future / ETF tickers mapped to their dominant commodity theme.
_COMMODITY_TICKER_THEMES: dict[str, str] = {
    "CL=F": "OIL",
    "BZ=F": "OIL",
    "XC=F": "OIL",
    "GC=F": "GOLD",
    "PL=F": "PLATINUM",
    "PA=F": "PLATINUM",
    "HG=F": "COPPER",
    "SB=F": "GLOBAL_RISK",  # sugar - no dedicated theme; ride the market signal
    "DJP": "GLOBAL_RISK",  # broad commodity ETN
    "XRH0.L": "PLATINUM",
}

# Industry/sub-industry keyword -> sector theme.
_SECTOR_KEYWORDS: dict[str, str] = {
    "bank": "FINANCIALS",
    "financ": "FINANCIALS",
    "insur": "FINANCIALS",
    "mining": "MINING",
    "metal": "MINING",
    "resource": "MINING",
    "gold": "GOLD",
    "platinum": "PLATINUM",
    "coal": "COAL",
    "oil": "OIL",
    "gas": "OIL",
    "energy": "ENERGY",
    "retail": "RETAIL",
    "consumer": "RETAIL",
    "telecom": "TELCO",
    "property": "PROPERTY",
    "real estate": "PROPERTY",
    "industrial": "INDUSTRIAL",
}


def themes_for_stock(
    code: str, industry: str | None, sub_industry: str | None
) -> list[str]:
    """Return the macro themes relevant to a given stock/commodity ticker.

    Always includes MARKET and RAND (rand moves affect virtually every JSE
    name); adds a commodity theme for known commodity tickers and a sector
    theme inferred from the industry/sub-industry text.
    """
    themes = {MARKET_THEME, "RAND"}

    if code in _COMMODITY_TICKER_THEMES:
        themes.add(_COMMODITY_TICKER_THEMES[code])

    haystack = f"{industry or ''} {sub_industry or ''}".lower()
    for keyword, theme in _SECTOR_KEYWORDS.items():
        if keyword in haystack:
            themes.add(theme)

    return sorted(themes)


# ---------------------------------------------------------------------------
# Pipeline entry point (called from data_downloader.py)
# ---------------------------------------------------------------------------


def run_daily_macro_sentiment_pipeline(
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> None:
    """Collect macro news, aggregate per theme, and persist today's scores."""
    from assets import database_queries as db_queries

    today = date.today()
    articles = collect_macro_articles(lookback_days)
    per_theme = aggregate_by_theme(articles)

    batch: list[dict[str, Any]] = []
    for theme in ALL_THEMES:
        agg = per_theme.get(theme)
        if agg is None:
            batch.append(
                {
                    "theme": theme,
                    "date": today,
                    "sentiment_score": 0.0,
                    "article_count": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 0,
                }
            )
        else:
            batch.append({"theme": theme, "date": today, **agg})
        record = batch[-1]
        logger.info(
            f"[macro] {theme}: score={record['sentiment_score']:+.3f} "
            f"articles={record['article_count']}"
        )

    if batch:
        db_queries.insert_macro_sentiment_batch(batch)
        logger.info(f"[macro] Stored macro sentiment for {len(batch)} themes.")
