#!/usr/bin/env python3
import sys
import os
import re
import html
import requests
import pandas as pd
import json
from datetime import datetime, timezone
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET
from sqlalchemy import create_engine, text

sys.path.insert(0, "/opt/Franco-Investment-Universe")
from assets.const import DB_PARAMS
from assets import database_queries as dbq

POS = {
    "gain",
    "gains",
    "surge",
    "beat",
    "growth",
    "record",
    "upgrade",
    "bullish",
    "profit",
    "strong",
    "optimistic",
    "expands",
}
NEG = {
    "loss",
    "losses",
    "drop",
    "plunge",
    "miss",
    "downgrade",
    "bearish",
    "fraud",
    "recall",
    "default",
    "weak",
    "warning",
    "lawsuit",
}
WAR = {"war", "missile", "invasion", "attack", "conflict", "military", "ceasefire"}
SANCTIONS = {"sanction", "sanctions", "embargo", "tariff", "restriction", "ban"}
SUPPLY = {
    "oil",
    "gas",
    "shipping",
    "strait",
    "supply chain",
    "pipeline",
    "freight",
    "commodity",
}
MACRO = {
    "inflation",
    "interest rate",
    "fed",
    "ecb",
    "central bank",
    "recession",
    "gdp",
    "cpi",
    "yield",
}
COUNTRY_TENSION = {
    "russia",
    "ukraine",
    "china",
    "taiwan",
    "iran",
    "israel",
    "gaza",
    "usa",
    "united states",
    "nato",
    "yemen",
}


def tokenize(text: str):
    text = html.unescape(text or "").lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


USE_OLLAMA = os.getenv("USE_OLLAMA", "0") == "1"


def ollama_score(txt: str):
    try:
        payload = {
            "model": "gpt-oss:20b",
            "stream": False,
            "format": "json",
            "prompt": (
                "Return strict JSON with keys: sentiment, war_mentions, sanction_mentions, "
                "supply_mentions, macro_mentions, tension_mentions. sentiment in [-1,1], others integers >=0. "
                "Text: " + txt[:1800]
            ),
        }
        r = requests.post(
            "http://127.0.0.1:11434/api/generate", json=payload, timeout=8
        )
        if r.status_code != 200:
            return None
        out = r.json().get("response", "").strip()
        js = json.loads(out)
        return {
            "sentiment": float(js.get("sentiment", 0.0)),
            "war_mentions": int(js.get("war_mentions", 0)),
            "sanction_mentions": int(js.get("sanction_mentions", 0)),
            "supply_mentions": int(js.get("supply_mentions", 0)),
            "macro_mentions": int(js.get("macro_mentions", 0)),
            "tension_mentions": int(js.get("tension_mentions", 0)),
        }
    except Exception:
        return None


def score_text(txt: str):
    words = tokenize(txt)
    if not words:
        return {
            "sentiment": 0.0,
            "war_mentions": 0,
            "sanction_mentions": 0,
            "supply_mentions": 0,
            "macro_mentions": 0,
            "tension_mentions": 0,
        }
    wc = max(len(words), 1)
    pos = sum(w in POS for w in words)
    neg = sum(w in NEG for w in words)
    war = sum(w in WAR for w in words)
    san = sum(w in SANCTIONS for w in words)
    sup = sum(w in SUPPLY for w in words)
    mac = sum(w in MACRO for w in words)
    ten = sum(w in COUNTRY_TENSION for w in words)
    return {
        "sentiment": (pos - neg) / wc,
        "war_mentions": war,
        "sanction_mentions": san,
        "supply_mentions": sup,
        "macro_mentions": mac,
        "tension_mentions": ten,
    }


def parse_rss_items(xml_text: str):
    root = ET.fromstring(xml_text)
    out = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        desc = (item.findtext("description") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        link = (item.findtext("link") or "").strip()
        dt = None
        if pub:
            try:
                dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z").replace(
                    tzinfo=timezone.utc
                )
            except Exception:
                try:
                    dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %z").astimezone(
                        timezone.utc
                    )
                except Exception:
                    dt = None
        out.append(
            {"title": title, "description": desc, "published_at": dt, "url": link}
        )
    return out


def fetch_google_news(query: str):
    url = f"https://news.google.com/rss/search?q={quote_plus(query + ' when:7d')}&hl=en-ZA&gl=ZA&ceid=ZA:en"
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    return parse_rss_items(r.text)


def create_table(engine):
    ddl = """
    CREATE TABLE IF NOT EXISTS news_features_daily (
      ticker VARCHAR(16) NOT NULL,
      feature_date DATE NOT NULL,
      article_count INTEGER NOT NULL DEFAULT 0,
      avg_sentiment DOUBLE PRECISION,
      war_mentions INTEGER NOT NULL DEFAULT 0,
      sanction_mentions INTEGER NOT NULL DEFAULT 0,
      supply_shock_mentions INTEGER NOT NULL DEFAULT 0,
      macro_uncertainty_mentions INTEGER NOT NULL DEFAULT 0,
      geo_tension_mentions INTEGER NOT NULL DEFAULT 0,
      source VARCHAR(32) NOT NULL DEFAULT 'google_news_rss',
      created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
      PRIMARY KEY (ticker, feature_date, source)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def upsert_daily(engine, rows):
    if not rows:
        return 0
    sql = text("""
      INSERT INTO news_features_daily (
        ticker, feature_date, article_count, avg_sentiment,
        war_mentions, sanction_mentions, supply_shock_mentions,
        macro_uncertainty_mentions, geo_tension_mentions, source, updated_at
      ) VALUES (
        :ticker, :feature_date, :article_count, :avg_sentiment,
        :war_mentions, :sanction_mentions, :supply_shock_mentions,
        :macro_uncertainty_mentions, :geo_tension_mentions, :source, NOW()
      )
      ON CONFLICT (ticker, feature_date, source)
      DO UPDATE SET
        article_count = EXCLUDED.article_count,
        avg_sentiment = EXCLUDED.avg_sentiment,
        war_mentions = EXCLUDED.war_mentions,
        sanction_mentions = EXCLUDED.sanction_mentions,
        supply_shock_mentions = EXCLUDED.supply_shock_mentions,
        macro_uncertainty_mentions = EXCLUDED.macro_uncertainty_mentions,
        geo_tension_mentions = EXCLUDED.geo_tension_mentions,
        updated_at = NOW();
    """)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)


def build_for_ticker(ticker: str, share_name: str):
    q = f'("{share_name}" OR {ticker} OR {ticker.replace(".JO", "")}) stock OR shares OR earnings OR outlook OR sanctions OR war'
    items = fetch_google_news(q)
    if not items:
        return []

    dedup = {}
    for it in items:
        key = (it["url"] or "") + "|" + (it["title"] or "")
        dedup[key] = it
    items = list(dedup.values())

    rows = []
    for it in items:
        if not it["published_at"]:
            continue
        d = it["published_at"].date()
        txt = f"{it['title']} {it['description']}"
        s = (ollama_score(txt) if USE_OLLAMA else None) or score_text(txt)
        rows.append({"feature_date": d, **s})

    if not rows:
        return []

    df = pd.DataFrame(rows)
    agg = df.groupby("feature_date", as_index=False).agg(
        {
            "sentiment": "mean",
            "war_mentions": "sum",
            "sanction_mentions": "sum",
            "supply_mentions": "sum",
            "macro_mentions": "sum",
            "tension_mentions": "sum",
        }
    )
    cnt = df.groupby("feature_date").size().reset_index(name="article_count")
    agg = agg.merge(cnt, on="feature_date", how="left")

    out = []
    for _, r in agg.iterrows():
        out.append(
            {
                "ticker": ticker,
                "feature_date": r["feature_date"],
                "article_count": int(r["article_count"]),
                "avg_sentiment": float(r["sentiment"]),
                "war_mentions": int(r["war_mentions"]),
                "sanction_mentions": int(r["sanction_mentions"]),
                "supply_shock_mentions": int(r["supply_mentions"]),
                "macro_uncertainty_mentions": int(r["macro_mentions"]),
                "geo_tension_mentions": int(r["tension_mentions"]),
                "source": "google_news_rss",
            }
        )
    return out


def main(limit=30):
    engine = create_engine(
        f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
    )
    create_table(engine)

    universe = dbq.fetch_stock_universe_from_db().head(limit)
    total = 0
    for _, row in universe.iterrows():
        t = row["code"]
        n = row["share_name"]
        try:
            rows = build_for_ticker(t, n)
            total += upsert_daily(engine, rows)
            print(f"{t}: upserted={len(rows)}")
        except Exception as e:
            print(f"{t}: error={e}")

    print(f"TOTAL_UPSERTED={total}")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(lim)
