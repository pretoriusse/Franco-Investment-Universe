#!/usr/bin/env python3
import sys
import math
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from sqlalchemy import create_engine, text

sys.path.insert(0, "/opt/Franco-Investment-Universe")
from assets.const import DB_PARAMS
from assets import database_queries as dbq

DDL = """
CREATE TABLE IF NOT EXISTS valuation_features_daily (
  ticker VARCHAR(16) NOT NULL,
  feature_date DATE NOT NULL,
  close_price DOUBLE PRECISION,
  shares_outstanding DOUBLE PRECISION,
  market_cap DOUBLE PRECISION,
  fcf_ttm DOUBLE PRECISION,
  net_income_ttm DOUBLE PRECISION,
  book_value_equity DOUBLE PRECISION,
  fcf_per_share DOUBLE PRECISION,
  earnings_per_share DOUBLE PRECISION,
  book_value_per_share DOUBLE PRECISION,
  fcf_yield DOUBLE PRECISION,
  earnings_yield DOUBLE PRECISION,
  pb_ratio DOUBLE PRECISION,
  pe_ratio DOUBLE PRECISION,
  beta DOUBLE PRECISION,
  dcf_intrinsic_per_share DOUBLE PRECISION,
  valuation_gap_pct DOUBLE PRECISION,
  source VARCHAR(32) NOT NULL DEFAULT 'yfinance_free',
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  PRIMARY KEY (ticker, feature_date, source)
);
"""

UPSERT = text("""
INSERT INTO valuation_features_daily (
  ticker,feature_date,close_price,shares_outstanding,market_cap,fcf_ttm,net_income_ttm,book_value_equity,
  fcf_per_share,earnings_per_share,book_value_per_share,fcf_yield,earnings_yield,pb_ratio,pe_ratio,beta,
  dcf_intrinsic_per_share,valuation_gap_pct,source,updated_at
) VALUES (
  :ticker,:feature_date,:close_price,:shares_outstanding,:market_cap,:fcf_ttm,:net_income_ttm,:book_value_equity,
  :fcf_per_share,:earnings_per_share,:book_value_per_share,:fcf_yield,:earnings_yield,:pb_ratio,:pe_ratio,:beta,
  :dcf_intrinsic_per_share,:valuation_gap_pct,:source,NOW()
)
ON CONFLICT (ticker, feature_date, source)
DO UPDATE SET
  close_price=EXCLUDED.close_price,
  shares_outstanding=EXCLUDED.shares_outstanding,
  market_cap=EXCLUDED.market_cap,
  fcf_ttm=EXCLUDED.fcf_ttm,
  net_income_ttm=EXCLUDED.net_income_ttm,
  book_value_equity=EXCLUDED.book_value_equity,
  fcf_per_share=EXCLUDED.fcf_per_share,
  earnings_per_share=EXCLUDED.earnings_per_share,
  book_value_per_share=EXCLUDED.book_value_per_share,
  fcf_yield=EXCLUDED.fcf_yield,
  earnings_yield=EXCLUDED.earnings_yield,
  pb_ratio=EXCLUDED.pb_ratio,
  pe_ratio=EXCLUDED.pe_ratio,
  beta=EXCLUDED.beta,
  dcf_intrinsic_per_share=EXCLUDED.dcf_intrinsic_per_share,
  valuation_gap_pct=EXCLUDED.valuation_gap_pct,
  updated_at=NOW();
""")


def val(x, d=np.nan):
    try:
        if x is None:
            return d
        x = float(x)
        if math.isfinite(x):
            return x
        return d
    except Exception:
        return d


def dcf_value_per_share(fcf_ttm, shares, growth=0.05, discount=0.14, terminal=0.04):
    if not fcf_ttm or not shares or fcf_ttm <= 0 or shares <= 0:
        return np.nan
    growth = max(min(growth, 0.25), -0.20)
    discount = max(discount, 0.08)
    terminal = min(terminal, discount - 0.01)

    fcf_year = fcf_ttm
    pv = 0.0
    for y in range(1, 6):
        fcf_year = fcf_year * (1 + growth)
        pv += fcf_year / ((1 + discount) ** y)

    tv = (fcf_year * (1 + terminal)) / (discount - terminal)
    pv += tv / ((1 + discount) ** 5)
    return pv / shares


def extract_series_row(df, labels):
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    for lab in labels:
        if lab in df.index:
            s = df.loc[lab]
            return pd.to_numeric(s, errors="coerce")
    return pd.Series(dtype="float64")


def build_fundamental_snapshots(ticker):
    tk = yf.Ticker(ticker)
    info = tk.info if tk.info else {}

    cf = tk.quarterly_cashflow
    fin = tk.quarterly_financials
    bs = tk.quarterly_balance_sheet

    fcf = extract_series_row(cf, ["Free Cash Flow"])
    if fcf.empty:
        ocf = extract_series_row(
            cf, ["Operating Cash Flow", "Total Cash From Operating Activities"]
        )
        capex = extract_series_row(cf, ["Capital Expenditure"])
        if not ocf.empty and not capex.empty:
            fcf = ocf + capex

    ni = extract_series_row(fin, ["Net Income", "Net Income Common Stockholders"])
    equity = extract_series_row(
        bs,
        [
            "Stockholders Equity",
            "Total Equity Gross Minority Interest",
            "Total Stockholder Equity",
        ],
    )

    cols = sorted(set(fcf.index).union(set(ni.index)).union(set(equity.index)))
    snaps = []
    for c in cols:
        dt = pd.to_datetime(c).date()
        snaps.append(
            {
                "report_date": dt,
                "fcf_ttm": val(fcf.get(c, np.nan)),
                "net_income_ttm": val(ni.get(c, np.nan)),
                "book_value_equity": val(equity.get(c, np.nan)),
                "shares_outstanding": val(info.get("sharesOutstanding", np.nan)),
                "beta": val(info.get("beta", np.nan)),
            }
        )

    # crude growth from last 2 valid FCF points
    valid_fcf = [
        s["fcf_ttm"] for s in snaps if pd.notna(s["fcf_ttm"]) and s["fcf_ttm"] > 0
    ]
    growth = 0.05
    if len(valid_fcf) >= 2 and valid_fcf[-2] != 0:
        growth = (valid_fcf[-1] / valid_fcf[-2]) - 1

    for s in snaps:
        s["dcf_intrinsic_per_share"] = dcf_value_per_share(
            s["fcf_ttm"],
            s["shares_outstanding"],
            growth=growth,
            discount=0.14,
            terminal=0.04,
        )

    return pd.DataFrame(snaps)


def build_daily_rows(ticker, start_date="2016-01-01"):
    px = dbq.get_ticker_from_db_with_date_select(
        ticker, start_date, date.today().isoformat()
    )
    if px.empty:
        return []

    px = px[["date", "close"]].copy()
    px["date"] = pd.to_datetime(px["date"]).dt.tz_localize(None).dt.date
    px = px.sort_values("date")

    snaps = build_fundamental_snapshots(ticker)
    if snaps.empty:
        return []

    snaps["report_date"] = pd.to_datetime(snaps["report_date"])
    px["date_dt"] = pd.to_datetime(px["date"])

    merged = pd.merge_asof(
        px.sort_values("date_dt"),
        snaps.sort_values("report_date"),
        left_on="date_dt",
        right_on="report_date",
        direction="backward",
    )

    rows = []
    for _, r in merged.iterrows():
        close = val(r["close"])
        shares = val(r.get("shares_outstanding"))
        fcf = val(r.get("fcf_ttm"))
        ni = val(r.get("net_income_ttm"))
        book = val(r.get("book_value_equity"))
        intrinsic = val(r.get("dcf_intrinsic_per_share"))

        mcap = close * shares if pd.notna(close) and pd.notna(shares) else np.nan
        fcf_ps = (
            fcf / shares
            if pd.notna(fcf) and pd.notna(shares) and shares > 0
            else np.nan
        )
        eps = (
            ni / shares if pd.notna(ni) and pd.notna(shares) and shares > 0 else np.nan
        )
        bvps = (
            book / shares
            if pd.notna(book) and pd.notna(shares) and shares > 0
            else np.nan
        )

        fcf_yield = (
            fcf / mcap if pd.notna(fcf) and pd.notna(mcap) and mcap != 0 else np.nan
        )
        earnings_yield = (
            ni / mcap if pd.notna(ni) and pd.notna(mcap) and mcap != 0 else np.nan
        )
        pb = (
            close / bvps if pd.notna(close) and pd.notna(bvps) and bvps != 0 else np.nan
        )
        pe = close / eps if pd.notna(close) and pd.notna(eps) and eps != 0 else np.nan
        gap = (
            ((intrinsic - close) / close)
            if pd.notna(intrinsic) and pd.notna(close) and close != 0
            else np.nan
        )

        rows.append(
            {
                "ticker": ticker,
                "feature_date": r["date"],
                "close_price": close,
                "shares_outstanding": shares,
                "market_cap": mcap,
                "fcf_ttm": fcf,
                "net_income_ttm": ni,
                "book_value_equity": book,
                "fcf_per_share": fcf_ps,
                "earnings_per_share": eps,
                "book_value_per_share": bvps,
                "fcf_yield": fcf_yield,
                "earnings_yield": earnings_yield,
                "pb_ratio": pb,
                "pe_ratio": pe,
                "beta": val(r.get("beta")),
                "dcf_intrinsic_per_share": intrinsic,
                "valuation_gap_pct": gap,
                "source": "yfinance_free",
            }
        )
    return rows


def main(limit=25, start_date="2016-01-01"):
    eng = create_engine(
        f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
    )
    with eng.begin() as c:
        c.execute(text(DDL))

    uni = dbq.fetch_stock_universe_from_db().head(limit)
    total = 0
    for _, r in uni.iterrows():
        t = r["code"]
        try:
            rows = build_daily_rows(t, start_date=start_date)
            if rows:
                with eng.begin() as c:
                    c.execute(UPSERT, rows)
                print(f"{t}: upserted={len(rows)}")
                total += len(rows)
            else:
                print(f"{t}: no-data")
        except Exception as e:
            print(f"{t}: error={e}")

    print(f"TOTAL_UPSERTED={total}")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    start = sys.argv[2] if len(sys.argv) > 2 else "2016-01-01"
    main(lim, start)
