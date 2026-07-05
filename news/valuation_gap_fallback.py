#!/usr/bin/env python3
import sys, math
import numpy as np
import yfinance as yf
from sqlalchemy import create_engine, text

sys.path.insert(0, "/opt/Franco-Investment-Universe")
from assets.const import DB_PARAMS
from assets import database_queries as dbq

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
  pb_ratio=EXCLUDED.pb_ratio,
  pe_ratio=EXCLUDED.pe_ratio,
  beta=EXCLUDED.beta,
  updated_at=NOW();
""")


def v(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else np.nan
    except:
        return np.nan


def fetch_static(ticker):
    t = yf.Ticker(ticker)
    info = t.info or {}
    fi = getattr(t, "fast_info", {}) or {}
    shares = v(info.get("sharesOutstanding", np.nan))
    mcap = v(info.get("marketCap", fi.get("market_cap", np.nan)))
    beta = v(info.get("beta", np.nan))
    pe = v(info.get("trailingPE", np.nan))
    pb = v(info.get("priceToBook", np.nan))
    return shares, mcap, beta, pe, pb


def main(path):
    eng = create_engine(
        f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
    )
    tickers = [x.strip() for x in open(path) if x.strip()]
    total = 0
    for t in tickers:
        try:
            px = dbq.get_ticker_from_db_with_date_select(t, "2016-01-01", "2100-01-01")
            if px.empty:
                print(t, "no-price")
                continue
            shares, mcap, beta, pe, pb = fetch_static(t)
            rows = []
            for _, r in px.iterrows():
                close = v(r["close"])
                row = {
                    "ticker": t,
                    "feature_date": r["date"],
                    "close_price": close,
                    "shares_outstanding": shares,
                    "market_cap": mcap,
                    "fcf_ttm": np.nan,
                    "net_income_ttm": np.nan,
                    "book_value_equity": np.nan,
                    "fcf_per_share": np.nan,
                    "earnings_per_share": np.nan,
                    "book_value_per_share": np.nan,
                    "fcf_yield": np.nan,
                    "earnings_yield": np.nan,
                    "pb_ratio": pb,
                    "pe_ratio": pe,
                    "beta": beta,
                    "dcf_intrinsic_per_share": np.nan,
                    "valuation_gap_pct": np.nan,
                    "source": "yfinance_fallback",
                }
                rows.append(row)
            with eng.begin() as c:
                c.execute(UPSERT, rows)
            total += len(rows)
            print(t, "upserted", len(rows))
        except Exception as e:
            print(t, "error", e)
    print("TOTAL", total)


if __name__ == "__main__":
    main(sys.argv[1])
