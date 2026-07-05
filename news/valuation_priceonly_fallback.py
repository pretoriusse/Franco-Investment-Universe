#!/usr/bin/env python3
import sys, numpy as np
from sqlalchemy import create_engine, text

sys.path.insert(0, "/opt/Franco-Investment-Universe")
from assets.const import DB_PARAMS
from assets import database_queries as dbq

UPS = text("""
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
DO UPDATE SET close_price=EXCLUDED.close_price, updated_at=NOW();
""")

eng = create_engine(
    f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
)
missing = [x.strip() for x in open(sys.argv[1]) if x.strip()]
tot = 0
for t in missing:
    px = dbq.get_ticker_from_db_with_date_select(t, "2016-01-01", "2100-01-01")
    if px.empty:
        print(t, "no-price")
        continue
    rows = [
        {
            "ticker": t,
            "feature_date": r["date"],
            "close_price": float(r["close"]) if r["close"] is not None else np.nan,
            "shares_outstanding": np.nan,
            "market_cap": np.nan,
            "fcf_ttm": np.nan,
            "net_income_ttm": np.nan,
            "book_value_equity": np.nan,
            "fcf_per_share": np.nan,
            "earnings_per_share": np.nan,
            "book_value_per_share": np.nan,
            "fcf_yield": np.nan,
            "earnings_yield": np.nan,
            "pb_ratio": np.nan,
            "pe_ratio": np.nan,
            "beta": np.nan,
            "dcf_intrinsic_per_share": np.nan,
            "valuation_gap_pct": np.nan,
            "source": "price_only_fallback",
        }
        for _, r in px.iterrows()
    ]
    with eng.begin() as c:
        c.execute(UPS, rows)
    print(t, "upserted", len(rows))
    tot += len(rows)
print("TOTAL", tot)
