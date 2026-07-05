#!/usr/bin/env python3
import sys, math
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import tensorflow as tf

sys.path.insert(0, "/opt/Franco-Investment-Universe")
from assets.const import DB_PARAMS

ENGINE = create_engine(
    f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
)

# Prefer GPU when available
try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass


def train_tf_regressor(X_train, y_train, input_dim):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber")
    es = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True
    )
    model.fit(X_train, y_train, epochs=40, batch_size=256, verbose=0, callbacks=[es])
    return model


DDL1 = """
CREATE TABLE IF NOT EXISTS valuation_dcf_daily (
  ticker VARCHAR(16) NOT NULL,
  feature_date DATE NOT NULL,
  fcf_base DOUBLE PRECISION,
  growth_assumption DOUBLE PRECISION,
  discount_rate DOUBLE PRECISION,
  terminal_growth DOUBLE PRECISION,
  pv_5y_cashflows DOUBLE PRECISION,
  pv_terminal_value DOUBLE PRECISION,
  enterprise_value DOUBLE PRECISION,
  intrinsic_value_per_share DOUBLE PRECISION,
  valuation_gap_pct DOUBLE PRECISION,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  PRIMARY KEY (ticker, feature_date)
);
"""

DDL2 = """
CREATE TABLE IF NOT EXISTS valuation_news_model_outputs_daily (
  ticker VARCHAR(16) NOT NULL,
  feature_date DATE NOT NULL,
  pred_next_day_return DOUBLE PRECISION,
  pred_next_day_abs_return DOUBLE PRECISION,
  model_version VARCHAR(64) NOT NULL,
  trained_rows INTEGER,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  PRIMARY KEY (ticker, feature_date, model_version)
);
"""

UP_DCF = text("""
INSERT INTO valuation_dcf_daily (
  ticker,feature_date,fcf_base,growth_assumption,discount_rate,terminal_growth,
  pv_5y_cashflows,pv_terminal_value,enterprise_value,intrinsic_value_per_share,valuation_gap_pct,updated_at
) VALUES (
  :ticker,:feature_date,:fcf_base,:growth_assumption,:discount_rate,:terminal_growth,
  :pv_5y_cashflows,:pv_terminal_value,:enterprise_value,:intrinsic_value_per_share,:valuation_gap_pct,NOW()
)
ON CONFLICT (ticker,feature_date)
DO UPDATE SET
  fcf_base=EXCLUDED.fcf_base,
  growth_assumption=EXCLUDED.growth_assumption,
  discount_rate=EXCLUDED.discount_rate,
  terminal_growth=EXCLUDED.terminal_growth,
  pv_5y_cashflows=EXCLUDED.pv_5y_cashflows,
  pv_terminal_value=EXCLUDED.pv_terminal_value,
  enterprise_value=EXCLUDED.enterprise_value,
  intrinsic_value_per_share=EXCLUDED.intrinsic_value_per_share,
  valuation_gap_pct=EXCLUDED.valuation_gap_pct,
  updated_at=NOW();
""")

UP_PRED = text("""
INSERT INTO valuation_news_model_outputs_daily (
  ticker,feature_date,pred_next_day_return,pred_next_day_abs_return,model_version,trained_rows,updated_at
) VALUES (
  :ticker,:feature_date,:pred_next_day_return,:pred_next_day_abs_return,:model_version,:trained_rows,NOW()
)
ON CONFLICT (ticker,feature_date,model_version)
DO UPDATE SET
  pred_next_day_return=EXCLUDED.pred_next_day_return,
  pred_next_day_abs_return=EXCLUDED.pred_next_day_abs_return,
  trained_rows=EXCLUDED.trained_rows,
  updated_at=NOW();
""")


def safe(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else np.nan
    except:
        return np.nan


def calc_dcf_row(row, growth_default=0.05):
    close = safe(row["close_price"])
    shares = safe(row["shares_outstanding"])
    mcap = safe(row.get("market_cap"))
    fcf = safe(row["fcf_ttm"])
    ni = safe(row["net_income_ttm"])
    beta = safe(row["beta"])
    pe = safe(row.get("pe_ratio"))
    pb = safe(row.get("pb_ratio"))
    e_yield = safe(row.get("earnings_yield"))

    if np.isnan(mcap) and (not np.isnan(close)) and (not np.isnan(shares)):
        mcap = close * shares

    if np.isnan(fcf) or fcf <= 0:
        if not np.isnan(ni) and ni > 0:
            fcf = ni * 0.75
        elif not np.isnan(e_yield) and not np.isnan(mcap) and mcap > 0:
            ni = e_yield * mcap
            fcf = ni * 0.75
        elif (
            not np.isnan(pe)
            and pe > 0
            and not np.isnan(close)
            and not np.isnan(shares)
            and shares > 0
        ):
            eps = close / pe
            ni = eps * shares
            fcf = ni * 0.75
        elif (
            not np.isnan(pb)
            and pb > 0
            and not np.isnan(close)
            and not np.isnan(shares)
            and shares > 0
        ):
            bvps = close / pb
            book_equity = bvps * shares
            ni = book_equity * 0.12  # assumed ROE proxy
            fcf = ni * 0.70
        elif not np.isnan(mcap) and mcap > 0:
            fcf = mcap * 0.06  # fallback normalized FCF yield
        else:
            return None

    if np.isnan(shares) or shares <= 0:
        return None

    # SA-oriented discount rate approximation
    rf = 0.10
    erp = 0.055
    country_risk = 0.025
    beta = 1.0 if np.isnan(beta) else min(max(beta, 0.4), 2.2)
    discount = rf + beta * erp + country_risk
    discount = min(max(discount, 0.11), 0.28)

    growth = safe(row.get("growth_assumption", np.nan))
    if np.isnan(growth):
        growth = growth_default
    growth = min(max(growth, -0.15), 0.20)
    terminal = 0.04 if discount > 0.05 else 0.02
    if terminal >= discount:
        terminal = discount - 0.01

    f = fcf
    pv5 = 0.0
    for y in range(1, 6):
        f = f * (1 + growth)
        pv5 += f / ((1 + discount) ** y)

    tv = (f * (1 + terminal)) / max((discount - terminal), 1e-6)
    pv_tv = tv / ((1 + discount) ** 5)
    ev = pv5 + pv_tv
    intrinsic = ev / shares
    gap = (
        (intrinsic - close) / close if (not np.isnan(close) and close != 0) else np.nan
    )

    return {
        "fcf_base": fcf,
        "growth_assumption": growth,
        "discount_rate": discount,
        "terminal_growth": terminal,
        "pv_5y_cashflows": pv5,
        "pv_terminal_value": pv_tv,
        "enterprise_value": ev,
        "intrinsic_value_per_share": intrinsic,
        "valuation_gap_pct": gap,
    }


def process_ticker(ticker):
    q = text("""
      SELECT v.ticker, v.feature_date, v.close_price, v.shares_outstanding, v.fcf_ttm, v.net_income_ttm, v.beta,
             v.market_cap, v.pe_ratio, v.pb_ratio, v.fcf_yield, v.earnings_yield,
             n.article_count, n.avg_sentiment, n.war_mentions, n.sanction_mentions, n.supply_shock_mentions,
             g.conflict_events, g.war_events, g.sanctions_events, g.shipping_events, g.major_power_tension_events
      FROM valuation_features_daily v
      LEFT JOIN news_features_daily n ON n.ticker=v.ticker AND n.feature_date=v.feature_date
      LEFT JOIN news_event_daily_global g ON g.event_date=v.feature_date
      WHERE v.ticker=:t
      ORDER BY v.feature_date
    """)
    df = pd.read_sql(q, ENGINE, params={"t": ticker})
    if df.empty or len(df) < 300:
        return 0, 0

    # Growth assumption from 1y change in FCF (where available)
    fcf = pd.to_numeric(df["fcf_ttm"], errors="coerce")
    df["growth_assumption"] = (fcf / fcf.shift(252) - 1.0).clip(-0.15, 0.20)
    df["growth_assumption"] = df["growth_assumption"].fillna(0.05)

    # Build DCF daily rows
    dcf_rows = []
    for _, r in df.iterrows():
        d = calc_dcf_row(r)
        if not d:
            continue
        dcf_rows.append({"ticker": ticker, "feature_date": r["feature_date"], **d})

    if dcf_rows:
        with ENGINE.begin() as c:
            c.execute(UP_DCF, dcf_rows)

    # Merge DCF back for model features
    dcf_df = pd.DataFrame(dcf_rows)
    if dcf_df.empty:
        return len(dcf_rows), 0

    df = df.merge(
        dcf_df[["feature_date", "intrinsic_value_per_share", "valuation_gap_pct"]],
        on="feature_date",
        how="left",
    )
    df["ret_1d"] = pd.to_numeric(df["close_price"], errors="coerce").pct_change()
    df["target_ret_next"] = df["ret_1d"].shift(-1)
    df["target_abs_next"] = df["target_ret_next"].abs()

    feature_cols = [
        "ret_1d",
        "valuation_gap_pct",
        "pe_ratio",
        "pb_ratio",
        "fcf_yield",
        "earnings_yield",
        "article_count",
        "avg_sentiment",
        "war_mentions",
        "sanction_mentions",
        "supply_shock_mentions",
        "conflict_events",
        "war_events",
        "sanctions_events",
        "shipping_events",
        "major_power_tension_events",
    ]
    for c in feature_cols:
        if c not in df:
            df[c] = np.nan
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pred_rows = []
    mask = df["target_ret_next"].notna()
    if mask.sum() < 250:
        return len(dcf_rows), 0

    train_idx = np.where(mask)[0]
    cutoff = int(len(train_idx) * 0.7)
    tr = train_idx[:cutoff]
    if len(tr) < 200:
        return len(dcf_rows), 0

    Xnp = X.to_numpy(dtype=np.float32)
    y_ret = df["target_ret_next"].to_numpy(dtype=np.float32)
    y_abs = df["target_abs_next"].to_numpy(dtype=np.float32)

    m_ret = train_tf_regressor(Xnp[tr], y_ret[tr], Xnp.shape[1])
    m_abs = train_tf_regressor(Xnp[tr], y_abs[tr], Xnp.shape[1])

    pr_ret = m_ret.predict(Xnp, verbose=0).reshape(-1)
    pr_abs = m_abs.predict(Xnp, verbose=0).reshape(-1)

    for i, r in df.iterrows():
        pred_rows.append(
            {
                "ticker": ticker,
                "feature_date": r["feature_date"],
                "pred_next_day_return": float(pr_ret[i])
                if np.isfinite(pr_ret[i])
                else None,
                "pred_next_day_abs_return": float(pr_abs[i])
                if np.isfinite(pr_abs[i])
                else None,
                "model_version": "val_news_tf_v2",
                "trained_rows": int(len(tr)),
            }
        )

    with ENGINE.begin() as c:
        c.execute(UP_PRED, pred_rows)

    return len(dcf_rows), len(pred_rows)


def main(limit=None):
    with ENGINE.begin() as c:
        c.execute(text(DDL1))
        c.execute(text(DDL2))

    tickers = pd.read_sql(
        "select distinct ticker from valuation_features_daily order by ticker", ENGINE
    )["ticker"].tolist()
    if limit:
        tickers = tickers[:limit]

    dcf_total = pred_total = 0
    for t in tickers:
        try:
            d, p = process_ticker(t)
            dcf_total += d
            pred_total += p
            print(f"{t}: dcf={d} pred={p}")
        except Exception as e:
            print(f"{t}: error={e}")

    print(f"TOTAL_DCF={dcf_total}")
    print(f"TOTAL_PRED={pred_total}")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(lim)
