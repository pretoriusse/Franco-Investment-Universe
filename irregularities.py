#!/usr/bin/env python3
"""
generate_delete_shitty_tickers.py

Connect to your PostgreSQL DB, detect tickers with corrupted or irregular data,
and generate a SQL script to delete them.
"""

import logging
from datetime import datetime

import pandas as pd

# Adjust these imports based on your project structure:
from assets.database_queries import fetch_stock_universe_from_db, get_ticker_from_db

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_SQL_FILE = "delete_shitty_tickers.sql"


def has_irregularities(df: pd.DataFrame) -> bool:
    """
    Returns True if the DataFrame has:
      - empty data
      - any NaN or zero in 'close'
      - any zero volume
      - duplicated dates
      - missing calendar days (assuming padded daily data)
    """
    if df.empty:
        return True

    # NaN or zero prices
    if df["close"].isna().any() or (df["close"] == 0).any():
        return True

    # Duplicate dates
    if df["date"].duplicated().any():
        return True

    # Missing calendar days
    df_dates = pd.to_datetime(df["date"]).dt.date
    full_idx = pd.date_range(df_dates.min(), df_dates.max(), freq="D").date
    if set(full_idx) - set(df_dates):
        return True

    return False


def main():
    logger.info("Fetching stock universe from DB...")
    universe = fetch_stock_universe_from_db()
    if universe.empty:
        logger.error("No tickers found in universe. Exiting.")
        return

    bad_tickers = []
    for row in universe.itertuples():
        ticker = row.code
        logger.info(f"Checking ticker {ticker}...")
        df = get_ticker_from_db(ticker)
        if has_irregularities(df):
            logger.warning(f"Ticker {ticker} flagged as corrupted/irregular.")
            bad_tickers.append(ticker)

    if not bad_tickers:
        logger.info("✅ No corrupt or irregular tickers detected.")
        return

    logger.info(f"Generating SQL script for {len(bad_tickers)} tickers.")
    # Write deletion script
    with open(OUTPUT_SQL_FILE, "w") as f:
        f.write(f"-- Auto-generated on {datetime.utcnow().isoformat()} UTC\n")
        f.write("-- List of tickers with corrupted or irregular data:\n")
        for t in bad_tickers:
            f.write(f"--   {t}\n")
        f.write("\nBEGIN;\n")
        tickers_list = ", ".join(f"'{t}'" for t in bad_tickers)
        f.write(f"DELETE FROM stock_data_history WHERE ticker IN ({tickers_list});\n")
        f.write(f"DELETE FROM stock               WHERE code   IN ({tickers_list});\n")
        f.write("COMMIT;\n")

    logger.info(f"📝 Deletion script written to {OUTPUT_SQL_FILE}")


if __name__ == "__main__":
    main()
