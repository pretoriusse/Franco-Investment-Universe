import pandas as pd
import numpy as np
import logging
import time
import random
from tqdm import tqdm
from assets.database_queries import (
    fetch_stock_universe_from_db,
    get_ticker_from_db,
    insert_stock_data_history_batch,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Outlier detection threshold: value <= 0 or < 10% of neighbors
OUTLIER_FACTOR = 0.1


class OutlierFixer:
    def __init__(self):
        # load all stock tickers (excluding commodities)
        self.universe = fetch_stock_universe_from_db()

    def fetch_history(self, ticker: str) -> pd.DataFrame:
        """
        Fetch full history for a ticker from the DB via get_ticker_from_db.
        Returns DataFrame with 'date','open','high','low','close','volume'.
        """
        df = get_ticker_from_db(ticker)
        if df.empty:
            logging.warning(f"No history returned for {ticker}")
            return df
        # Remove duplicate/uppercase Close and keep single lowercase close
        if "Close" in df.columns:
            # drop the uppercase Close column that duplicates lowercase close
            df = df.drop(columns=["Close"])
        # Standardize column names if needed
        if "close" not in df.columns and "Close" in df.columns:
            df = df.rename(columns={"Close": "close"})
        # Select only the relevant columns, ignore others
        df = df.loc[:, ["date", "open", "high", "low", "close", "volume"]]
        # Ensure date type and sort
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
        # standardize columns
        # standardize and dedupe columns
        df = df.rename(columns={"Close": "close"})
        # Remove duplicate column names and unwanted columns
        df = df.loc[:, ["date", "open", "high", "low", "close", "volume"]]

        df = df[["date", "open", "high", "low", "close", "volume"]]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def detect_and_fix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify outliers (close <= 0 or < 10% of neighbor close).
        Replace each with linear interpolation of prev/next close.
        """
        if df.empty:
            return df

        # pad missing calendar days
        df = df.set_index("date").asfreq("D", method="pad").reset_index()

        # replace zeros with NaN and forward-fill
        df[["open", "high", "low", "close"]] = (
            df[["open", "high", "low", "close"]].replace(0, np.nan).ffill()
        )

        # compute neighbor series
        prev_close = df["close"].shift(1)
        next_close = df["close"].shift(-1)

        # mask outliers
        mask = (
            (df["close"] <= 0)
            | (df["close"] < prev_close * OUTLIER_FACTOR)
            | (df["close"] < next_close * OUTLIER_FACTOR)
        )

        # fix each outlier by linear interpolation
        for idx in df.index[mask]:
            if idx == 0 or idx == len(df) - 1:
                continue
            y0 = prev_close.iloc[idx]
            y1 = next_close.iloc[idx]
            # if neighbors valid
            if pd.isna(y0) or pd.isna(y1):
                continue
            frac = 0.5
            new_val = y0 + frac * (y1 - y0)
            logging.info(
                f"Fixing outlier {df.at[idx, 'date'].date()}: {df.at[idx, 'close']} → {new_val:.2f}"
            )
            for col in ["open", "high", "low", "close"]:
                df.at[idx, col] = new_val

        return df

    def apply_updates(self, ticker: str, df: pd.DataFrame):
        """
        Upsert corrected rows back into the DB.
        """
        if df.empty:
            return
        batch = []
        total = len(df)
        for i, row in df.iterrows():
            batch.append(
                {
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                    "adj_close": row["close"],
                }
            )
            if len(batch) >= 500:
                insert_stock_data_history_batch(batch, on_conflict_update=True)
                batch.clear()
        if batch:
            insert_stock_data_history_batch(batch, on_conflict_update=True)

    def run(self):
        """
        Process each ticker: fetch history, fix outliers, update DB.
        """
        for _, row in tqdm(
            self.universe.iterrows(), total=len(self.universe), desc="Tickers"
        ):
            ticker = row["code"]
            if row.get("commodity"):
                continue
            try:
                logging.info(f"Processing {ticker}")
                df = self.fetch_history(ticker)
                df_fixed = self.detect_and_fix(df)
                self.apply_updates(ticker, df_fixed)
            except Exception as e:
                logging.error(f"Error processing {ticker}: {e}")
            time.sleep(random.uniform(0.5, 1.5))


if __name__ == "__main__":
    OutlierFixer().run()
