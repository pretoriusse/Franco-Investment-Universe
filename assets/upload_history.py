import pandas as pd
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import datetime
import logging
from assets.database_queries import fetch_stock_universe_from_db, fetch_latest_date_for_ticker, insert_stock_data_history_batch
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UploadProgress:
    def __init__(self, total):
        self.total = total
        self.progress = tqdm(total=total, desc="Total Progress", unit="ticker", position=0)
        self.current_progresses = {}

    def update_ticker_progress(self, ticker, progress):
        if ticker not in self.current_progresses:
            self.current_progresses[ticker] = tqdm(total=100, desc=f"{ticker} Progress", unit="%", position=len(self.current_progresses) + 1, leave=True)

        ticker_progress = self.current_progresses[ticker]
        ticker_progress.n = progress
        ticker_progress.refresh()

        if progress == 100:
            self.progress.update(1)
            ticker_progress.close()
            del self.current_progresses[ticker]

def make_dates_timezone_naive(date_obj):
    """Convert date_obj to timezone-naive if it is timezone-aware."""
    if isinstance(date_obj, pd.Timestamp) and date_obj.tzinfo is not None:
        return date_obj.tz_localize(None)
    return date_obj

def upload_ticker(ticker, comparison_market, comparison_sector, progress: UploadProgress):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Fetch the latest date for the ticker from the database
            latest_date = fetch_latest_date_for_ticker(ticker)
            logging.info(f"latest_date for {ticker}: {latest_date}")

            if latest_date:
                # Convert to pd.Timestamp for consistent comparison
                check_date = pd.Timestamp(latest_date) + pd.Timedelta(days=1)
                now = pd.Timestamp.now()

                if check_date.date() > now.date():
                    logging.info(f"No new data needed for ticker {ticker} as the latest date is today.")
                    progress.update_ticker_progress(ticker, 100)
                    return

                start_date = check_date.strftime('%Y-%m-%d')
                df: pd.DataFrame = yf.download(ticker, start=start_date, interval='1d')
            else:
                df: pd.DataFrame = yf.download(ticker, interval='1d', period='max')

            if df.empty:
                logging.error(f"No data found for ticker {ticker}. It may be delisted or not available.")
                progress.update_ticker_progress(ticker, 100)
                return

            df['Date'] = df.index
            all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
            df = df.reindex(all_dates, method='pad').reset_index(drop=True)

            batch = []
            for index, row in df.iterrows():
                row_data = {
                    'date': row['Date'].strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'open': row['Open'],
                    'high': row['High'],
                    'low': row['Low'],
                    'close': row['Close'],
                    'volume': row['Volume'],
                    'comparison_market': f'{comparison_market}.JO',
                    'comparison_sector': f'{comparison_sector}.JO',
                    'adj_close': row['Adj Close']
                }
                batch.append(row_data)

                if len(batch) >= 500:  # Increased batch size to reduce the number of inserts
                    insert_stock_data_history_batch(batch)
                    batch.clear()
                    progress_percentage = round(((index + 1) / len(df)) * 100, 2)
                    progress.update_ticker_progress(ticker, progress_percentage)

            if batch:
                insert_stock_data_history_batch(batch)
                progress.update_ticker_progress(ticker, 100)

            logging.info(f"Successfully uploaded data for ticker {ticker}")
            break  # Exit the retry loop if successful

        except Exception as e:
            logging.error(f"An error occurred: {e} for ticker {ticker}. Attempt {attempt + 1}/{max_retries}")
            time.sleep(5)
    else:
        logging.error(f"Failed to download data for {ticker} after {max_retries} attempts")

def main():
    # Fetch stock universe from the database
    stock_universe = fetch_stock_universe_from_db()
    if stock_universe.empty:
        logging.error("No tickers found in the database.")
        return

    progress = UploadProgress(stock_universe.shape[0])

    with ThreadPoolExecutor(max_workers=50) as executor:  # Increase threads for faster uploads
        futures = []
        for _, row in stock_universe.iterrows():
            if row['commodity']:
                continue
            futures.append(executor.submit(upload_ticker, row['code'], row['rsi_comparison_market'], row['rsi_comparison_sector'], progress))
            time.sleep(2)  # Reduce delay to maximize throughput

        for future in futures:
            future.result()

    logging.info("All tickers have been processed.")

if __name__ == "__main__":
    main()
