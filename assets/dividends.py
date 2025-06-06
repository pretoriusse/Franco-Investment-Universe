import yfinance as yf
import pandas as pd
import threading
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from assets.database_queries import fetch_latest_dividend_date, insert_dividends_batch, fetch_stock_universe_from_db

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

def make_dates_date_only(date_obj):
    """Convert date_obj to a date without time or timezone information."""
    if isinstance(date_obj, pd.Timestamp):
        return date_obj.date()
    return date_obj

def upload_dividends(ticker, progress: UploadProgress, max_retries=5, delay=2):
    attempt = 0
    while attempt < max_retries:
        try:
            # Fetch the latest date for dividends and convert it to a date object without timezone information
            latest_date = fetch_latest_dividend_date(ticker)
            if latest_date:
                latest_date = make_dates_date_only(latest_date)

            stock = yf.Ticker(ticker)
            dividends = stock.dividends

            if latest_date:
                # Remove any timezone information from the dividend dates and filter
                dividends.index = dividends.index.map(make_dates_date_only)
                dividends = dividends[dividends.index > latest_date]

            if dividends.empty:
                print(f"No new dividends data found for ticker {ticker}")
                progress.update_ticker_progress(ticker, 100)
                return

            batch = [
                {'date': date.strftime('%Y-%m-%d'), 'ticker': ticker, 'dividend': float(dividend)}
                for date, dividend in dividends.items()
            ]

            insert_dividends_batch(batch)
            progress.update_ticker_progress(ticker, 100)
            break  # Exit the loop if successful

        except Exception as e:
            print(f"An error occurred while processing dividend for ticker {ticker}: {e}. Retrying ({attempt + 1}/{max_retries})...")
            attempt += 1
            time.sleep(delay * attempt)  # Exponential backoff

    if attempt == max_retries:
        print(f"Failed to process dividend for ticker {ticker} after {max_retries} attempts.")

def main():
    stock_universe = fetch_stock_universe_from_db()
    progress = UploadProgress(stock_universe.shape[0])

    with ThreadPoolExecutor(max_workers=5) as executor:  # Reduce max_workers to limit the number of simultaneous requests
        futures = []
        for index, row in stock_universe.iterrows():
            futures.append(executor.submit(upload_dividends, row['code'], progress))
            time.sleep(2)  # Add delay between each ticker start (2 seconds in this example)

        for future in futures:
            future.result()

    print("All tickers' dividends have been processed.")

if __name__ == "__main__":
    main()
