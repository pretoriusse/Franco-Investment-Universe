import yfinance as yf
import pandas as pd
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from datetime import datetime
from assets.database_queries import (
    fetch_latest_commodity_date,
    insert_commodities_batch,
    close_session,
    fetch_commodity_universe_from_db
)

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

def fetch_data(ticker, start_date):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, interval="1d")
    
    if hist.empty:
        return None

    hist['Date'] = hist.index
    all_dates = pd.date_range(start=hist.index.min(), end=hist.index.max(), freq='D')
    hist = hist.reindex(all_dates, method='pad').reset_index(drop=True)

    data_list = []
    for index, row in hist.iterrows():
        adj_close = row['Adj Close'] if 'Adj Close' in row else row['Close']
        data_list.append({
            'date': row['Date'].strftime('%Y-%m-%d'),
            'ticker': ticker,
            'Open': round(row['Open'], 2),
            'High': round(row['High'], 2),
            'Low': round(row['Low'], 2),
            'Close': round(row['Close'], 2),
            'Adj Close': round(adj_close, 2)
        })

    return data_list

def process_commodity_upload(ticker, progress):
    try:
        latest_date = fetch_latest_commodity_date(ticker)
        current_date = datetime.now().date()
        if latest_date:
            if latest_date == current_date:
                print(f"No new data needed for ticker {ticker} as the latest date is today.")
                return
            start_date = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            start_date = '2004-01-01'

        data_list = fetch_data(ticker, start_date)
        if data_list:
            insert_commodities_batch(data_list)
    except Exception as e:
        print(f"An error occurred while processing ticker {ticker}: {e}")
    finally:
        close_session()

    # Update the progress bar
    progress.update_ticker_progress(ticker, 100)
    # Add a delay of 5 seconds
    time.sleep(5)

def main():
    try:
        commodity_universe = fetch_commodity_universe_from_db()
        tickers = commodity_universe['code'].tolist()

        if not tickers:
            print("No commodities found in the database.")
            return

        progress = UploadProgress(len(tickers))

        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            for ticker in tickers:
                futures.append(executor.submit(process_commodity_upload, ticker, progress))
                time.sleep(5)  # Ensure there is a delay between starting threads

            for future in futures:
                future.result()
        progress.close()

    except Exception:
        print('No Commodities in database.')

if __name__ == "__main__":
    main()
