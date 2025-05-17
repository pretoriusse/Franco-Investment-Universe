import os
import pandas as pd
from tqdm import tqdm
import time
import random
import logging
from assets.database_queries import fetch_stock_universe_from_db, fetch_latest_date_for_ticker, insert_stock_data_history_batch
import yfinance as yf
from yfinance.exceptions import YFPricesMissingError, YFRateLimitError

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
    max_retries = 5
    delay = 5  # Initial delay for retry backoff
    for attempt in range(max_retries):
        try:
            df: pd.DataFrame
            # Fetch the latest date for the ticker from the database
            latest_date = fetch_latest_date_for_ticker(ticker)
            logging.info(f"latest_date for {ticker}: {latest_date}")
            
            #latest_date = None

            if latest_date:
                # Convert to pd.Timestamp for consistent comparison
                check_date = pd.Timestamp(latest_date) + pd.Timedelta(days=1)
                now = pd.Timestamp.now()

                # Only download if there's new data to fetch
                if check_date.date() > now.date():
                    logging.info(f"No new data needed for ticker {ticker} as the latest date is today.")
                    progress.update_ticker_progress(ticker, 100)
                    return

                start_date = check_date.strftime('%Y-%m-%d')
                logging.info(f"Downloading data for {ticker} starting from {start_date}")
                df = yf.download(ticker, start=start_date, interval='1d') # type: ignore

            else:
                # New ticker without data in the database; download from the earliest available date
                logging.info(f"Downloading complete data for new ticker {ticker}")
                df = yf.download(ticker, interval='1d', period='max') # type: ignore

            # Check if DataFrame is empty (e.g., if the ticker is delisted)
            
            if df.empty:
                logging.error(f"No data found for ticker {ticker}. It may be delisted or not available.")
                progress.update_ticker_progress(ticker, 100)
                return
            
            tickerName = ticker.replace('.JO', '')

            df.to_csv(os.path.join('data', f'{tickerName}', 'yfdata.csv'))
            
            lines: list
            with open(os.path.join('data', f'{tickerName}', 'yfdata.csv'), 'r') as f:
                lines = f.readlines()

            lines.pop(1)
            
            try:

                lines.remove('Date,,,,,\n')
            
            except ValueError:
                pass

            os.remove(os.path.join('data', f'{tickerName}', 'yfdata.csv'))
            
            with open(os.path.join('data', f'{tickerName}', 'yfdata.csv'), 'w+') as f:
                f.writelines(lines)


            df = pd.read_csv(os.path.join('data', f'{tickerName}', 'yfdata.csv'))
            df.reset_index(drop=False)
            df.rename(columns={'Price': 'Date'}, inplace=True)

            # Ensure the Date column is in datetime format
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)

            # Fill missing dates with the previous available values (pad method)
            all_dates = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
            df = df.set_index('Date').reindex(all_dates, method='pad').reset_index(drop=False)

            # Batch processing and data insertion
            batch = []
            index = 0
            for i, row in df.iterrows():
                row_data = {
                    'date': row['index'].strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'open': row['Open'],
                    'high': row['High'],
                    'low': row['Low'],
                    'close': row['Close'],
                    'volume': row['Volume'],
                    'comparison_market': f'{comparison_market}',
                    'comparison_sector': f'{comparison_sector}',
                    'adj_close': row['Close']
                }
                batch.append(row_data)

                # Insert batch if size reaches 500 to optimize database performance
                if len(batch) >= 500:
                    insert_stock_data_history_batch(batch, on_conflict_update=True)  # Use upsert
                    batch.clear()
                    progress_percentage = round(((int(index) + 1) / len(df)) * 100, 2)
                    progress.update_ticker_progress(ticker, progress_percentage)
                
                index += 1

            # Insert any remaining data
            if batch:
                insert_stock_data_history_batch(batch, on_conflict_update=True)  # Use upsert
                progress.update_ticker_progress(ticker, 100)

            logging.info(f"Successfully uploaded data for ticker {ticker}")
            break  # Exit the retry loop if successful

        except YFPricesMissingError as e:
            logging.error(f"{ticker} possibly delisted; {e}. Retrying.")
            delay *= 2
            progress.update_ticker_progress(ticker, 100)
        
        except YFRateLimitError as e:
            logging.warning(f"Rate limit hit for {ticker}, sleeping {delay}s (attempt {attempt}/{max_retries})")
            time.sleep(delay)
            delay *= 2
        
        except Exception as e:
            logging.error(f"An error occurred: {e} for ticker {ticker}. Attempt {attempt + 1}/{max_retries}")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
    else:
        logging.error(f"Failed to download data for {ticker} after {max_retries} attempts. Skipping...")

def main():
    # Fetch stock universe from the database
    stock_universe = fetch_stock_universe_from_db()
    if stock_universe.empty:
        logging.error("No tickers found in the database.")
        return

    progress = UploadProgress(stock_universe.shape[0])

    # Process each ticker sequentially with a delay
    for _, row in stock_universe.iterrows():
        if not row['commodity']:  # Skip commodities if needed
            upload_ticker(row['code'], row['rsi_comparison_market'], row['rsi_comparison_sector'], progress)
            time.sleep(1 + random.uniform(1, 5))  # Wait 1 to 5 seconds between downloads

    logging.info("All tickers have been processed.")

if __name__ == "__main__":
    main()
