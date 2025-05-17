import yfinance as yf
import pandas as pd
import logging
from datetime import timedelta, datetime
from decimal import InvalidOperation
import os
from assets.database_queries import (
    fetch_latest_date_for_zar,
    insert_zar_usd_batch,
    update_zar_periods,
    insert_zar_good_period,
    insert_zar_bad_period,
    close_session,
    fetch_all_zar_usd
)

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def download_zar():
    logging.info("Starting download of ZAR data")

    # Fetch the latest date from the database to know what data to insert later
    latest_date = fetch_latest_date_for_zar("ZAR=X")
    logging.debug(f"Latest date from the database: {latest_date}")

    # Download data from 6 months ago to ensure enough coverage for moving averages
    start_date = (datetime.now() - timedelta(days=6 * 30)).strftime('%Y-%m-%d')
    logging.debug(f"Downloading data from {start_date}")

    # Download data from Yahoo Finance starting from 6 months ago
    zar_data = yf.download("ZAR=X", start=start_date, interval="1d")
    zar_data.reset_index(inplace=True)

    if zar_data.empty:
        logging.info("No ZAR data to download.")
        return
    
    tickerName = "ZAR"

    zar_data.to_csv(os.path.join('data', f'{tickerName}', 'yfdata.csv'))
    
    lines: list
    with open(os.path.join('data', f'{tickerName}', 'yfdata.csv'), 'r') as f:
        lines = f.readlines()

    lines.pop(1)

    os.remove(os.path.join('data', f'{tickerName}', 'yfdata.csv'))
    
    with open(os.path.join('data', f'{tickerName}', 'yfdata.csv'), 'w+') as f:
        f.writelines(lines)


    zar_data = pd.read_csv(os.path.join('data', f'{tickerName}', 'yfdata.csv'))
    zar_data.reset_index(drop=True, inplace=True)

    # Ensure the Date column is in datetime format
    zar_data.dropna(subset=['Date'], inplace=True)
    logging.debug(f"Data after resampling: {zar_data.head()}")

    # Calculate moving averages and overbought/oversold values on the entire dataset
    logging.info("Calculating moving averages and overbought/oversold values")
    zar_data = calculate_moving_averages(zar_data)

    # Drop rows with NaN values for moving averages
    zar_data = zar_data.dropna(subset=['Overbought_Oversold'])
    logging.debug(f"Data after calculating moving averages: {zar_data.head()}")

    # Filter data to include only new records based on the latest date from the database
    if latest_date:
        latest_date = pd.Timestamp(latest_date)  # Convert latest_date to pandas Timestamp
        zar_data = zar_data[pd.to_datetime(zar_data['Date']) > latest_date]
        logging.debug(f"Data filtered from the last database date {latest_date}: {zar_data.head()}")

    # Check if there's data to insert after filtering
    if zar_data.empty:
        logging.error("No new records to insert into ZAR/USD after filtering.")
        return

    # Prepare data for insertion into zar_usd
    batch = []
    for date, row in zar_data.iterrows():
        record = {
            'date': row.get('Date'),
            'high': row.get('High'),
            'low': row.get('Low'),
            'close': row.get('Close'),
            'adj_close': row.get('Close'),
            'volume': row.get('Volume', 0),  # Default volume to 0 if not available
            'open': row.get('Open'),
            'overbought_oversold': row.get('Overbought_Oversold')
        }

        # Check if all required fields are not None
        if any(v is None for v in record.values()):
            logging.error(f"Record with missing values found: {record}")
            continue

        batch.append(record)

    # Validate batch before insertion
    if not batch:
        logging.error("No valid records to insert into ZAR/USD after cleaning. Batch might contain invalid dates or missing data.")
        return

    logging.debug(f"Prepared batch for insertion into zar_usd: {batch[:5]}")  # Log only the first 5 records for brevity

    # Insert data into the zar_usd table
    try:
        insert_zar_usd_batch(batch)
        logging.info("ZAR data uploaded successfully to zar_usd!")
    except Exception as e:
        logging.error(f"Failed to insert batch into ZAR/USD: {e}")

def calculate_moving_averages(data, short_window=24, long_window=55):
    data['MA24'] = data['Close'].rolling(window=short_window).mean()
    data['MA55'] = data['Close'].rolling(window=long_window).mean()
    data['Overbought_Oversold'] = ((data['MA24'] / data['MA55']) - 1).round(2)
    return data

def update_zar_periods():
    logging.info("Starting to update ZAR periods")

    zar_data = fetch_all_zar_usd()
    current_period = None
    current_type = None

    for date, overbought_oversold in zar_data:
        if overbought_oversold is None:
            continue  # Skip rows with NaN values

        try:
            if overbought_oversold > 0:
                if current_type != 'bad':
                    if current_period:
                        insert_zar_bad_period(current_period)
                    current_period = [date, date, 'bad']
                    current_type = 'bad'
                else:
                    current_period[1] = date
            elif overbought_oversold < 0:
                if current_type != 'good':
                    if current_period:
                        insert_zar_good_period(current_period)
                    current_period = [date, date, 'good']
                    current_type = 'good'
                else:
                    current_period[1] = date
        except InvalidOperation:
            logging.error(f"Invalid decimal operation for date {date} with value {overbought_oversold}")

    if current_period:
        if current_type == 'bad':
            insert_zar_bad_period(current_period)
        else:
            insert_zar_good_period(current_period)

    logging.info("ZAR periods updated successfully!")

def process_zar():
    download_zar()
    update_zar_periods()

if __name__ == '__main__':
    process_zar()
    close_session()
