import yfinance as yf
import pandas as pd
import logging
from datetime import timedelta, datetime
from decimal import Decimal, InvalidOperation
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ZAR Functions
def download_zar():
    logging.info("Starting download of ZAR data")

    latest_date = fetch_latest_date_for_zar("ZAR=X")
    logging.debug(f"Latest date from the database: {latest_date}")

    check_date = latest_date + timedelta(days=1) if latest_date else None

    if latest_date:
        start_date = check_date.strftime('%Y-%m-%d')
        logging.debug(f"Data will be downloaded starting from {start_date}")
        if check_date > datetime.now().date():
            logging.info("Data is up to date; no new data to download.")
            return
    else:
        start_date = '2002-01-01'
        logging.debug(f"No previous data found. Data will be downloaded starting from {start_date}")

    # Download data from Yahoo Finance starting from the latest date
    zar_data = yf.download("ZAR=X", start=start_date, interval="1d")
    zar_data.reset_index(inplace=True)
    
    if zar_data.empty:
        logging.info("No new ZAR data to download.")
        return

    # Ensure the Date column is in datetime format and set as index
    zar_data['Date'] = pd.to_datetime(zar_data['Date'])
    zar_data.set_index('Date', inplace=True)
    
    # Resample the data to fill in missing dates
    zar_data = zar_data.asfreq('D', method='pad')
    logging.debug(f"Data after resampling: {zar_data.head()}")

    logging.info("Calculating moving averages and overbought/oversold values")
    zar_data = calculate_moving_averages(zar_data)
    zar_data = zar_data.dropna(subset=['Overbought_Oversold']).iloc[55:]  # Remove first 55 values and rows with NaN values
    logging.debug(f"Data after calculating moving averages: {zar_data.head()}")

    # Prepare data for insertion into zar_usd
    batch = []
    for date, row in zar_data.iterrows():
        record = {
            'date': date.strftime('%Y-%m-%d'),
            'high': row['High'],
            'low': row['Low'],
            'close': row['Close'],
            'adj_close': row['Adj Close'],
            'volume': row['Volume'],
            'open': row['Open'],
            'overbought_oversold': row['Overbought_Oversold']
        }
        batch.append(record)

    logging.debug(f"Prepared batch for insertion into zar_usd: {batch[:5]}")  # Log only the first 5 records for brevity

    # Insert data into the zar_usd table
    insert_zar_usd_batch(batch)
    logging.info("ZAR data uploaded successfully to zar_usd!")

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
            else:
                if current_type == 'bad':
                    current_period[1] = date
                elif current_type == 'good':
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
