import os
import yfinance as yf
import pandas as pd
import pdfkit
from jinja2 import Environment, FileSystemLoader
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import schedule
import time
from datetime import datetime, timedelta
from email.utils import formataddr
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from keras.api import Sequential as SequentialType
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.backend import clear_session # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import base64
import logging
from colorama import init as colorama_init
from colorama import Fore
import threading
import gc
from PIL import Image
import re
import hashlib
import json
from assets import upload_history, zar_process, dividends, fetch_daily_commodity_data
from assets.const import EMAIL_ADDRESS, SERVER_ADDRESS, SERVER_PORT, EMAIL_PASSWORD
from assets import database_queries as db_queries  # Importing database queries
from PyPDF2 import PdfReader, PdfWriter
import boto3
from botocore.exceptions import NoCredentialsError

# Colorama init
colorama_init()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
mlp.rcParams['figure.max_open_warning'] = 200  # Increase the limit to 100 or any suitable number
csv_file = 'investment_universe.csv'  # CSV file with your stock data
graph_dir = 'plots'

# DigitalOcean Spaces credentials
SPACES_KEY = 'DO00W9U289PF7UNPEPGV'
SPACES_SECRET = 'aK9NjzisQNh80HGdUbNSb7FkXkV2eg/Lydr68FBRnTA'
SPACES_REGION = 'nyc3'
SPACES_BUCKET = 'pretoriusresearch'
SPACES_URL = f'https://{SPACES_BUCKET}.{SPACES_REGION}.digitaloceanspaces.com'

# Path to wkhtmltopdf executable
path_wkhtmltopdf = r'/usr/bin/wkhtmltopdf'
pdfkit_config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

# Load HTML template
env = Environment(loader=FileSystemLoader('.'))

# Metrics
METRIC_ACCURACY = 'accuracy'

# ENABLE DEBUGGING and or Predictions
DEBUGGING = False
PREDICTION = True
SUMMARY_REPORT = True

DIRECTORIES = ['data', 'logs', 'plots', 'reports', 'models', 'runs', 'data/runs']

# Configure Tensorflow
# Set GPU options for memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logger.error(f"Error setting GPU memory growth: {e}")

if DEBUGGING:
    # Define hyperparameters
    HP_LSTM_UNITS = hp.HParam('lstm_units', hp.Discrete([400]))
    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.2, 0.3, 0.4]))
    HP_EPOCHS = hp.HParam('epochs', hp.Discrete([2]))
else:
    # Define hyperparameters
    HP_LSTM_UNITS = hp.HParam('lstm_units', hp.Discrete([400]))  # Set to 200
    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.2, 0.3, 0.4]))
    HP_EPOCHS = hp.HParam('epochs', hp.Discrete([200]))  # Set to 500

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_LSTM_UNITS, HP_DROPOUT, HP_EPOCHS],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )

def sanitize_ticker(ticker):
    # Replace or remove characters that are not alphanumeric or dot
    sanitized_ticker = re.sub(r'[^A-Za-z0-9.]', '', ticker)
    return sanitized_ticker


def sanitize_ticker_search(ticker):
    # Replace or remove characters that are not alphanumeric or underscore
    sanitized_ticker = re.sub(r'[^A-Za-z0-9_]', '', ticker)
    return sanitized_ticker


# Image Functions
def resize_image(image_path, output_path, max_width=800):
    with Image.open(image_path) as img:
        width_percent = (max_width / float(img.size[0]))
        height = int((float(img.size[1]) * float(width_percent)))
        img = img.resize((max_width, height), Image.Resampling.LANCZOS)
        img.save(output_path)


def compress_image(image_path, output_path, quality=75):
    with Image.open(image_path) as img:
        # Convert to RGB if image has an alpha channel
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(output_path, optimize=True, quality=quality)


def convert_to_jpeg(image_path, output_path):
    with Image.open(image_path) as img:
        rgb_img = img.convert('RGB')  # PNG to JPEG conversion
        rgb_img.save(output_path, format='JPEG', quality=85)


def process_image(img_path):
    resized_path = img_path.replace('.png', '_resized.png')
    compressed_path = img_path.replace('.png', '_compressed.jpg')
    
    resize_image(img_path, resized_path)
    compress_image(resized_path, compressed_path)
    return compressed_path


def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        return None


def calculate_moving_averages(data, short_window=24, long_window=55):
    data['MA24'] = data['Adj Close'].rolling(window=short_window).mean()
    data['MA55'] = data['Adj Close'].rolling(window=long_window).mean()
    data['Overbought_Oversold'] = ((data['MA24'] / data['MA55']) - 1).round(2)
    data['Overbought_Oversold_Value'] = ((data['MA24'] / data['MA55'])).round(2)  # Normalized for baseline 1
    return data


def calculate_bollinger_bands(data, window=20):
    rolling_mean = data['Adj Close'].rolling(window=window).mean()
    rolling_std = data['Adj Close'].rolling(window=window).std()
    data['Bollinger_High'] = rolling_mean + (rolling_std * 2)
    data['Bollinger_Low'] = rolling_mean - (rolling_std * 2)
    return data


def calculate_z_score(data, window=20):
    rolling_mean = data['Adj Close'].rolling(window=window).mean()
    rolling_std = data['Adj Close'].rolling(window=window).std()
    data['Z-Score'] = ((data['Adj Close'] - rolling_mean) / rolling_std).round(2)
    return data


def rsi_calculate(data, window=14):
    delta = data['Adj Close'].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_rsi_adjusted_close_for_all(data, windows=[14]):
    for window in windows:
        data[f'RSI_{window}'] = rsi_calculate(data, window)
    return data


def make_dates_timezone_naive(data):
    # Convert all datetime objects to timezone-naive
    data['date'] = pd.to_datetime(data['date']).dt.tz_localize(None)
    return data


# Plotting
def plot_price_and_bollinger_bands_adjusted_close(data, ticker):
    data['date'] = pd.to_datetime(data['date'])
    end_date = data['date'].max()
    start_date = end_date - pd.DateOffset(years=2)
    data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

    # Ensure the directory exists
    os.makedirs(f'{graph_dir}/{ticker}', exist_ok=True)

    plt.figure(figsize=(16, 7))

    # Plotting the actual closing prices
    plt.plot(data['date'], data['Adj Close'], label=f'{ticker} Price', color='blue')

    # Plotting the moving averages
    plt.plot(data['date'], data['MA24'], label='24-day MA', color='green')
    plt.plot(data['date'], data['MA55'], label='55-day MA', color='red')

    # Adding Bollinger bands
    plt.fill_between(data['date'], data['Bollinger_High'], data['Bollinger_Low'], color='grey', alpha=0.3)

    plt.title(f'Price momentum [{ticker}]')
    plt.xlabel('date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()

    # Formatting date
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()

    # Saving the plot
    ticker = ticker.replace('.JO', '')
    plt.savefig(f'{graph_dir}/{ticker}/adj_bollinger.png')
    plt.close()


def plot_overbought_oversold_adjusted_close(data, ticker, name):
    data = make_dates_timezone_naive(data)
    data['date'] = pd.to_datetime(data['date'])
    end_date = data['date'].max()
    start_date = end_date - pd.DateOffset(years=2)
    data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

    # Ensure the directory exists
    os.makedirs(f'{graph_dir}/{ticker}', exist_ok=True)

    plt.figure(figsize=(18, 6))

    # Plotting overbought/oversold with 0 as baseline
    plt.axhline(0, color='black', linewidth=1)
    plt.plot(data['date'], data['Overbought_Oversold'], label='Overbought/Oversold', color='black', linestyle='--')

    # Highlighting overbought and oversold areas
    plt.fill_between(data['date'], 0, data['Overbought_Oversold'], where=(data['Overbought_Oversold'] > 0),
                     facecolor='green', alpha=0.3)
    plt.fill_between(data['date'], 0, data['Overbought_Oversold'], where=(data['Overbought_Oversold'] < 0),
                     facecolor='red', alpha=0.3)

    plt.title(f'Overbought/Oversold for {name} ({ticker})')
    plt.xlabel('date')
    plt.ylabel('Overbought/Oversold')
    plt.legend()
    plt.grid()

    # Formatting date
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()

    # Saving the plot
    ticker = ticker.replace('.JO', '')
    plt.savefig(f'{graph_dir}/{ticker}/adj_overbought_oversold.png')
    plt.close()


def plot_overbought_oversold_zar(data, ticker):
    data['date'] = pd.to_datetime(data['date'])
    end_date = data['date'].max()
    start_date = end_date - pd.DateOffset(years=2)
    data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

    # Ensure the directory exists
    os.makedirs(f'{graph_dir}/{ticker}', exist_ok=True)

    plt.figure(figsize=(18, 6))

    # Plotting overbought/oversold with 0 as baseline
    plt.axhline(0, color='black', linewidth=1)
    plt.plot(data['date'], data['Overbought_Oversold'], label='Overbought/Oversold', color='black', linestyle='--')

    # Highlighting overbought and oversold areas
    plt.fill_between(data['date'], 0, data['Overbought_Oversold'], where=(data['Overbought_Oversold'] > 0),
                     facecolor='red', alpha=0.3)
    plt.fill_between(data['date'], 0, data['Overbought_Oversold'], where=(data['Overbought_Oversold'] < 0),
                     facecolor='green', alpha=0.3)

    plt.title(f'Overbought/Oversold for {ticker}')
    plt.xlabel('date')
    plt.ylabel('Overbought/Oversold')
    plt.legend()
    plt.grid()

    # Formatting date
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()

    # Saving the plot
    ticker = ticker.replace('.JO', '')
    plt.savefig(f'{graph_dir}/{ticker}/adj_overbought_oversold.png')
    plt.close()


def plot_stock_adjusted_close_last_two_years(unscaled_close, ticker, next_week_predictions, next_month_predictions, name):
    logger.info(f"Starting to plot stock data for the last year for ticker: {ticker}")
    plt.figure(figsize=(16, 8))

    # Ensure the directory exists
    ticker_clean = ticker.replace('.JO', '')
    dir_path = f'{graph_dir}/{ticker_clean}'
    try:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Directory created or already exists: {dir_path}")
    except Exception as e:
        logger.error(f"Error creating directory {dir_path}: {e}")
        return

    # Filter for the last 3 months of data
    try:
        last_date = unscaled_close['date'].max()
        three_months_ago = last_date - pd.DateOffset(months=3)
        last_three_months = unscaled_close[unscaled_close['date'] >= three_months_ago]
        logger.info(f"Filtered data for the last three months successfully")
    except Exception as e:
        logger.error(f"Error filtering data for the last three months: {e}")
        return

    try:
        plt.plot(last_three_months['date'], last_three_months['Adj Close'], label='Historical Data', color='blue')
        logger.info(f"Plotted historical data")
    except Exception as e:
        logger.error(f"Error plotting historical data: {e}")
        return

    # Generate future dates
    try:
        next_week_dates = [last_date + timedelta(days=i) for i in range(1, 7)]
        next_month_dates = [last_date + timedelta(days=i) for i in range(1, 30)]
        next_week_dates.insert(0, last_date)
        next_month_dates.insert(0, last_date)
        logger.info(f"Generated future dates successfully")
    except Exception as e:
        logger.error(f"Error generating future dates: {e}")
        return
    
    if PREDICTION:
        # Adjust prediction lengths to match dates
        if len(next_week_dates) != len(next_week_predictions):
            logger.warning(f"Length mismatch: next_week_dates ({len(next_week_dates)}) and next_week_predictions ({len(next_week_predictions)})")
            if len(next_week_dates) > len(next_week_predictions):
                next_week_dates = next_week_dates[:len(next_week_predictions)]
            else:
                next_week_predictions = next_week_predictions[:len(next_week_dates)]

        if len(next_month_dates) != len(next_month_predictions):
            logger.warning(f"Length mismatch: next_month_dates ({len(next_month_dates)}) and next_month_predictions ({len(next_month_predictions)})")
            if len(next_month_dates) > len(next_month_predictions):
                next_month_dates = next_month_dates[:len(next_month_predictions)]
            else:
                next_month_predictions = next_month_predictions[:len(next_month_dates)]

        try:
            plt.plot(next_week_dates, next_week_predictions, label='Next Week Predictions', color='cyan')
            plt.plot(next_month_dates, next_month_predictions, label='Next Month Predictions', color='magenta')
            logger.info(f"Plotted predictions data")
        except Exception as e:
            logger.error(f"Error plotting predictions data: {e}")
            return

    # Date formatting for x-axis
    try:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.gcf().autofmt_xdate()
        logger.info(f"Formatted x-axis")
    except Exception as e:
        logger.error(f"Error formatting x-axis: {e}")
        return

    try:
        plt.title(f'Adjusted Close Prediction for {name} ({ticker})')
        plt.xlabel('date')
        plt.ylabel('Price (R)')
        plt.legend()
        logger.info(f"Set plot titles and labels")
    except Exception as e:
        logger.error(f"Error setting plot titles and labels: {e}")
        return

    # Save plot to file
    try:
        file_path = os.path.join(dir_path, 'adj_close_prediction.png')
        plt.savefig(file_path)
        plt.close()
        logger.info(f"Plot saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving plot to file: {e}")


def plot_model_vs_actual(model, X_train, y_train, X_test, y_test, debug_plot_path, scaler=None, focus_last_n=200):
    logger.info(f"Starting to plot model vs actual data with enhanced insights.")

    # Generate predictions for training and testing sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # If a scaler is provided, inverse transform the predictions and actual values
    if scaler:
        y_train_pred = scaler.inverse_transform(y_train_pred)
        y_test_pred = scaler.inverse_transform(y_test_pred)
        y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate error metrics
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    # Calculate residuals
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred

    # Create a figure with multiple subplots
    fig, axs = plt.subplots(4, 1, figsize=(16, 24), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
    fig.suptitle('Model Predictions vs Actual Values (Last {} Data Points)'.format(focus_last_n))

    # Plot actual vs predicted on the first subplot
    ax1 = axs[0]
    
    # Plot only the last `focus_last_n` data points
    if len(y_train) > focus_last_n:
        y_train = y_train[-focus_last_n:]
        y_train_pred = y_train_pred[-focus_last_n:]
    if len(y_test) > focus_last_n:
        y_test = y_test[-focus_last_n:]
        y_test_pred = y_test_pred[-focus_last_n:]

    # Plot training data
    ax1.plot(y_train, label='Actual Train Data', color='blue', alpha=0.7, linewidth=2)
    ax1.plot(y_train_pred, label='Predicted Train Data', color='green', alpha=0.7, linewidth=2)
    
    # Plot testing data
    ax1.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, label='Actual Test Data', color='orange', alpha=0.7, linewidth=2)
    ax1.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_pred, label='Predicted Test Data', color='red', alpha=0.7, linewidth=2)

    ax1.set_ylabel('Price (R)')
    ax1.legend()

    # Annotate the plot with error metrics
    metrics_text = f"MSE: {mse:.5f}\nMAE: {mae:.5f}\nR²: {r2:.5f}"
    ax1.text(0.02, 0.95, metrics_text, transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    # Residuals plot on the second subplot
    ax2 = axs[1]
    ax2.plot(train_residuals, label='Train Residuals', color='blue', alpha=0.7)
    ax2.plot(np.arange(len(y_train), len(y_train) + len(y_test)), test_residuals, label='Test Residuals', color='orange', alpha=0.7)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_ylabel('Residuals')
    ax2.legend()

    # Zoomed-in view on the third subplot
    ax3 = axs[2]
    zoom_range = 50  # Adjust this value as needed
    ax3.plot(y_test[-zoom_range:], label='Actual Test Data (Zoomed)', color='orange', alpha=0.7, linewidth=2)
    ax3.plot(y_test_pred[-zoom_range:], label='Predicted Test Data (Zoomed)', color='red', alpha=0.7, linewidth=2)
    ax3.set_ylabel('Price (R)')
    ax3.legend()

    # Residuals distribution on the fourth subplot
    ax4 = axs[3]
    ax4.hist(test_residuals, bins=50, color='orange', alpha=0.7, label='Test Residuals Distribution')
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Frequency')
    ax4.legend()

    # Save plot to file
    try:
        os.makedirs(os.path.dirname(debug_plot_path), exist_ok=True)
        plt.savefig(debug_plot_path)
        plt.close()
        logger.info(f"Enhanced model vs actual plot saved to: {debug_plot_path}")
    except Exception as e:
        logger.error(f"Error saving enhanced model vs actual plot to file: {e}")
        return

    logger.info(f"Completed plotting enhanced model vs actual data.")


def plot_volume_data_last_two_years(unscaled_volume, ticker, next_week_volume_predictions=[], next_month_volume_predictions=[], name=''):
    logger.info(f"Starting to plot volume data for the last year for ticker: {ticker}")
    plt.figure(figsize=(16, 8))

    # Ensure the directory exists
    ticker_clean = ticker.replace('.JO', '')
    dir_path = f'{graph_dir}/{ticker_clean}'
    try:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Directory created or already exists: {dir_path}")
    except Exception as e:
        logger.error(f"Error creating directory {dir_path}: {e}")
        return

    # Filter for the last 3 months of data
    try:
        last_date = unscaled_volume['date'].max()
        three_months_ago = last_date - pd.DateOffset(months=3)
        last_three_months = unscaled_volume[unscaled_volume['date'] >= three_months_ago]
        logger.info(f"Filtered data for the last three months successfully")
    except Exception as e:
        logger.error(f"Error filtering data for the last three months: {e}")
        return

    try:
        plt.plot(last_three_months['date'], last_three_months['volume'], label='Historical volume', color='blue')
        logger.info(f"Plotted historical volume data")
    except Exception as e:
        logger.error(f"Error plotting historical volume data: {e}")
        return


    # Date formatting for x-axis
    try:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.gcf().autofmt_xdate()
        logger.info(f"Formatted x-axis")
    except Exception as e:
        logger.error(f"Error formatting x-axis: {e}")
        return

    try:
        plt.title(f'volume for {name} ({ticker})')
        plt.xlabel('date')
        plt.ylabel('volume')
        plt.legend()
        logger.info(f"Set plot titles and labels")
    except Exception as e:
        logger.error(f"Error setting plot titles and labels: {e}")
        return

    # Save plot to file
    try:
        file_path = os.path.join(dir_path, 'volume.png')
        plt.savefig(file_path)
        plt.close()
        logger.info(f"volume plot saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving volume plot to file: {e}")


# Processing data
def process_ticker_adjusted_close(ticker, commodity, name):
    os.makedirs(os.path.join('data', ticker.replace('.JO', '')), exist_ok=True)
    # Fetching data for the specified period
    starttime_dt = datetime.now() - timedelta(days=1440)
    start_date = starttime_dt.strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    print(Fore.LIGHTGREEN_EX + f"Creating bollinger data for: {ticker}\n" + Fore.RESET)

    if not commodity:
        stock_data = db_queries.get_ticker_from_db_with_date_select(ticker, start_date, end_date)
    else:
        stock_data = db_queries.get_commodities_from_db(ticker)

    ticker = ticker.replace('.JO', '')

    stock_data.to_csv(os.path.join('data', f'{ticker}', 'DataFrame.csv'))

    stock_data = pd.read_csv(os.path.join('data', f'{ticker}', 'DataFrame.csv'))
    try:
        stock_data['date'] = pd.to_datetime(stock_data['date'])
    except KeyError:
        stock_data['date'] = pd.to_datetime(stock_data.index)

    stock_data = calculate_moving_averages(stock_data)
    stock_data.to_csv(os.path.join('data', f'{ticker}', 'MovingAverages.csv'))
    stock_data = calculate_bollinger_bands(stock_data)
    stock_data.to_csv(os.path.join('data', f'{ticker}', 'Bollinger bands.csv'))
    stock_data = calculate_z_score(stock_data)
    stock_data = calculate_rsi_adjusted_close_for_all(stock_data, windows=[14])

    plot_price_and_bollinger_bands_adjusted_close(stock_data, ticker)
    plot_overbought_oversold_adjusted_close(stock_data, ticker, name)

    return stock_data


def process_zar_bollinger():
    ticker = "ZAR"
    os.makedirs(os.path.join('data', ticker), exist_ok=True)
    # Fetching data for the specified period
    starttime_dt = datetime.now() - timedelta(days=1440)
    start_date = starttime_dt.strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    print(Fore.LIGHTGREEN_EX + f"Creating bollinger data for: {ticker}")
    print(Fore.RESET)

    stock_data: pd.DataFrame = yf.download(f"ZAR=X", start=start_date, end=end_date)

    ticker = ticker.replace('.JO', '')

    stock_data.to_csv(os.path.join('data', f'{ticker}', 'DataFrame.csv'))

    stock_data = pd.read_csv(os.path.join('data', f'{ticker}', 'DataFrame.csv'))
    stock_data['date'] = pd.to_datetime(stock_data['Date'])

    stock_data = calculate_moving_averages(stock_data)
    stock_data = calculate_bollinger_bands(stock_data)
    stock_data = calculate_z_score(stock_data)

    plot_price_and_bollinger_bands_adjusted_close(stock_data, ticker)
    plot_overbought_oversold_zar(stock_data, ticker)


# ML Stuff
def calculate_accuracy(y_true, y_pred, tolerance=0.05):
    """Calculate the percentage of predictions within a tolerance of the actual values."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.mean(np.abs((y_true - y_pred) / y_true) <= tolerance)
    return accuracy * 100


def get_data_hash(data):
    """Calculate the hash of the data to check if it has changed."""
    data_str = data.to_json()
    return hashlib.md5(data_str.encode()).hexdigest()


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    
    X = np.array(X)
    y = np.array(y)

    # Reshape X if it's not in the correct 2D shape
    if len(X.shape) == 2:  # X is 2D, but we need 3D for LSTM (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y


def train_new_model(X, y, model_dir, model_path, hparams, sanitized_ticker):
    # Create the directory if it does not exist
    os.makedirs(model_dir, exist_ok=True)
    
    model: SequentialType = Sequential()
    model.add(LSTM(units=hparams['HP_LSTM_UNITS'], return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(hparams['HP_DROPOUT']))
    model.add(LSTM(units=hparams['HP_LSTM_UNITS']))
    model.add(Dropout(hparams['HP_DROPOUT']))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit the model
    history = model.fit(X, y, epochs=hparams['HP_EPOCHS'], batch_size=64, validation_split=0.1)

    # Save the model
    model.save(model_path)
    
    # Save metadata (you can extend this as needed)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred, tolerance=0.05)
    accuracy10 = calculate_accuracy(y, y_pred, tolerance=0.10)
    accuracy15 = calculate_accuracy(y, y_pred, tolerance=0.15)
    accuracy30 = calculate_accuracy(y, y_pred, tolerance=0.30)
    accuracy50 = calculate_accuracy(y, y_pred, tolerance=0.50)

    model_metadata = {
        'last_trained_date': str(pd.to_datetime('today')),
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        'r2_score': r2,
        'predictions_within_5_percent': accuracy,
        'predictions_within_10_percent': accuracy10,
        'predictions_within_15_percent': accuracy15,
        'predictions_within_30_percent': accuracy30,
        'predictions_within_50_percent': accuracy50
    }

    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump(model_metadata, f)

    logger.info(f"Model Performance for {sanitized_ticker}:")
    logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
    logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"Accuracy within ±5% tolerance: {accuracy:.2f}%")
    logger.info(f"Accuracy within ±10% tolerance: {accuracy10:.2f}%")
    logger.info(f"Accuracy within ±15% tolerance: {accuracy15:.2f}%")
    logger.info(f"Accuracy within ±30% tolerance: {accuracy30:.2f}%")
    logger.info(f"Accuracy within ±50% tolerance: {accuracy50:.2f}%")

    return model


def load_model_metadata(model_dir: str):
    metadata_path = os.path.join(model_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def save_predictions_to_db(ticker: str, start_date, next_month_predictions: list):
    # Generate the dates corresponding to the predictions
    prediction_dates = [start_date + timedelta(days=i) for i in range(1, len(next_month_predictions) + 1)]

    # Insert each prediction into the database
    for date, adj_close in zip(prediction_dates, next_month_predictions):
        db_queries.insert_prediction(date=date, adj_close=adj_close, code=ticker)


def predict_adjusted_close_value(hist, hparams, ticker):
    logger.info(f"Starting prediction for ticker: {ticker}")
    
    # Clear any previous session
    clear_session()

    # Scale the data
    scaler = MinMaxScaler()
    hist['Adj Close'] = scaler.fit_transform(hist[['Adj Close']])
    seq_length = 60
    X, y = create_sequences(hist['Adj Close'].values, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sanitize the ticker name
    sanitized_ticker = sanitize_ticker(ticker)

    # Model directory
    model_dir = os.path.join('models', sanitized_ticker)
    model_path = os.path.join(model_dir, f'{sanitized_ticker}_Adjusted_Close_Model.keras')
    
    # Get the last date in the data
    last_date_in_data = hist['date'].max()

    # Check if model exists and load metadata
    model_metadata = load_model_metadata(model_dir)
    
    # Check if model exists and if there is new data to train on
    if os.path.exists(model_path):
        
        model: SequentialType = load_model(model_path)
        
    else:
        logger.info(f"No existing model found for ticker: {ticker} or no metadata. Creating and training a new model.")
        model: SequentialType = train_new_model(X_train, y_train, model_dir, model_path, hparams, sanitized_ticker)

    # Make predictions
    last_sequence = hist['Adj Close'].values[-seq_length:].reshape((1, seq_length, 1))
    next_week_predictions = []
    next_month_predictions = []

    # Predict for the next week
    for _ in range(7):
        next_week_prediction = model.predict(last_sequence)[0][0]
        next_week_predictions.append(next_week_prediction)
        last_sequence = np.append(last_sequence[:, 1:, :], [[[next_week_prediction]]], axis=1)

    # Predict for the next month
    for _ in range(30):
        next_month_prediction = model.predict(last_sequence)[0][0]
        next_month_predictions.append(next_month_prediction)
        last_sequence = np.append(last_sequence[:, 1:, :], [[[next_month_prediction]]], axis=1)

    # Inverse transform predictions back to original scale
    next_week_predictions = scaler.inverse_transform(np.array(next_week_predictions).reshape(-1, 1)).flatten()
    next_month_predictions = scaler.inverse_transform(np.array(next_month_predictions).reshape(-1, 1)).flatten()

    # Save predictions to the database
    save_predictions_to_db(ticker, last_date_in_data, next_month_predictions)

    # Clear session and free up GPU resources
    clear_session()
    del model
    gc.collect()

    return next_week_prediction, next_month_prediction, next_week_predictions.tolist(), next_month_predictions.tolist()


# Function to fetch data and run predictions for each ticker
def fetch_data(hparams: dict):
    logger.info("Starting data fetch process")
    stocks_df = db_queries.fetch_stock_universe_from_db()
    stock_images = []
    total_value_next_week = 0
    total_value_next_month = 0
    starttime_dt = datetime.now() - timedelta(days=4015)
    start_date = starttime_dt.strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    for index, row in stocks_df.iterrows():
        try_count = 0
        ticker = row['code']
        name = row['share_name']
        logger.info(f"Processing ticker: {ticker}, name: {name}")

        while try_count < 5:
            try:
                logger.info(f"Attempting to fetch data for {ticker} (Attempt {try_count + 1})")

                if not row['commodity']:
                    hist = db_queries.get_ticker_from_db_with_date_select(ticker, start_date, end_date)
                    if hist.empty:
                        raise ValueError(f"No data found for {ticker}")
                else:
                    hist = db_queries.get_commodities_from_db(ticker)
                    if hist.empty:
                        raise ValueError(f"No data found for {ticker}")

                # Ensure `hist` is a DataFrame and dates are timezone-naive
                hist = make_dates_timezone_naive(hist)
                hist.reset_index(inplace=True)
                hist['date'] = pd.to_datetime(hist['date'])
                unscaled_close = hist[['date', 'Adj Close']].copy()
                unscaled_volume = hist[['date', 'volume']].copy()
                logger.info(f"Data successfully fetched for {ticker}")
                break

            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}. Retrying in 5 seconds...")
                try_count += 1
                time.sleep(5)

        if try_count == 5:
            logger.error(f"Failed to fetch data for {ticker} after 5 attempts. Skipping ticker.")
            continue

        try:
            logger.info(f"Calculating metrics for {ticker}")
            current_price = round(hist.iloc[-1]['Adj Close'], 2)
            
            # Ensure hist remains a DataFrame after each transformation
            hist = calculate_moving_averages(hist)
            hist = calculate_bollinger_bands(hist)
            hist = calculate_z_score(hist)
            hist = calculate_rsi_adjusted_close_for_all(hist, windows=[14])

            # Convert back to DataFrame if any function accidentally returns a NumPy array
            if isinstance(hist, np.ndarray):
                hist = pd.DataFrame(hist)

            logger.info(f"Metrics calculated for {ticker}")

            stocks_df.at[index, 'Current Price'] = round(current_price, 2)
            current_value = round(current_price * row.get('Initial Amount of Stocks', 1), 2)
            z_score = round(hist.iloc[-1]['Z-Score'], 2)
            overbought_oversold = round(hist.iloc[-1]['Overbought_Oversold'], 2)

            if not PREDICTION:
                next_week_prediction = next_month_prediction = 0
                next_month_predictions = next_week_predictions = []
            else:
                logger.info(f"Generating predictions for {ticker}")
                next_week_prediction, next_month_prediction, next_week_predictions, next_month_predictions = predict_adjusted_close_value(hist, hparams, ticker)
                logger.info(f"Predictions generated for {ticker}")

            # Add the plot_model_vs_actual function here


            stocks_df.at[index, 'Current Value'] = round(current_value, 2)
            stocks_df.at[index, 'Next Week Prediction'] = round(next_week_prediction - 1, 2)
            stocks_df.at[index, 'Next Month Prediction'] = round(next_month_prediction - 1, 2)
            stocks_df.at[index, 'Z-Score'] = round(z_score, 2)
            stocks_df.at[index, 'Overbought_Oversold'] = round(overbought_oversold, 2)
            stocks_df.at[index, 'Overbought_Oversold_Value'] = round(overbought_oversold + 1, 2)
            logger.info(f"Data updated for {ticker} in DataFrame")

            logger.info(f"Generating plots for {ticker}")
            plot_stock_adjusted_close_last_two_years(unscaled_close, ticker.replace('.JO', ''), next_week_predictions, next_month_predictions, name)
            if not row['commodity']:
                plot_volume_data_last_two_years(unscaled_volume, ticker.replace('.JO', ''))
            logger.info(f"Plots generated for {ticker}")

            if row['commodity']:
                stock_images.append({
                    'code': ticker.replace('.JO', ''),
                    'name': name,
                    'adj_prediction': encode_image(process_image(f"{graph_dir}/{ticker.replace('.JO', '')}/adj_close_prediction.png")),
                    'volume_prediction': encode_image(process_image(f"{graph_dir}/{ticker.replace('.JO', '')}/volume.png")),
                    'bollinger': encode_image(process_image(f"{graph_dir}/{ticker.replace('.JO', '')}/adj_bollinger.png")),
                    'overbought_oversold': encode_image(process_image(f"{graph_dir}/{ticker.replace('.JO', '')}/adj_overbought_oversold.png"))
                })
            else:
                stock_images.append({
                    'code': ticker.replace('.JO', ''),
                    'name': name,
                    'adj_prediction': encode_image(process_image(f"{graph_dir}/{ticker.replace('.JO', '')}/adj_close_prediction.png")),
                    'bollinger': encode_image(process_image(f"{graph_dir}/{ticker.replace('.JO', '')}/adj_bollinger.png")),
                    'overbought_oversold': encode_image(process_image(f"{graph_dir}/{ticker.replace('.JO', '')}/adj_overbought_oversold.png"))
                })
            logger.info(f"Images encoded for {ticker}")

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}. Skipping to next ticker.")
            continue

    #stocks_df.drop(columns=['Commodity'], inplace=True)
    logger.info("Data fetch process completed")
    return stocks_df, stock_images, total_value_next_week, total_value_next_month


def generate_bollinger_and_overbought_oversold_adjusted_close():
    os.makedirs(graph_dir, exist_ok=True)

    # Load tickers from CSV
    df: pd.DataFrame = db_queries.fetch_stock_universe_from_db()
    
    threads:list[threading.Thread] = []
    for index, row in df.iterrows():
        #process_ticker_adjusted_close(row['code'], row['Commodity'])
        thread = threading.Thread(target=process_ticker_adjusted_close, args=(row['code'], row['commodity'], row['share_name']), name=f"Bollinger {row['code']}")
        thread.start()
        time.sleep(1)
        threads.append(thread)

    for thread in threads:
        thread.join()


# Function to calculate RSI
def rsi_calculate(data, window=14):
    delta = data['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_rsi_adjusted_close(data, market, sector, ticker):
    rsi_1m_sector = 0
    rsi_3m_sector = 0
    rsi_6m_sector = 0
    rsi_1m_market = 0
    rsi_3m_market = 0
    rsi_6m_market = 0

    # Ensure the date index is set
    data.set_index(pd.to_datetime(data.index), inplace=True)
    market.set_index(pd.to_datetime(market.index), inplace=True)
    sector.set_index(pd.to_datetime(sector.index), inplace=True)

    # Create a unified date index
    unified_index = data.index.union(market.index).union(sector.index)

    # Reindex all data frames to the unified index
    data = data.reindex(unified_index).ffill()
    market = market.reindex(unified_index).ffill()
    sector = sector.reindex(unified_index).ffill()

    # Combine the DataFrames
    combined_df = pd.concat([data, market, sector], axis=1)

    # Save the combined DataFrame to a CSV file
    ticker = ticker.replace('.JO', '')
    os.makedirs(os.path.join('data', ticker), exist_ok=True)
    combined_df.to_csv(f"data/{ticker}/unified_data.csv")

    try:
        stock_now = data['Adj Close'].iloc[-1]

        stock_20_day = data['Adj Close'].iloc[-20]

        stock_60_day = data['Adj Close'].iloc[-60]

        stock_120_day = data['Adj Close'].iloc[-120]

    except Exception as ex:
        print(Fore.RED + f"{ex} for {ticker}")
        print(Fore.RESET)

    try:
        market_now = market['Adj Close'].iloc[-1]

        market_20_day = market['Adj Close'].iloc[-20]

        market_60_day = market['Adj Close'].iloc[-60]

        market_120_day = market['Adj Close'].iloc[-120]

        # Market RSI
        stock_on_stock_20_day = stock_now / stock_20_day
        market_on_market_20_day = market_now / market_20_day
        rsi_20_day = stock_on_stock_20_day / market_on_market_20_day

        stock_on_stock_60_day = stock_now / stock_60_day
        market_on_market_60_day = market_now / market_60_day
        rsi_60_day = stock_on_stock_60_day / market_on_market_60_day

        stock_on_stock_120_day = (stock_now / stock_120_day) + 0.2
        market_on_market_120_day = (market_now / market_120_day)
        rsi_120_day = (stock_on_stock_120_day / market_on_market_120_day)

        rsi_1m_market = rsi_20_day
        rsi_3m_market = rsi_60_day
        rsi_6m_market = rsi_120_day

    except Exception as ex:
        print(Fore.RED + f"{ex} for {ticker}")
        print(Fore.RESET)

    try:
        sector_now = sector['Adj Close'].iloc[-1]

        sector_20_day = sector['Adj Close'].iloc[-20]

        sector_60_day = sector['Adj Close'].iloc[-60]

        sector_120_day = sector['Adj Close'].iloc[-120]

        # Sector RSI
        stock_on_stock_20_day = stock_now / stock_20_day
        sector_on_sector_20_day = sector_now / sector_20_day
        sector_rsi_20_day = stock_on_stock_20_day / sector_on_sector_20_day

        stock_on_stock_60_day = stock_now / stock_60_day
        sector_on_sector_60_day = sector_now / sector_60_day
        sector_rsi_60_day = stock_on_stock_60_day / sector_on_sector_60_day

        stock_on_stock_120_day = stock_now / stock_120_day
        sector_on_sector_120_day = sector_now / sector_120_day
        sector_rsi_120_day = stock_on_stock_120_day / sector_on_sector_120_day

        rsi_1m_sector = sector_rsi_20_day
        rsi_3m_sector = sector_rsi_60_day
        rsi_6m_sector = sector_rsi_120_day

    except Exception as ex:
        print(Fore.RED + f"{ex} for {ticker}")
        print(Fore.RESET)

    return {
        'rsi_1m_sector': rsi_1m_sector,
        'rsi_3m_sector': rsi_3m_sector,
        'rsi_6m_sector': rsi_6m_sector,
        'rsi_1m_market': rsi_1m_market,
        'rsi_3m_market': rsi_3m_market,
        'rsi_6m_market': rsi_6m_market
    }


# Assuming we have historical data for each stock to calculate RSI
def add_adjusted_close_rsi_comparisons(df):
    starttime_dt = datetime.now() - timedelta(weeks=104)
    start_date = starttime_dt.strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    for index, row in df.iterrows():
        ticker = row['code']
        print(Fore.LIGHTMAGENTA_EX + f"Generating RSI for: {ticker}" + Fore.RESET)
        comparison_sector = row['rsi_comparison_sector']
        comparison_market = row['rsi_comparison_market']
        # Load historical data for the ticker
        historical_data = yf.download(f"{ticker}", start=start_date, end=end_date, interval='1d')

        comparison_sector_data = yf.download(f"{comparison_sector}", start=start_date, end=end_date, interval='1d')

        comparison_market_data = yf.download(f"{comparison_market}", start=start_date, end=end_date, interval='1d')

        # Calculate RSI for the last 1 month, 3 months, and 6 months
        rsi = calculate_rsi_adjusted_close(historical_data, comparison_market_data, comparison_sector_data, ticker)

        df.at[index, 'SECTOR RSI 1M'] = round(rsi['rsi_1m_sector'], 2)
        df.at[index, 'SECTOR RSI 3M'] = round(rsi['rsi_3m_sector'], 2)
        df.at[index, 'SECTOR RSI 6M'] = round(rsi['rsi_6m_sector'], 2)

        df.at[index, 'MARKET RSI 1M'] = round(rsi['rsi_1m_market'], 2)
        df.at[index, 'MARKET RSI 3M'] = round(rsi['rsi_3m_market'], 2)
        df.at[index, 'MARKET RSI 6M'] = round(rsi['rsi_6m_market'], 2)

    return df


# Reporting
def upload_to_spaces(file_path, spaces_access_key, spaces_secret_key, bucket_name, region_name, endpoint_url, today):
    session = boto3.session.Session()
    client = session.client('s3',
                            region_name=region_name,
                            endpoint_url=endpoint_url,
                            aws_access_key_id=spaces_access_key,
                            aws_secret_access_key=spaces_secret_key)

    # Create the directory structure
    file_name = os.path.basename(file_path)
    remote_path = f"reports/{today}/{file_name}"
    
    client.upload_file(file_path, bucket_name, remote_path, ExtraArgs={'ACL': 'public-read'})
    
    return f"{endpoint_url}/{bucket_name}/{remote_path}"


def prepare_stock_images(top_bottom_data):
    stock_images = []
    added_tickers = set()

    for metric in top_bottom_data:
        for group in ['top_10', 'bottom_10']:
            for entry in top_bottom_data[metric][group]:
                ticker = entry['code']
                if ticker not in added_tickers:
                    stock_img = {
                        'name': entry['share_name'],
                        'ticker': ticker,
                        'prediction': encode_image_to_base64(f'plots/{ticker.replace(".JO", "")}/adj_close_prediction_compressed.jpg'),
                        'bollinger': encode_image_to_base64(f'plots/{ticker.replace(".JO", "")}/adj_bollinger_compressed.jpg'),
                        'overbought_oversold': encode_image_to_base64(f'plots/{ticker.replace(".JO", "")}/adj_overbought_oversold_compressed.jpg')
                    }
                    stock_images.append(stock_img)
                    added_tickers.add(ticker)

    return stock_images


def compress_pdf(filename):
    print(f"Compressing PDF report: {filename}")
    reader = PdfReader(filename)
    writer = PdfWriter()

    for page in reader.pages:
        page.compress_content_streams()  # This is where the compression happens
        writer.add_page(page)

    compressed_filename = filename.replace('.pdf', '_compressed.pdf')
    with open(compressed_filename, 'wb') as f:
        writer.write(f)

    print(f"Compressed PDF report created at: {compressed_filename}")

    return compressed_filename


def create_detailed_pdf(data, stock_images, filename, total_value_next_week, total_value_next_month, summary_report=False):
    print(f"Creating PDF report: {filename}")
    options = {
        'page-size': 'Letter',
        'encoding': "UTF-8"
    }

    env = Environment(loader=FileSystemLoader('.'))

    if summary_report:
        print("Preparing summary report...")
        data['Z_Score'] = pd.to_numeric(data['Z-Score'], errors='coerce').fillna(0)
        data['Current Price'] = data['Current Price'].replace(0, pd.NA).fillna(1e-6)
        data['Next_Week_Prediction_Change'] = ((data['Next Week Prediction'] - data['Current Price']) / data['Current Price']) * 100
        data['Next_Month_Prediction_Change'] = ((data['Next Month Prediction'] - data['Current Price']) / data['Current Price']) * 100

        metrics = ['Z_Score', 'Next_Week_Prediction_Change', 'Next_Month_Prediction_Change', 'Overbought_Oversold_Value', 'SECTOR RSI 1M', 'SECTOR RSI 3M', 'SECTOR RSI 6M', 'MARKET RSI 1M', 'MARKET RSI 3M', 'MARKET RSI 6M']
        top_bottom_data = {
            metric: {
                'top_10': data.nlargest(10, metric).to_dict(orient='records'),
                'bottom_10': data.nsmallest(10, metric).to_dict(orient='records')
            }
            for metric in metrics
        }

        # Prepare stock images based on top/bottom data
        stock_images = prepare_stock_images(top_bottom_data)

        template = env.get_template('summary_template.html')
        rendered = template.render(
            top_bottom_data=top_bottom_data,
            summary=create_summary(data, total_value_next_week, total_value_next_month),
            stock_images=stock_images
        )

    else:
        template = env.get_template('detailed_template.html')
        rendered = template.render(
            stocks=data.to_dict(orient='records'),
            summary=create_summary(data, total_value_next_week, total_value_next_month),
            stock_images=stock_images
        )

    # Write the HTML to a file for inspection
    html_file_path = filename.replace('.pdf', '.html')
    with open(html_file_path, 'w') as file:
        file.write(rendered)

    # Convert the HTML report to PDF
    pdfkit.from_file(html_file_path, filename, options=options)

    print(f"PDF report created at: {filename}")


def create_html_summary(data, total_value_next_week, total_value_next_month, template):
    summary = create_summary(data, total_value_next_week, total_value_next_month)
    html_content = template.render(stocks=data.to_dict(orient='records'), summary=summary)
    return html_content


def create_summary(data, total_value_next_week, total_value_next_month):
    try:
        total_invested = data['Initial Purchase Amount'].sum()
    except Exception:
        total_invested = 1

    try:
        current_value = data['Current Value'].sum()
    except Exception:
        current_value = 0

    profit_loss = current_value - total_invested
    summary = (
        f"Total Invested: R{total_invested:,.2f}<br>"
        f"Current Value: R{current_value:,.2f}<br>"
        f"Profit/Loss: R{profit_loss:,.2f} ({(profit_loss / total_invested) * 100:,.2f}%)<br>"
        f"Projected Portfolio Value (Next Week): R{total_value_next_week:,.2f}<br>"
        f"Projected Portfolio Value (Next Month): R{total_value_next_month:,.2f}"
    )
    return summary


def send_email(subject, template_path, top_bottom_data, summary_report_url, detailed_report_url, reciepients=[formataddr(("Raine Pretorius", 'raine.pretorius1@gmail.com')), formataddr(("Franco Pretorius", 'francopret@gmail.com'))]):
    print(Fore.LIGHTGREEN_EX + "Sending email" + Fore.RESET)
    reciepients = [formataddr(("Raine Pretorius", 'raine.pretorius1@gmail.com'))]
    message = MIMEMultipart()
    message['From'] = formataddr(("Stock Bot", EMAIL_ADDRESS))
    message['To'] = ','.join(reciepients)
    message['Subject'] = subject

    # Load the HTML template
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(template_path)
    html_content = template.render(
        top_bottom_data=top_bottom_data,
        summary_report_url=summary_report_url,
        detailed_report_url=detailed_report_url
    )

    message.attach(MIMEText(html_content, 'html'))

    with smtplib.SMTP(SERVER_ADDRESS, SERVER_PORT) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(message)
    print(Fore.GREEN + "Email sent" + Fore.RESET)


def daily_job():
    start_time = datetime.now()
    execute_time = datetime.today().strftime('%Y-%m-%d %H:%M')
    today = datetime.today().strftime('%Y-%m-%d')

    print(Fore.YELLOW + "Starting daily job" + Fore.RESET)

    threads = []

    # Function to start and append a thread
    def start_thread(target, name):
        thread = threading.Thread(target=target, name=name)
        thread.start()
        threads.append(thread)
        print(Fore.GREEN + f"{name} thread started" + Fore.RESET)

    for thread in threads:
        thread.join()

    process_zar_bollinger()

    for direc in DIRECTORIES:
        os.makedirs(direc, exist_ok=True)

    generate_bollinger_and_overbought_oversold_adjusted_close()

    hparams = {
        'HP_LSTM_UNITS': 400,
        'HP_DROPOUT': 0.3,
        'HP_EPOCHS': 20 if DEBUGGING else 200
    }
    
    stock_data, stock_images, total_value_next_week, total_value_next_month = fetch_data(hparams)

    print(Fore.GREEN + "Data fetched and predictions done." + Fore.RESET)

    stock_data = add_adjusted_close_rsi_comparisons(stock_data)

    stock_data.to_csv(os.path.join('runs', f"{execute_time.replace(':', '')}_adjusted_close.csv"), index=False)
    stock_data.to_csv(os.path.join('data', 'runs', f"{execute_time.replace(':', '')}_adjusted_close.csv"), index=False)

    if not DEBUGGING:
        try:
            #upload_dataframe_to_postgresql(stock_data)
            pass
        except Exception as ex:
            print(Fore.RED + 'Data already exists for today' + Fore.RESET)
            pass

    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)

    template_path = 'email_template.html'

    end_time = datetime.now()
    running_time = end_time - start_time
    minutes = round(running_time.seconds / 60, 2)

    print(Fore.MAGENTA + f"\nTime Took:\t{minutes} minutes\n" + Fore.RESET)
    
    os.makedirs(os.path.join(reports_dir, f'{today}'), exist_ok=True)
    attachment_urls = []

    if SUMMARY_REPORT:
        summary_pdf_filename = os.path.join(reports_dir, f'{today}', 'adjusted_close_summary.pdf')
        create_detailed_pdf(stock_data, stock_images, summary_pdf_filename, total_value_next_week, total_value_next_month, summary_report=True)
        compressed_summary_path = compress_pdf(summary_pdf_filename)
        summary_url = upload_to_spaces(compressed_summary_path, SPACES_KEY, SPACES_SECRET, SPACES_BUCKET, SPACES_REGION, SPACES_URL)
        attachment_urls.append(summary_url)
    
    detailed_pdf_filename = os.path.join(reports_dir, f'{today}', 'adjusted_close_detailed.pdf')
    create_detailed_pdf(stock_data, stock_images, detailed_pdf_filename, total_value_next_week, total_value_next_month, summary_report=False)
    compressed_detailed_path = compress_pdf(detailed_pdf_filename)
    detailed_url = upload_to_spaces(compressed_detailed_path, SPACES_KEY, SPACES_SECRET, SPACES_BUCKET, SPACES_REGION, SPACES_URL)
    attachment_urls.append(detailed_url)
    
    print(Fore.GREEN + "PDF created and uploaded" + Fore.RESET)

    try:
        # Prepare the data for the email template
        top_bottom_data = {
            'Z_Score': {
                'top_10': stock_data.nlargest(10, 'Z-Score').to_dict(orient='records'),
                'bottom_10': stock_data.nsmallest(10, 'Z-Score').to_dict(orient='records')
            },
            'Next_Week_Prediction_Change': {
                'top_10': stock_data.nlargest(10, 'Next_Week_Prediction_Change').to_dict(orient='records'),
                'bottom_10': stock_data.nsmallest(10, 'Next_Week_Prediction_Change').to_dict(orient='records')
            },
            'Next_Month_Prediction_Change': {
                'top_10': stock_data.nlargest(10, 'Next_Month_Prediction_Change').to_dict(orient='records'),
                'bottom_10': stock_data.nsmallest(10, 'Next_Month_Prediction_Change').to_dict(orient='records')
            },
            'Overbought_Oversold_Value': {
                'top_10': stock_data.nlargest(10, 'Overbought_Oversold_Value').to_dict(orient='records'),
                'bottom_10': stock_data.nsmallest(10, 'Overbought_Oversold_Value').to_dict(orient='records')
            }
            # Add more metrics as needed
        }

        # Send the email with the report links
        send_email(
            subject=f'Daily Stock Report {today}',
            template_path=template_path,
            top_bottom_data=top_bottom_data,
            summary_report_url=summary_url,
            detailed_report_url=detailed_url
        )
    except Exception as ex:
        logger.error("Email not sent:\n%s", ex)
        print(Fore.RED + "Email not sent" + Fore.RESET)
        pass

    print("Job completed" + Fore.RESET)
 

def setup_scheduler():
    schedule.every().day.at("06:30").do(daily_job)
    while True:
        schedule.run_pending()
        time.sleep(15)


if __name__ == '__main__':
    daily_job()
    setup_scheduler()
  