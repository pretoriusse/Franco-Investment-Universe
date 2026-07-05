"""Daily close-price report pipeline.

Entry point: ``daily_job()`` (called by ``main.py`` at 06:00, or directly via
``python close_report.py``). For every ticker in the stock universe the job:

1. Loads OHLCV history from the ``sharesdata`` database.
2. Computes technical indicators (moving averages, Bollinger bands, Z-score, RSI).
3. Runs the per-ticker LSTM model to predict the close price 7 and 30 days out
   (training a model on the fly if none exists on disk).
4. Renders price/volume/indicator charts to ``plots/`` and base64-encodes them.
5. Uploads the run results to the ``close_runs`` table and renders the summary
   and detailed HTML/PDF reports for subscribers.

The adjusted-close twin of this script is ``adjusted_close_report.py``.
"""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

import yfinance as yf
import pandas as pd
import pdfkit
from jinja2 import Environment, FileSystemLoader
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import schedule
import time
from datetime import datetime, timedelta, timezone
from email.utils import formataddr
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from keras.api import Sequential as SequentialType
from keras.models import Sequential  # type: ignore
from keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.backend import clear_session  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import base64
import logging
from colorama import init as colorama_init
from colorama import Fore
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from PIL import Image
import re
import hashlib
import json
from assets.const import EMAIL_ADDRESS, SERVER_ADDRESS, SERVER_PORT, EMAIL_PASSWORD
from assets import database_queries as db_queries  # Importing database queries
from PyPDF2 import PdfReader, PdfWriter
from sqlalchemy.exc import SQLAlchemyError
import boto3
import subprocess
import matplotlib
import platform
import shutil
from typing import Any

from jinja2 import Template
from assets.report_models import (
    FetchResult,
    ModelHyperParams,
    PredictionResult,
    RankedStockImage,
    RSIComparison,
    SentimentAdjustment,
    StockChartImages,
    StockRecord,
    TopBottomData,
)
from webapp.models import Subscribers

# Colorama init
colorama_init()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
mlp.rcParams["figure.max_open_warning"] = (
    200  # Increase the limit to 100 or any suitable number
)
csv_file = "investment_universe.csv"  # CSV file with your stock data
graph_dir = "plots"
matplotlib.use("Agg")

# DigitalOcean Spaces credentials
SPACES_KEY = "DO00W9U289PF7UNPEPGV"
SPACES_SECRET = "aK9NjzisQNh80HGdUbNSb7FkXkV2eg/Lydr68FBRnTA"
SPACES_REGION = "nyc3"
SPACES_BUCKET = "pretoriusresearch"
SPACES_URL = f"https://{SPACES_BUCKET}.{SPACES_REGION}.digitaloceanspaces.com"

# Path to wkhtmltopdf executable. Prefer whatever is on PATH, then fall back to
# the conventional location for the host OS (the pipeline runs on Linux in
# production but is developed on Windows). Override with the WKHTMLTOPDF_PATH env var.
path_wkhtmltopdf = (
    os.environ.get("WKHTMLTOPDF_PATH")
    or shutil.which("wkhtmltopdf")
    or (
        r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
        if platform.system() == "Windows"
        else "/usr/bin/wkhtmltopdf"
    )
)
pdfkit_config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

# Where finished reports are written. Linux box uses the mounted backup share;
# Windows dev box falls back to ~/Shares/Reports. Override with REPORTS_DIR.
REPORTS_DIR = os.environ.get("REPORTS_DIR") or (
    os.path.join(os.path.expanduser("~"), "Shares", "Reports")
    if platform.system() == "Windows"
    else "/mnt/backups/Shares/Reports"
)

# Load HTML template
env = Environment(loader=FileSystemLoader("."))

# Metrics
METRIC_ACCURACY = "accuracy"

# ENABLE DEBUGGING and or Predictions
DEBUGGING = False
PREDICTION = True
SUMMARY_REPORT = True

DIRECTORIES = ["data", "logs", "plots", REPORTS_DIR, "models", "runs", "data/runs"]

# Sentiment adjustment weight: at a compound score of ±1.0 the week prediction
# shifts by ±SENTIMENT_WEEK_WEIGHT and month by ±SENTIMENT_MONTH_WEIGHT of the
# current price.  Keep these small — sentiment is one signal among many.
SENTIMENT_WEEK_WEIGHT = 0.005  # 0.5 % per unit of compound score
SENTIMENT_MONTH_WEIGHT = 0.003  # 0.3 % per unit of compound score

# In-memory model cache: model_path -> loaded Keras model. Retained only so the
# daily_job() teardown can clear it; predict_close_value no longer populates it
# (see the note there). Kept typed for the strict-typing pass.
_model_cache: dict[str, SequentialType] = {}

# matplotlib's pyplot API uses global figure state and is NOT thread-safe.
# The bollinger workers (see generate_bollinger_and_overbought_oversold_close)
# run on a thread pool, so every pyplot section must hold this lock.
_plot_lock = threading.Lock()

# Configure Tensorflow
# Set GPU options for memory growth
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        logger.error(f"Error setting GPU memory growth: {e}")

if DEBUGGING:
    # Define hyperparameters
    HP_LSTM_UNITS = hp.HParam("lstm_units", hp.Discrete([400]))
    HP_DROPOUT = hp.HParam("dropout", hp.Discrete([0.2, 0.3, 0.4]))
    HP_EPOCHS = hp.HParam("epochs", hp.Discrete([2]))
else:
    # Define hyperparameters
    HP_LSTM_UNITS = hp.HParam("lstm_units", hp.Discrete([400]))  # Set to 200
    HP_DROPOUT = hp.HParam("dropout", hp.Discrete([0.2, 0.3, 0.4]))
    HP_EPOCHS = hp.HParam("epochs", hp.Discrete([200]))  # Set to 500

with tf.summary.create_file_writer("logs/hparam_tuning").as_default():
    hp.hparams_config(
        hparams=[HP_LSTM_UNITS, HP_DROPOUT, HP_EPOCHS],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name="Accuracy")],
    )


def generate_email_hash(email: str) -> str:
    """Generate a unique hash for each email based on the email address and current timestamp."""
    hash_input = f"{email}_{datetime.now(timezone.utc).isoformat()}"
    return hashlib.sha256(hash_input.encode()).hexdigest()


def sanitize_ticker(ticker: str) -> str:
    """Strip every character that is not alphanumeric or a dot (keeps '.JO')."""
    sanitized_ticker = re.sub(r"[^A-Za-z0-9.]", "", ticker)
    return sanitized_ticker


def sanitize_ticker_search(ticker: str) -> str:
    """Strip every character that is not alphanumeric or an underscore."""
    sanitized_ticker = re.sub(r"[^A-Za-z0-9_]", "", ticker)
    return sanitized_ticker


# Image Functions
def resize_image(image_path: str, output_path: str, max_width: int = 800) -> None:
    """Resize an image down to *max_width* pixels, preserving aspect ratio."""
    with Image.open(image_path) as img:
        width_percent = max_width / float(img.size[0])
        height = int((float(img.size[1]) * float(width_percent)))
        img = img.resize((max_width, height), Image.Resampling.LANCZOS)
        img.save(output_path)


def compress_image(image_path: str, output_path: str, quality: int = 75) -> None:
    """Save a JPEG-compressed copy of an image (alpha channel is flattened)."""
    with Image.open(image_path) as img:
        # Convert to RGB if image has an alpha channel
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(output_path, optimize=True, quality=quality)


def convert_to_jpeg(image_path: str, output_path: str) -> None:
    """Convert a PNG (or any Pillow-readable image) to an 85%-quality JPEG."""
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")  # PNG to JPEG conversion
        rgb_img.save(output_path, format="JPEG", quality=85)


def process_image(img_path: str) -> str:
    """Return the chart PNG path unchanged.

    Previously this also wrote ``_resized.png`` and ``_compressed.jpg`` copies
    next to every chart, but the reports always embed the original full-size
    PNG (the templates hardcode ``data:image/png``) and nothing reads the
    copies — so for ~200 tickers × 4 charts that was ~1,600 wasted disk writes
    per run. ``resize_image`` / ``compress_image`` remain available for callers
    that genuinely need a smaller variant.
    """
    return img_path


def encode_image(image_path: str) -> str:
    """Return the base64 string of an image file (raises if the file is missing)."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_image_to_base64(image_path: str) -> str | None:
    """Like ``encode_image`` but returns ``None`` instead of raising when missing."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        return None


def generate_pdf_with_password(
    input_html: str,
    output_pdf: str,
    user_password: str,
    owner_password: str = "0306245353082",
) -> None:
    """Render *input_html* to PDF and encrypt it with qpdf (printing disabled)."""
    temp_output_pdf = output_pdf.split(".pdf")[0] + "_temp.pdf"
    options = {
        "page-size": "Letter",
        "encoding": "UTF-8",
        "javascript-delay": 2000,  # Add delay for JS to execute
        "debug-javascript": True,  # Enable JS debugging
    }
    try:
        # Generate the PDF
        pdfkit.from_file(input_html, output_pdf, options=options)
        logger.debug(f"PDF successfully created at: {output_pdf}")

        # Encrypt the PDF with a password and disable print/copy permissions
        qpdf_cmd = [
            "qpdf",
            "--encrypt",
            user_password,
            owner_password,
            "256",
            "--disable-print",
            "--",
            output_pdf,
            temp_output_pdf,
        ]
        subprocess.run(qpdf_cmd, check=True)
        logger.debug(
            f"Password protection and print restriction added to: {temp_output_pdf}"
        )

        # Replace the original PDF with the protected version
        mv_cmd = ["mv", temp_output_pdf, output_pdf]
        subprocess.run(mv_cmd, check=True)
        logger.debug(
            f"Original PDF replaced with password-protected version: {output_pdf}"
        )

    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred: {e}")


def calculate_moving_averages(
    data: pd.DataFrame, short_window: int = 24, long_window: int = 55
) -> pd.DataFrame:
    """Add MA24/MA55 columns plus the MA-ratio overbought/oversold indicator."""
    data["MA24"] = data["close"].rolling(window=short_window).mean()
    data["MA55"] = data["close"].rolling(window=long_window).mean()
    data["Overbought_Oversold"] = ((data["MA24"] / data["MA55"]) - 1).round(2)
    data["Overbought_Oversold_Value"] = (data["MA24"] / data["MA55"]).round(
        2
    )  # Normalized for baseline 1
    return data


def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add Bollinger_High/Bollinger_Low columns (rolling mean ± 2 std)."""
    rolling_mean = data["close"].rolling(window=window).mean()
    rolling_std = data["close"].rolling(window=window).std()
    data["Bollinger_High"] = rolling_mean + (rolling_std * 2)
    data["Bollinger_Low"] = rolling_mean - (rolling_std * 2)
    return data


def calculate_z_score(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add a Z-Score column: deviation of close from its rolling mean in stds."""
    rolling_mean = data["close"].rolling(window=window).mean()
    rolling_std = data["close"].rolling(window=window).std()
    data["Z-Score"] = ((data["close"] - rolling_mean) / rolling_std).round(2)
    return data


def rsi_calculate(data: pd.DataFrame, window: int = 14) -> "pd.Series[float]":
    """Return the classic Relative Strength Index series for the close column."""
    delta = data["close"].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_rsi_close_for_all(
    data: pd.DataFrame, windows: list[int] = [14]
) -> pd.DataFrame:
    """Add an ``RSI_{window}`` column for each requested RSI window."""
    for window in windows:
        data[f"RSI_{window}"] = rsi_calculate(data, window)
    return data


# Latest-row risk metric columns copied into the run DataFrame/CSV.
RISK_METRIC_COLS = [
    "daily_tr_return",
    "daily_close_return",
    "true_range",
    "true_range_pct",
    "atr_14_pct",
    "vol_24d",
    "vol_55d",
    "vol_ratio_24_55",
    "return_24d",
    "return_55d",
    "risk_adj_mom_24d",
    "drawdown_55d",
    "avg_volume_24d",
    "volume_ratio_24d",
    "ma_24d",
    "ma_55d",
    "price_vs_ma_24d",
    "price_vs_ma_55d",
    "ma_trend_24_55",
]


def calculate_risk_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Add total-return volatility, ATR, momentum, drawdown and volume metrics.

    Pandas port of the Excel sheet formulas: ``data`` holds one ticker sorted
    oldest-first, so shift/rolling naturally use the previous *available*
    trading day rather than assuming every share trades every day.
    Total-return metrics use the ``Adj Close`` column; ``close`` is kept as a
    cross-check via ``daily_close_return``.
    """
    tr_close = data["Adj Close"]
    prev_close = data["close"].shift(1)

    data["daily_tr_return"] = tr_close.pct_change(fill_method=None)
    data["daily_close_return"] = data["close"].pct_change(fill_method=None)

    # True range: intraday range plus overnight gaps vs previous close.
    data["true_range"] = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["true_range_pct"] = data["true_range"] / data["close"]
    data["atr_14_pct"] = data["true_range_pct"].rolling(window=14).mean()

    # Annualised total-return volatility (sample std, 252 trading days).
    data["vol_24d"] = data["daily_tr_return"].rolling(window=24).std() * np.sqrt(252)
    data["vol_55d"] = data["daily_tr_return"].rolling(window=55).std() * np.sqrt(252)
    data["vol_ratio_24_55"] = data["vol_24d"] / data["vol_55d"]

    data["return_24d"] = tr_close / tr_close.shift(24) - 1
    data["return_55d"] = tr_close / tr_close.shift(55) - 1
    data["risk_adj_mom_24d"] = data["return_24d"] / (
        data["vol_24d"] / np.sqrt(252 / 24)
    )

    data["drawdown_55d"] = tr_close / tr_close.rolling(window=55).max() - 1

    data["avg_volume_24d"] = data["volume"].rolling(window=24).mean()
    data["volume_ratio_24d"] = data["volume"] / data["avg_volume_24d"]

    # Total-return moving averages (distinct from MA24/MA55 on raw close).
    data["ma_24d"] = tr_close.rolling(window=24).mean()
    data["ma_55d"] = tr_close.rolling(window=55).mean()
    data["price_vs_ma_24d"] = tr_close / data["ma_24d"] - 1
    data["price_vs_ma_55d"] = tr_close / data["ma_55d"] - 1
    data["ma_trend_24_55"] = data["ma_24d"] / data["ma_55d"] - 1

    # Excel's IFERROR equivalent: anything that failed to calculate (short
    # history → NaN, zero denominator → ±inf) becomes 0 so downstream
    # round()/report code never sees NaN/inf or object-dtype columns.
    data[RISK_METRIC_COLS] = (
        data[RISK_METRIC_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    return data


def make_dates_timezone_naive(data: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone info from the 'date' column so dates compare cleanly."""
    data.loc[:, "date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
    return data


# Plotting
def plot_price_and_bollinger_bands_close(data: pd.DataFrame, ticker: str) -> None:
    """Plot 2 years of close price with MA24/MA55 and Bollinger bands to plots/{ticker}/adj_bollinger.png."""
    data = data.dropna(
        subset=["Bollinger_High", "Bollinger_Low", "close"]
    )  # Drop rows with NaN values
    data.loc[:, "date"] = pd.to_datetime(data["date"])
    end_date = data["date"].max()
    start_date = end_date - pd.DateOffset(years=2)
    data = data[(data["date"] >= start_date) & (data["date"] <= end_date)]

    # Ensure the directory exists
    os.makedirs(f"{graph_dir}/{ticker}", exist_ok=True)

    plt.figure(figsize=(16, 7))

    # Plotting the actual closing prices
    plt.plot(data["date"], data["close"], label=f"{ticker} Price", color="blue")

    # Plotting the moving averages
    if "MA24" in data and "MA55" in data:
        plt.plot(data["date"], data["MA24"], label="24-day MA", color="green")
        plt.plot(data["date"], data["MA55"], label="55-day MA", color="red")

    # Adding Bollinger bands
    if "Bollinger_High" in data and "Bollinger_Low" in data:
        plt.fill_between(
            data["date"],
            data["Bollinger_High"],
            data["Bollinger_Low"],
            color="grey",
            alpha=0.3,
        )

    # Set y-axis limits to ensure all data is visible
    plt.ylim(
        min(data["close"].min(), data["Bollinger_Low"].min()),
        max(data["close"].max(), data["Bollinger_High"].max()),
    )

    plt.title(f"Price momentum [{ticker}]")
    plt.xlabel("date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()

    # Formatting date
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()

    # Saving the plot
    ticker = ticker.replace(".JO", "")
    plt.savefig(f"{graph_dir}/{ticker}/adj_bollinger.png")
    plt.close()


def plot_overbought_oversold_close(data: pd.DataFrame, ticker: str, name: str) -> None:
    """Plot the MA-ratio overbought/oversold indicator (green above 0, red below)."""
    data = make_dates_timezone_naive(data)
    data.loc[:, "date"] = pd.to_datetime(data["date"])
    end_date = data["date"].max()
    start_date = end_date - pd.DateOffset(years=2)
    data = data[(data["date"] >= start_date) & (data["date"] <= end_date)]

    # Ensure the directory exists
    os.makedirs(f"{graph_dir}/{ticker}", exist_ok=True)

    plt.figure(figsize=(18, 6))

    # Plotting overbought/oversold with 0 as baseline
    plt.axhline(0, color="black", linewidth=1)
    plt.plot(
        data["date"],
        data["Overbought_Oversold"],
        label="Overbought/Oversold",
        color="black",
        linestyle="--",
    )

    # Highlighting overbought and oversold areas
    plt.fill_between(
        data["date"],
        0,
        data["Overbought_Oversold"],
        where=(data["Overbought_Oversold"] > 0),
        facecolor="green",
        alpha=0.3,
    )
    plt.fill_between(
        data["date"],
        0,
        data["Overbought_Oversold"],
        where=(data["Overbought_Oversold"] < 0),
        facecolor="red",
        alpha=0.3,
    )

    plt.title(f"Overbought/Oversold for {name} ({ticker})")
    plt.xlabel("date")
    plt.ylabel("Overbought/Oversold")
    plt.legend()
    plt.grid()

    # Formatting date
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()

    # Saving the plot
    ticker = ticker.replace(".JO", "")
    plt.savefig(f"{graph_dir}/{ticker}/adj_overbought_oversold.png")
    plt.close()


def plot_overbought_oversold_zar(data: pd.DataFrame, ticker: str) -> None:
    """Overbought/oversold plot for the ZAR/USD rate.

    Colours are inverted relative to equities: a rising ZAR/USD number means a
    weaker rand, so positive readings are shaded red and negative ones green.
    """
    data.loc[:, "date"] = pd.to_datetime(data["date"])
    end_date = data["date"].max()
    start_date = end_date - pd.DateOffset(years=2)
    data = data[(data["date"] >= start_date) & (data["date"] <= end_date)]

    # Ensure the directory exists
    os.makedirs(f"{graph_dir}/{ticker}", exist_ok=True)

    plt.figure(figsize=(18, 6))

    # Plotting overbought/oversold with 0 as baseline
    plt.axhline(0, color="black", linewidth=1)
    plt.plot(
        data["date"],
        data["Overbought_Oversold"],
        label="Overbought/Oversold",
        color="black",
        linestyle="--",
    )

    # Highlighting overbought and oversold areas
    plt.fill_between(
        data["date"],
        0,
        data["Overbought_Oversold"],
        where=(data["Overbought_Oversold"] > 0),
        facecolor="red",
        alpha=0.3,
    )
    plt.fill_between(
        data["date"],
        0,
        data["Overbought_Oversold"],
        where=(data["Overbought_Oversold"] < 0),
        facecolor="green",
        alpha=0.3,
    )

    plt.title(f"Overbought/Oversold for {ticker}")
    plt.xlabel("date")
    plt.ylabel("Overbought/Oversold")
    plt.legend()
    plt.grid()

    # Formatting date
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()

    # Saving the plot
    ticker = ticker.replace(".JO", "")
    plt.savefig(f"{graph_dir}/{ticker}/adj_overbought_oversold.png")
    plt.close()


def plot_stock_close_last_two_years(
    unscaled_close: pd.DataFrame,
    ticker: str,
    next_week_predictions: list[float],
    next_month_predictions: list[float],
    name: str,
) -> None:
    """Plot the last 3 months of close prices with the 7- and 30-day prediction paths."""
    logger.info(
        f"Starting close to plot stock data for the last year for ticker: {ticker}"
    )
    plt.figure(figsize=(16, 8))

    # Ensure the directory exists
    ticker_clean = ticker.replace(".JO", "")
    dir_path = f"{graph_dir}/{ticker_clean}"
    try:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Directory created or already exists: {dir_path}")
    except Exception as e:
        logger.error(f"Error creating directory {dir_path}: {e}")
        return

    # Filter for the last 3 months of data
    try:
        last_date = unscaled_close["date"].max()
        three_months_ago = last_date - pd.DateOffset(months=3)
        last_three_months = unscaled_close[unscaled_close["date"] >= three_months_ago]
        logger.info(f"Filtered data for the last three months successfully")
    except Exception as e:
        logger.error(f"Error filtering data for the last three months: {e}")
        return

    try:
        plt.plot(
            last_three_months["date"],
            last_three_months["close"],
            label="Historical Data",
            color="blue",
        )
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
            logger.warning(
                f"Length mismatch: next_week_dates ({len(next_week_dates)}) and next_week_predictions ({len(next_week_predictions)})"
            )
            if len(next_week_dates) > len(next_week_predictions):
                next_week_dates = next_week_dates[: len(next_week_predictions)]
            else:
                next_week_predictions = next_week_predictions[: len(next_week_dates)]

        if len(next_month_dates) != len(next_month_predictions):
            logger.warning(
                f"Length mismatch: next_month_dates ({len(next_month_dates)}) and next_month_predictions ({len(next_month_predictions)})"
            )
            if len(next_month_dates) > len(next_month_predictions):
                next_month_dates = next_month_dates[: len(next_month_predictions)]
            else:
                next_month_predictions = next_month_predictions[: len(next_month_dates)]

        try:
            plt.plot(
                next_week_dates,
                next_week_predictions,
                label="Next Week Predictions",
                color="cyan",
            )
            plt.plot(
                next_month_dates,
                next_month_predictions,
                label="Next Month Predictions",
                color="magenta",
            )
            logger.info(f"Plotted predictions data")
        except Exception as e:
            logger.error(f"Error plotting predictions data: {e}")
            return

    # Date formatting for x-axis
    try:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.gcf().autofmt_xdate()
        logger.info(f"Formatted x-axis")
    except Exception as e:
        logger.error(f"Error formatting x-axis: {e}")
        return

    try:
        plt.title(f"Close Prediction for {name} ({ticker})")
        plt.xlabel("date")
        plt.ylabel("Price (R)")
        plt.legend()
        logger.info(f"Set plot titles and labels")
    except Exception as e:
        logger.error(f"Error setting plot titles and labels: {e}")
        return

    # Save plot to file
    try:
        file_path = os.path.join(dir_path, "close_prediction.png")
        plt.savefig(file_path)
        plt.close()
        logger.info(f"Plot saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving plot to file: {e}")


def plot_model_vs_actual(
    model: SequentialType,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    debug_plot_path: str,
    scaler: MinMaxScaler | None = None,
    focus_last_n: int = 200,
) -> None:
    """Diagnostic plot: predictions vs actuals, residuals and their distribution.

    Only used when debugging model quality; saves a 4-panel figure to
    *debug_plot_path* annotated with MSE/MAE/R² for the test split.
    """
    logger.info(f"Starting close to plot model vs actual data with enhanced insights.")

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
    fig, axs = plt.subplots(
        4, 1, figsize=(16, 24), gridspec_kw={"height_ratios": [2, 1, 1, 1]}
    )
    fig.suptitle(
        "Model Predictions vs Actual Values (Last {} Data Points)".format(focus_last_n)
    )

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
    ax1.plot(y_train, label="Actual Train Data", color="blue", alpha=0.7, linewidth=2)
    ax1.plot(
        y_train_pred,
        label="Predicted Train Data",
        color="green",
        alpha=0.7,
        linewidth=2,
    )

    # Plot testing data
    ax1.plot(
        np.arange(len(y_train), len(y_train) + len(y_test)),
        y_test,
        label="Actual Test Data",
        color="orange",
        alpha=0.7,
        linewidth=2,
    )
    ax1.plot(
        np.arange(len(y_train), len(y_train) + len(y_test)),
        y_test_pred,
        label="Predicted Test Data",
        color="red",
        alpha=0.7,
        linewidth=2,
    )

    ax1.set_ylabel("Price (R)")
    ax1.legend()

    # Annotate the plot with error metrics
    metrics_text = f"MSE: {mse:.5f}\nMAE: {mae:.5f}\nR²: {r2:.5f}"
    ax1.text(
        0.02,
        0.95,
        metrics_text,
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Residuals plot on the second subplot
    ax2 = axs[1]
    ax2.plot(train_residuals, label="Train Residuals", color="blue", alpha=0.7)
    ax2.plot(
        np.arange(len(y_train), len(y_train) + len(y_test)),
        test_residuals,
        label="Test Residuals",
        color="orange",
        alpha=0.7,
    )
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_ylabel("Residuals")
    ax2.legend()

    # Zoomed-in view on the third subplot
    ax3 = axs[2]
    zoom_range = 50  # Adjust this value as needed
    ax3.plot(
        y_test[-zoom_range:],
        label="Actual Test Data (Zoomed)",
        color="orange",
        alpha=0.7,
        linewidth=2,
    )
    ax3.plot(
        y_test_pred[-zoom_range:],
        label="Predicted Test Data (Zoomed)",
        color="red",
        alpha=0.7,
        linewidth=2,
    )
    ax3.set_ylabel("Price (R)")
    ax3.legend()

    # Residuals distribution on the fourth subplot
    ax4 = axs[3]
    ax4.hist(
        test_residuals,
        bins=50,
        color="orange",
        alpha=0.7,
        label="Test Residuals Distribution",
    )
    ax4.set_xlabel("Residuals")
    ax4.set_ylabel("Frequency")
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


def plot_volume_data_last_two_years(
    unscaled_volume: pd.DataFrame,
    ticker: str,
    next_week_volume_predictions: list[float] = [],
    next_month_volume_predictions: list[float] = [],
    name: str = "",
) -> None:
    """Plot the last 3 months of traded volume to plots/{ticker}/volume.png."""
    logger.info(
        f"Starting close to plot volume data for the last year for ticker: {ticker}"
    )
    plt.figure(figsize=(16, 8))

    # Ensure the directory exists
    ticker_clean = ticker.replace(".JO", "")
    dir_path = f"{graph_dir}/{ticker_clean}"
    try:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Directory created or already exists: {dir_path}")
    except Exception as e:
        logger.error(f"Error creating directory {dir_path}: {e}")
        return

    # Filter for the last 3 months of data
    try:
        last_date = unscaled_volume["date"].max()
        three_months_ago = last_date - pd.DateOffset(months=3)
        last_three_months = unscaled_volume[unscaled_volume["date"] >= three_months_ago]
        logger.info(f"Filtered data for the last three months successfully")
    except Exception as e:
        logger.error(f"Error filtering data for the last three months: {e}")
        return

    try:
        plt.plot(
            last_three_months["date"],
            last_three_months["volume"],
            label="Historical volume",
            color="blue",
        )
        logger.info(f"Plotted historical volume data")
    except Exception as e:
        logger.error(f"Error plotting historical volume data: {e}")
        return

    # Date formatting for x-axis
    try:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.gcf().autofmt_xdate()
        logger.info(f"Formatted x-axis")
    except Exception as e:
        logger.error(f"Error formatting x-axis: {e}")
        return

    try:
        plt.title(f"Volume for {name} ({ticker})")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.legend()
        logger.info(f"Set plot titles and labels")
    except Exception as e:
        logger.error(f"Error setting plot titles and labels: {e}")
        return

    # Save plot to file
    try:
        file_path = os.path.join(dir_path, "volume.png")
        plt.savefig(file_path)
        plt.close()
        logger.info(f"volume plot saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving volume plot to file: {e}")


# Processing data
def process_ticker_close(ticker: str, commodity: bool, name: str) -> pd.DataFrame:
    """Build indicator CSVs and Bollinger/overbought-oversold charts for one ticker.

    Pulls ~4 years of history from the DB (or the commodities view when
    *commodity* is truthy), computes MA/Bollinger/Z-score/RSI columns and
    writes the intermediate frames under ``data/{ticker}/``.
    """
    os.makedirs(os.path.join("data", ticker.replace(".JO", "")), exist_ok=True)
    # Fetching data for the specified period
    starttime_dt = datetime.now() - timedelta(days=1440)
    start_date = starttime_dt.strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    logger.debug(
        Fore.LIGHTGREEN_EX
        + f"Creating close bollinger data for: {ticker}\n"
        + Fore.RESET
    )

    if not commodity:
        stock_data = db_queries.get_ticker_from_db_with_date_select(
            ticker, start_date, end_date
        )
    else:
        stock_data = db_queries.get_commodities_from_db(ticker)

    ticker = ticker.replace(".JO", "")

    stock_data.to_csv(os.path.join("data", f"{ticker}", "DataFrame.csv"))

    stock_data = pd.read_csv(os.path.join("data", f"{ticker}", "DataFrame.csv"))
    try:
        stock_data["date"] = pd.to_datetime(stock_data["date"])
    except KeyError:
        stock_data["date"] = pd.to_datetime(stock_data.index)

    stock_data = calculate_moving_averages(stock_data)
    stock_data.to_csv(os.path.join("data", f"{ticker}", "MovingAverages.csv"))
    stock_data = calculate_bollinger_bands(stock_data)
    stock_data.to_csv(os.path.join("data", f"{ticker}", "Bollinger bands.csv"))
    stock_data = calculate_z_score(stock_data)
    stock_data = calculate_rsi_close_for_all(stock_data, windows=[14])

    # pyplot is global/not thread-safe; serialise the actual drawing while the
    # slow DB read above is free to overlap across pool workers.
    with _plot_lock:
        plot_price_and_bollinger_bands_close(stock_data, ticker)
        plot_overbought_oversold_close(stock_data, ticker, name)

    return stock_data


def process_zar_bollinger() -> None:
    """Download ~4 years of ZAR/USD rates and render its Bollinger/momentum charts."""
    ticker = "ZAR"
    os.makedirs(os.path.join("data", ticker), exist_ok=True)
    # Fetching data for the specified period
    starttime_dt = datetime.now() - timedelta(days=1440)
    start_date = starttime_dt.strftime("%Y-%m-%d")
    logger.debug(Fore.LIGHTGREEN_EX + f"Creating bollinger data for: {ticker}")
    logger.debug(Fore.RESET)

    zar_data: pd.DataFrame = yf.download("ZAR=X", start=start_date, interval="1d")
    zar_data.reset_index(inplace=True)

    if zar_data.empty:
        logging.info("No ZAR data to download.")
        return

    tickerName = ticker.replace(".JO", "")

    zar_data["close"] = zar_data["Close"]

    zar_data.to_csv(os.path.join("data", f"{tickerName}", "yfdata.csv"))

    # yfinance writes a second header row (the ticker level of its MultiIndex
    # columns); drop it so the CSV parses back into a flat frame.
    lines: list
    with open(os.path.join("data", f"{tickerName}", "yfdata.csv"), "r") as f:
        lines = f.readlines()

    lines.pop(1)

    os.remove(os.path.join("data", f"{tickerName}", "yfdata.csv"))

    with open(os.path.join("data", f"{tickerName}", "yfdata.csv"), "w+") as f:
        f.writelines(lines)

    zar_data = pd.read_csv(os.path.join("data", f"{tickerName}", "yfdata.csv"))
    zar_data.reset_index(drop=True, inplace=True)

    # Ensure the Date column is in datetime format
    zar_data.dropna(subset=["Date"], inplace=True)

    ticker = ticker.replace(".JO", "")

    zar_data.to_csv(os.path.join("data", f"{ticker}", "DataFrame.csv"))

    zar_data = pd.read_csv(os.path.join("data", f"{ticker}", "DataFrame.csv"))
    zar_data["date"] = pd.to_datetime(zar_data["Date"])

    zar_data = calculate_moving_averages(zar_data)
    zar_data = calculate_bollinger_bands(zar_data)
    zar_data = calculate_z_score(zar_data)

    plot_price_and_bollinger_bands_close(zar_data, ticker)
    plot_overbought_oversold_zar(zar_data, ticker)


# ML Stuff
def calculate_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, tolerance: float = 0.05
) -> float:
    """Calculate the percentage of predictions within a tolerance of the actual values."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.mean(np.abs((y_true - y_pred) / y_true) <= tolerance)
    return float(accuracy * 100)


def get_data_hash(data: pd.DataFrame) -> str:
    """Calculate the hash of the data to check if it has changed."""
    data_str = data.to_json()
    return hashlib.md5(data_str.encode()).hexdigest()


def create_sequences(
    data: np.ndarray, seq_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a 1-D series into (samples, seq_length, 1) windows and next-step targets."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])

    X = np.array(X)
    y = np.array(y)

    # Reshape X if it's not in the correct 2D shape
    if (
        len(X.shape) == 2
    ):  # X is 2D, but we need 3D for LSTM (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y


def train_new_model(
    X: np.ndarray,
    y: np.ndarray,
    model_dir: str,
    model_path: str,
    hparams: ModelHyperParams,
    sanitized_ticker: str,
) -> SequentialType:
    """Train a fresh two-layer LSTM for a ticker and persist it with metadata.

    Used as a fallback when no saved model exists for a ticker on report day;
    the dedicated training pipeline lives in ``training/close.py``.
    Saves the model to *model_path* and accuracy metrics to ``metadata.json``.
    """
    os.makedirs(model_dir, exist_ok=True)

    model: SequentialType = Sequential()
    model.add(
        LSTM(
            units=hparams.lstm_units, return_sequences=True, input_shape=(X.shape[1], 1)
        )
    )
    model.add(Dropout(hparams.dropout))
    model.add(LSTM(units=hparams.lstm_units))
    model.add(Dropout(hparams.dropout))
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")

    # Fit the model
    model.fit(X, y, epochs=hparams.epochs, batch_size=64, validation_split=0.1)

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
        "last_trained_date": str(pd.to_datetime("today")),
        "mean_squared_error": mse,
        "mean_absolute_error": mae,
        "r2_score": r2,
        "predictions_within_5_percent": accuracy,
        "predictions_within_10_percent": accuracy10,
        "predictions_within_15_percent": accuracy15,
        "predictions_within_30_percent": accuracy30,
        "predictions_within_50_percent": accuracy50,
    }

    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
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


def load_model_metadata(model_dir: str) -> dict[str, Any] | None:
    """Return the parsed ``metadata.json`` for a model dir, or None if absent."""
    metadata_path = os.path.join(model_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata: dict[str, Any] = json.load(f)
            return metadata
    return None


def save_predictions_to_db(
    ticker: str, start_date: datetime, next_month_predictions: list[float]
) -> None:
    """Persist daily predictions to the ``predictions`` table.

    Currently disabled: the insert loop is commented out, so this is a no-op
    kept for when per-day prediction storage is re-enabled.
    """
    # Generate the dates corresponding to the predictions
    prediction_dates = [
        start_date + timedelta(days=i)
        for i in range(1, len(next_month_predictions) + 1)
    ]

    # Insert each prediction into the database
    """for date, close in zip(prediction_dates, next_month_predictions):
        db_queries.insert_prediction(date=date, close=close, code=ticker)"""


def predict_close_value(
    hist: pd.DataFrame, hparams: ModelHyperParams, ticker: str
) -> PredictionResult:
    """Predict the close price 7 and 30 days ahead with the ticker's LSTM model.

    The model is fed the last ``seq_length`` scaled closes and rolled forward
    one day at a time, feeding each prediction back into the input window.

    Returns:
        PredictionResult: the predicted prices (in rand, i.e. inverse-scaled) at
        the 7- and 30-day horizons plus the full day-by-day prediction paths
        used for plotting.
    """
    logger.info(f"Starting close prediction for ticker: {ticker}")

    # Scale a copy of the close series; leave the caller's DataFrame untouched.
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(hist[["close"]]).flatten()
    seq_length = 60
    X, y = create_sequences(scaled_close, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    sanitized_ticker = sanitize_ticker(ticker)
    model_ticker = sanitized_ticker.replace(".JO", "")
    model_dir = os.path.join("models", model_ticker)
    model_path = os.path.join(model_dir, f"{model_ticker}_Close_Model.keras")

    last_date_in_data = hist["date"].max()

    # Each ticker's model is used exactly once per run, so caching every model
    # only pins GPU memory — 208 LSTMs would blow the ~6-8 GB VRAM budget while
    # never producing a cache hit. Load it, use it, and free it again at the end
    # of this call (see the clear_session() below).
    if os.path.exists(model_path):
        model: SequentialType = load_model(model_path)
        logger.info(f"Loaded model for {ticker} from disk")
    else:
        logger.info(f"No existing model for {ticker}. Training new model.")
        model = train_new_model(
            X_train, y_train, model_dir, model_path, hparams, sanitized_ticker
        )

    # Roll the model forward: 7 steps for the week horizon, then 30 more for
    # the month horizon, each prediction appended to the input window.
    last_sequence = scaled_close[-seq_length:].reshape((1, seq_length, 1))
    next_week_predictions = []
    next_month_predictions = []

    with tf.device("/GPU:0"):
        for _ in range(7):
            next_week_prediction = model(last_sequence, training=False)[0][0].numpy()
            next_week_predictions.append(next_week_prediction)
            last_sequence = np.append(
                last_sequence[:, 1:, :], [[[next_week_prediction]]], axis=1
            )

        for _ in range(30):
            next_month_prediction = model(last_sequence, training=False)[0][0].numpy()
            next_month_predictions.append(next_month_prediction)
            last_sequence = np.append(
                last_sequence[:, 1:, :], [[[next_month_prediction]]], axis=1
            )

    # Map the scaled model outputs back to actual prices.
    next_week_predictions = scaler.inverse_transform(
        np.array(next_week_predictions).reshape(-1, 1)
    ).flatten()
    next_month_predictions = scaler.inverse_transform(
        np.array(next_month_predictions).reshape(-1, 1)
    ).flatten()

    save_predictions_to_db(ticker, last_date_in_data, next_month_predictions)

    # Free this ticker's model from (GPU) memory before moving to the next one;
    # otherwise VRAM use grows linearly with the number of tickers.
    del model
    clear_session()

    # Return real (inverse-scaled) prices at each horizon, not the raw scaled
    # model output — previously the scaled value leaked out here and made the
    # downstream "Next Week/Month Prediction" columns meaningless.
    return PredictionResult(
        next_week_price=float(next_week_predictions[-1]),
        next_month_price=float(next_month_predictions[-1]),
        next_week_path=next_week_predictions.tolist(),
        next_month_path=next_month_predictions.tolist(),
    )


def apply_sentiment_adjustment(
    next_week_pred: float,
    next_month_pred: float,
    current_price: float,
    ticker: str,
) -> SentimentAdjustment:
    """
    Fetch the latest news-sentiment score for *ticker* and apply a small
    linear adjustment to the LSTM predictions.

    The adjustment is intentionally conservative: at a compound score of ±1.0
    the week prediction shifts by ±SENTIMENT_WEEK_WEIGHT of the current price.

    Returns a SentimentAdjustment with the adjusted week/month prices and the
    sentiment score that drove them.
    """
    try:
        score = db_queries.get_latest_sentiment_score(ticker)
    except Exception as exc:
        logger.warning(f"Could not fetch sentiment for {ticker}: {exc}")
        score = 0.0

    if score == 0.0 or current_price <= 0:
        return SentimentAdjustment(next_week_pred, next_month_pred, score)

    week_adj = score * SENTIMENT_WEEK_WEIGHT * current_price
    month_adj = score * SENTIMENT_MONTH_WEIGHT * current_price

    return SentimentAdjustment(
        next_week_price=round(next_week_pred + week_adj, 4),
        next_month_price=round(next_month_pred + month_adj, 4),
        sentiment_score=score,
    )


def fetch_data(hparams: ModelHyperParams) -> FetchResult:
    """Run the full per-ticker pipeline: history, indicators, predictions, charts.

    For every stock in the universe this fetches ~11 years of history,
    computes the technical indicators, generates the 7/30-day LSTM price
    predictions (with the news-sentiment bias applied) and renders the chart
    images embedded in the reports.

    Returns:
        FetchResult: the populated universe DataFrame, the per-ticker chart
        images and the projected portfolio values. The ``Next Week/Month
        Prediction`` columns hold the predicted *fractional change* vs the
        current price (e.g. 0.05 = +5%), which the report layer multiplies by 100.
    """
    logger.info("Starting close data fetch process")
    stocks_df: pd.DataFrame = db_queries.fetch_stock_universe_from_db()
    stock_images: list[StockChartImages] = []
    total_value_next_week: float = 0.0
    total_value_next_month: float = 0.0
    starttime_dt = datetime.now() - timedelta(days=4015)
    start_date = starttime_dt.strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    for index, row in stocks_df.iterrows():
        try_count = 0
        ticker = row["code"]
        name = row["share_name"]
        logger.info(f"Processing ticker: {ticker}, name: {name}")

        while try_count < 5:
            try:
                logger.info(
                    f"Attempting to fetch data for {ticker} (Attempt {try_count + 1})"
                )

                if not row["commodity"]:
                    hist = db_queries.get_ticker_from_db_with_date_select(
                        ticker, start_date, end_date
                    )
                    if hist.empty:
                        raise ValueError(f"No data found for {ticker}")
                else:
                    hist = db_queries.get_commodities_from_db(ticker)
                    if hist.empty:
                        raise ValueError(f"No data found for {ticker}")

                # Ensure `hist` is a DataFrame and dates are timezone-naive
                hist = make_dates_timezone_naive(hist)
                hist.reset_index(inplace=True)
                hist["date"] = pd.to_datetime(hist["date"])
                unscaled_close = hist[["date", "close"]].copy()
                unscaled_volume = hist[["date", "volume"]].copy()
                logger.info(f"Data successfully fetched for {ticker}")
                break

            except Exception as e:
                logger.error(
                    f"Error fetching data for {ticker}: {e}. Retrying in 5 seconds..."
                )
                try_count += 1
                time.sleep(5)

        if try_count == 5:
            logger.error(
                f"Failed to fetch data for {ticker} after 5 attempts. Skipping ticker."
            )
            continue

        try:
            logger.info(f"Calculating metrics for {ticker}")
            current_price = round(hist.iloc[-1]["close"], 2)

            # Ensure hist remains a DataFrame after each transformation
            hist = calculate_moving_averages(hist)
            hist = calculate_bollinger_bands(hist)
            hist = calculate_z_score(hist)
            hist = calculate_rsi_close_for_all(hist, windows=[14])
            hist = calculate_risk_metrics(hist)

            # Convert back to DataFrame if any function accidentally returns a NumPy array
            if isinstance(hist, np.ndarray):
                hist = pd.DataFrame(hist)

            logger.info(f"Metrics calculated for {ticker}")

            stocks_df.at[index, "Current Price"] = round(current_price, 2)
            current_value = round(
                current_price * row.get("Initial Amount of Stocks", 1), 2
            )
            z_score = round(hist.iloc[-1]["Z-Score"], 2)
            overbought_oversold = round(hist.iloc[-1]["Overbought_Oversold"], 2)

            next_week_predictions: list[float]
            next_month_predictions: list[float]
            if not PREDICTION:
                # Predictions disabled: report a 0% expected change.
                next_week_prediction = next_month_prediction = current_price
                next_month_predictions = next_week_predictions = []
                sentiment_score = 0.0
            else:
                logger.info(f"Generating predictions for {ticker}")
                prediction = predict_close_value(hist, hparams, ticker)
                next_week_predictions = prediction.next_week_path
                next_month_predictions = prediction.next_month_path
                # Apply news-sentiment bias on top of LSTM prediction
                adjustment = apply_sentiment_adjustment(
                    prediction.next_week_price,
                    prediction.next_month_price,
                    current_price,
                    ticker,
                )
                next_week_prediction = adjustment.next_week_price
                next_month_prediction = adjustment.next_month_price
                sentiment_score = adjustment.sentiment_score
                logger.info(
                    f"Predictions for {ticker}: week={next_week_prediction:.2f}  "
                    f"month={next_month_prediction:.2f}  sentiment={sentiment_score:+.3f}"
                )

            stocks_df.at[index, "Current Value"] = round(current_value, 2)

            # Accumulate the projected portfolio value at each horizon
            # (predicted price × shares held). These were previously left at 0,
            # so the report's "Projected Portfolio Value" lines always showed R0.00.
            shares_held = row.get("Initial Amount of Stocks", 1)
            total_value_next_week += round(next_week_prediction * shares_held, 2)
            total_value_next_month += round(next_month_prediction * shares_held, 2)

            # Store predictions as fractional change vs the current price
            # (0.05 = +5%); the report layer multiplies these by 100.
            if current_price > 0:
                stocks_df.at[index, "Next Week Prediction"] = round(
                    next_week_prediction / current_price - 1, 4
                )
                stocks_df.at[index, "Next Month Prediction"] = round(
                    next_month_prediction / current_price - 1, 4
                )
            else:
                stocks_df.at[index, "Next Week Prediction"] = 0.0
                stocks_df.at[index, "Next Month Prediction"] = 0.0
            stocks_df.at[index, "Sentiment Score"] = round(sentiment_score, 3)
            stocks_df.at[index, "Z-Score"] = round(z_score, 2)
            stocks_df.at[index, "Overbought_Oversold"] = round(overbought_oversold, 2)
            stocks_df.at[index, "Overbought_Oversold_Value"] = round(
                overbought_oversold + 1, 2
            )
            stocks_df.at[index, "MA24"] = hist.iloc[-1]["MA24"]
            stocks_df.at[index, "MA55"] = hist.iloc[-1]["MA55"]
            stocks_df.at[index, "Volume"] = hist.iloc[-1]["volume"]
            for metric_col in RISK_METRIC_COLS:
                stocks_df.at[index, metric_col] = round(hist.iloc[-1][metric_col], 4)

            logger.info(f"Data updated for {ticker} in DataFrame")

            logger.info(f"Generating plots for {ticker}")
            plot_stock_close_last_two_years(
                unscaled_close,
                ticker.replace(".JO", ""),
                next_week_predictions,
                next_month_predictions,
                name,
            )
            if not row["commodity"]:
                plot_volume_data_last_two_years(
                    unscaled_volume, ticker.replace(".JO", "")
                )
            logger.info(f"Plots generated for {ticker}")

            clean_ticker = ticker.replace(".JO", "")
            # Equities carry a traded-volume chart; commodities do not.
            volume_chart = (
                encode_image(process_image(f"{graph_dir}/{clean_ticker}/volume.png"))
                if not row["commodity"]
                else None
            )
            stock_images.append(
                StockChartImages(
                    code=clean_ticker,
                    name=name,
                    adj_prediction=encode_image(
                        process_image(
                            f"{graph_dir}/{clean_ticker}/close_prediction.png"
                        )
                    ),
                    bollinger=encode_image(
                        process_image(f"{graph_dir}/{clean_ticker}/adj_bollinger.png")
                    ),
                    overbought_oversold=encode_image(
                        process_image(
                            f"{graph_dir}/{clean_ticker}/adj_overbought_oversold.png"
                        )
                    ),
                    volume_prediction=volume_chart,
                )
            )
            logger.info(f"Images encoded for {ticker}")

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}. Skipping to next ticker.")
            continue

    # stocks_df.drop(columns=['Commodity'], inplace=True)
    logger.info("Data fetch process completed")
    return FetchResult(
        stocks=stocks_df,
        images=stock_images,
        total_value_next_week=total_value_next_week,
        total_value_next_month=total_value_next_month,
    )


def generate_bollinger_and_overbought_oversold_close() -> None:
    """Generate Bollinger/overbought-oversold charts for the whole universe.

    Work is spread over a small thread pool (one worker per core, capped) so the
    slow per-ticker DB reads overlap, instead of spawning ~200 threads 1s apart.
    matplotlib's pyplot state is not thread-safe, so the drawing itself is
    serialised via _plot_lock inside process_ticker_close.
    """
    os.makedirs(graph_dir, exist_ok=True)

    df: pd.DataFrame = db_queries.fetch_stock_universe_from_db()

    max_workers = min(8, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_ticker_close, row["code"], row["commodity"], row["share_name"]
            ): row["code"]
            for _, row in df.iterrows()
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                future.result()
            except Exception as ex:
                logger.error(f"Bollinger generation failed for {ticker}: {ex}")


def calculate_rsi_close(
    data: pd.DataFrame, market: pd.DataFrame, sector: pd.DataFrame, ticker: str
) -> RSIComparison:
    """Relative strength of a stock vs its market and sector benchmarks.

    Despite the RSI naming this is a price-relative measure: the stock's
    20/60/120-day price ratio divided by the benchmark's ratio over the same
    window (1.0 = moved in line with the benchmark). Any leg that fails —
    e.g. missing benchmark history — is logged and reported as 0.
    """
    rsi_1m_sector: float = 0.0
    rsi_3m_sector: float = 0.0
    rsi_6m_sector: float = 0.0
    rsi_1m_market: float = 0.0
    rsi_3m_market: float = 0.0
    rsi_6m_market: float = 0.0

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
    ticker = ticker.replace(".JO", "")
    os.makedirs(os.path.join("data", ticker), exist_ok=True)
    combined_df.to_csv(f"data/{ticker}/unified_data.csv")

    try:
        stock_now = data["close"].iloc[-1]

        stock_20_day = data["close"].iloc[-20]

        stock_60_day = data["close"].iloc[-60]

        stock_120_day = data["close"].iloc[-120]

    except Exception as ex:
        logger.error(Fore.RED + f"{ex} for {ticker}")
        logger.error(Fore.RESET)

    try:
        market_now = market["close"].iloc[-1]

        market_20_day = market["close"].iloc[-20]

        market_60_day = market["close"].iloc[-60]

        market_120_day = market["close"].iloc[-120]

        # Market RSI
        stock_on_stock_20_day = stock_now / stock_20_day
        market_on_market_20_day = market_now / market_20_day
        rsi_20_day = stock_on_stock_20_day / market_on_market_20_day

        stock_on_stock_60_day = stock_now / stock_60_day
        market_on_market_60_day = market_now / market_60_day
        rsi_60_day = stock_on_stock_60_day / market_on_market_60_day

        stock_on_stock_120_day = stock_now / stock_120_day
        market_on_market_120_day = market_now / market_120_day
        rsi_120_day = stock_on_stock_120_day / market_on_market_120_day

        rsi_1m_market = rsi_20_day
        rsi_3m_market = rsi_60_day
        rsi_6m_market = rsi_120_day

    except Exception as ex:
        logger.error(Fore.RED + f"{ex} for {ticker}")
        logger.error(Fore.RESET)

    try:
        sector_now = sector["close"].iloc[-1]

        sector_20_day = sector["close"].iloc[-20]

        sector_60_day = sector["close"].iloc[-60]

        sector_120_day = sector["close"].iloc[-120]

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
        logger.error(Fore.RED + f"{ex} for Close RSI Calculation {ticker}")
        logger.error(Fore.RESET)

    return RSIComparison(
        rsi_1m_sector=rsi_1m_sector,
        rsi_3m_sector=rsi_3m_sector,
        rsi_6m_sector=rsi_6m_sector,
        rsi_1m_market=rsi_1m_market,
        rsi_3m_market=rsi_3m_market,
        rsi_6m_market=rsi_6m_market,
    )


def add_close_rsi_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Add SECTOR/MARKET RSI 1M/3M/6M columns to the universe DataFrame.

    Each stock is compared against the benchmark tickers configured in its
    ``rsi_comparison_sector`` / ``rsi_comparison_market`` columns using two
    years of history. Failures fall back to 0 for all six columns.
    """
    starttime_dt = datetime.now() - timedelta(weeks=104)
    start_date = starttime_dt.strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    for index, row in df.iterrows():
        try:
            ticker = row["code"]
            logger.debug(
                Fore.LIGHTMAGENTA_EX
                + f"Generating Close RSI for: {ticker}"
                + Fore.RESET
            )
            """if not row['rsi_comparison_sector'] or not row['rsi_comparison_market']:
                df.at[index, 'SECTOR RSI 1M'] = round(0, 2)
                df.at[index, 'SECTOR RSI 3M'] = round(0, 2)
                df.at[index, 'SECTOR RSI 6M'] = round(0, 2)

                df.at[index, 'MARKET RSI 1M'] = round(0, 2)
                df.at[index, 'MARKET RSI 3M'] = round(0, 2)
                df.at[index, 'MARKET RSI 6M'] = round(0, 2)
                continue"""

            comparison_sector = row["rsi_comparison_sector"]
            comparison_market = row["rsi_comparison_market"]
            # Load historical data for the ticker

            historical_data = db_queries.get_ticker_from_db_with_date_select(
                f"{ticker}", start_date=start_date, end_date=end_date
            )

            comparison_sector_data = db_queries.get_ticker_from_db_with_date_select(
                f"{comparison_sector}", start_date=start_date, end_date=end_date
            )

            comparison_market_data = db_queries.get_ticker_from_db_with_date_select(
                f"{comparison_market}", start_date=start_date, end_date=end_date
            )

            # Calculate RSI for the last 1 month, 3 months, and 6 months
            rsi = calculate_rsi_close(
                historical_data, comparison_market_data, comparison_sector_data, ticker
            )

            df.at[index, "SECTOR RSI 1M"] = round(rsi.rsi_1m_sector, 2)
            df.at[index, "SECTOR RSI 3M"] = round(rsi.rsi_3m_sector, 2)
            df.at[index, "SECTOR RSI 6M"] = round(rsi.rsi_6m_sector, 2)

            df.at[index, "MARKET RSI 1M"] = round(rsi.rsi_1m_market, 2)
            df.at[index, "MARKET RSI 3M"] = round(rsi.rsi_3m_market, 2)
            df.at[index, "MARKET RSI 6M"] = round(rsi.rsi_6m_market, 2)

        except Exception as ex:
            logger.error(Fore.RED + f"Error calculating Close RSI for {ticker}:")
            logger.error(ex)
            logger.error(Fore.RESET)
            df.at[index, "SECTOR RSI 1M"] = round(0, 2)
            df.at[index, "SECTOR RSI 3M"] = round(0, 2)
            df.at[index, "SECTOR RSI 6M"] = round(0, 2)

            df.at[index, "MARKET RSI 1M"] = round(0, 2)
            df.at[index, "MARKET RSI 3M"] = round(0, 2)
            df.at[index, "MARKET RSI 6M"] = round(0, 2)

    return df


# Reporting
def upload_to_spaces(
    file_path: str,
    spaces_access_key: str,
    spaces_secret_key: str,
    bucket_name: str,
    region_name: str,
    endpoint_url: str,
    today: str,
) -> str:
    """Upload a report file to DigitalOcean Spaces and return its public URL."""
    session = boto3.session.Session()
    client = session.client(
        "s3",
        region_name=region_name,
        endpoint_url=endpoint_url,
        aws_access_key_id=spaces_access_key,
        aws_secret_access_key=spaces_secret_key,
    )

    # Create the directory structure
    file_name = os.path.basename(file_path)
    remote_path = f"reports/{today}/{file_name}"

    client.upload_file(
        file_path, bucket_name, remote_path, ExtraArgs={"ACL": "public-read"}
    )

    return f"{endpoint_url}/{bucket_name}/{remote_path}"


def prepare_stock_images(top_bottom_data: TopBottomData) -> list[RankedStockImage]:
    """Collect base64 chart images for every ticker appearing in a top/bottom-10 list."""
    stock_images: list[RankedStockImage] = []
    added_tickers: set[str] = set()

    for metric in top_bottom_data:
        for group in ["top_10", "bottom_10"]:
            for entry in top_bottom_data[metric][group]:
                ticker = entry["code"]
                if ticker not in added_tickers:
                    clean_ticker = ticker.replace(".JO", "")
                    stock_images.append(
                        RankedStockImage(
                            name=entry["share_name"],
                            ticker=ticker,
                            prediction=encode_image_to_base64(
                                f"plots/{clean_ticker}/close_prediction.png"
                            ),
                            bollinger=encode_image_to_base64(
                                f"plots/{clean_ticker}/adj_bollinger.png"
                            ),
                            overbought_oversold=encode_image_to_base64(
                                f"plots/{clean_ticker}/adj_overbought_oversold.png"
                            ),
                        )
                    )
                    added_tickers.add(ticker)

    return stock_images


def compress_pdf(filename: str) -> str:
    """Compress a PDF's content streams; returns the new ``*_compressed.pdf`` path."""
    logger.debug(f"Compressing PDF report: {filename}")
    reader = PdfReader(filename)
    writer = PdfWriter()

    for page in reader.pages:
        page.compress_content_streams()  # This is where the compression happens
        writer.add_page(page)

    compressed_filename = filename.replace(".pdf", "_compressed.pdf")
    with open(compressed_filename, "wb") as f:
        writer.write(f)

    logger.debug(f"Compressed PDF report created at: {compressed_filename}")

    return compressed_filename


def create_detailed_pdf(
    data: pd.DataFrame,
    stock_images: list[StockChartImages] | list[RankedStockImage],
    filename: str,
    total_value_next_week: float,
    total_value_next_month: float,
    summary_report: bool = False,
    today: str = "",
) -> None:
    """Render the summary or detailed report to HTML and convert it to PDF.

    With ``summary_report=True`` the data is ranked into top/bottom-10 lists
    per metric and rendered with ``summary_template.html``; otherwise the
    full universe is rendered with ``detailed_template.html``.
    """
    logger.debug(f"Creating PDF report: {filename}")
    options = {"page-size": "Letter", "encoding": "UTF-8"}

    env = Environment(loader=FileSystemLoader("."))

    if summary_report:
        logger.debug("Preparing summary report...")
        data["Z_Score"] = pd.to_numeric(data["Z-Score"], errors="coerce").fillna(0)
        data["Current Price"] = data["Current Price"].replace(0, pd.NA).fillna(1e-6)
        # Tickers skipped during fetch/prediction (see the `continue` paths in
        # fetch_data) never get these cells filled, so the column can be object
        # dtype mixing floats with non-numeric sentinels. Coerce before rounding
        # — round() over an object Series hits the string element and raises
        # "type str doesn't define __round__ method". Mirrors the Z-Score guard above.
        data["Next_Week_Prediction_Change"] = (
            pd.to_numeric(data["Next Week Prediction"], errors="coerce").fillna(0) * 100
        ).round(2)
        data["Next_Month_Prediction_Change"] = (
            pd.to_numeric(data["Next Month Prediction"], errors="coerce").fillna(0)
            * 100
        ).round(2)

        metrics = [
            "Z_Score",
            "Next_Week_Prediction_Change",
            "Next_Month_Prediction_Change",
            "Overbought_Oversold_Value",
            "SECTOR RSI 1M",
            "SECTOR RSI 3M",
            "SECTOR RSI 6M",
            "MARKET RSI 1M",
            "MARKET RSI 3M",
            "MARKET RSI 6M",
        ]
        top_bottom_data: TopBottomData = {
            metric: {
                "top_10": data.nlargest(10, metric).to_dict(orient="records"),
                "bottom_10": data.nsmallest(10, metric).to_dict(orient="records"),
            }
            for metric in metrics
        }

        # Prepare stock images based on top/bottom data
        stock_images = prepare_stock_images(top_bottom_data)

        template = env.get_template("summary_template.html")
        rendered = template.render(
            top_bottom_data=top_bottom_data,
            today=today,
            summary=create_summary(data, total_value_next_week, total_value_next_month),
            stock_images=stock_images,
        )

    else:
        template = env.get_template("detailed_template.html")
        rendered = template.render(
            stocks=data.to_dict(orient="records"),
            today=today,
            summary=create_summary(data, total_value_next_week, total_value_next_month),
            stock_images=stock_images,
        )

    # Write the HTML to a file for inspection
    html_file_path = filename.replace(".pdf", ".html")
    with open(html_file_path, "w") as file:
        file.write(rendered)

    # Convert the HTML report to PDF
    pdfkit.from_file(html_file_path, filename, options=options)

    logger.debug(f"PDF report created at: {filename}")


def create_user_detailed_pdf(
    data: pd.DataFrame,
    stock_images: list[StockChartImages],
    filename: str,
    total_value_next_week: float,
    total_value_next_month: float,
    subscriber: Subscribers,
    today: str = "",
) -> None:
    """Render a subscriber-personalised report (web view + optional PDF view).

    The web view always renders from ``web_template.html``. The print view
    renders from ``pdf_template.html`` when that template exists; PDF
    conversion itself is currently disabled.
    """
    logger.debug(f"Creating user: {subscriber.name}'s PDF report: {filename}")

    env = Environment(loader=FileSystemLoader("."))

    template = env.get_template("web_template.html")
    rendered = template.render(
        stocks=data.to_dict(orient="records"),
        today=today,
        summary=create_summary(data, total_value_next_week, total_value_next_month),
        stock_images=stock_images,
        username=subscriber.name,
        id_number=subscriber.id_number,
    )

    # Write the HTML to a file for inspection
    html_file_path = filename.replace(".pdf", ".html")
    with open(html_file_path, "w") as file:
        file.write(rendered)

    # The print-oriented template is optional — without this guard a missing
    # pdf_template.html aborted the whole subscriber loop (and the daily job).
    if os.path.exists("pdf_template.html"):
        template = env.get_template("pdf_template.html")
        rendered = template.render(
            stocks=data.to_dict(orient="records"),
            today=today,
            summary=create_summary(data, total_value_next_week, total_value_next_month),
            stock_images=stock_images,
            username=subscriber.name,
            id_number=subscriber.id_number,
        )

        html_file_path = filename.replace(".pdf", "_pdf.html")
        with open(html_file_path, "w") as file:
            file.write(rendered)

        # Convert the HTML report to PDF
        # pdfkit.from_file(html_file_path, filename, options=options)

        # generate_pdf_with_password(html_file_path, filename, subscriber.id_number)
    else:
        logger.warning(
            "pdf_template.html not found; skipping print view for subscriber report."
        )

    logger.debug(f"User: {subscriber.name}'s PDF report created at: {filename}")


def create_html_summary(
    data: pd.DataFrame,
    total_value_next_week: float,
    total_value_next_month: float,
    template: Template,
) -> str:
    """Render *template* with the stock records and the portfolio summary block."""
    summary = create_summary(data, total_value_next_week, total_value_next_month)
    html_content = template.render(
        stocks=data.to_dict(orient="records"), summary=summary
    )
    return html_content


def create_summary(
    data: pd.DataFrame, total_value_next_week: float, total_value_next_month: float
) -> str:
    """Build the HTML portfolio summary (invested, current and projected value)."""
    try:
        total_invested = data["Initial Purchase Amount"].sum()
    except Exception:
        total_invested = 1

    try:
        current_value = data["Current Value"].sum()
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


def send_email(
    subject: str,
    summary_report_url: str,
    detailed_report_url: str,
    top_bottom_data: TopBottomData,
    sorted_stock_data: list[StockRecord],
    subscriber_urls: dict[str, str],
) -> None:
    """Email the daily report to every active subscriber.

    Each subscriber gets an individually rendered message containing a unique
    tracking URL (stored back on the subscriber record) and a link to their
    personalised detailed report when one exists in *subscriber_urls*.
    """
    try:
        # Load the HTML template
        env = Environment(loader=FileSystemLoader("."))
        template = env.get_template("email_template.html")

        # Fetch subscribers from the database
        subscribers = db_queries.fetch_active_subscribers()
        logger.info(f"Fetched {len(subscribers)} active subscribers.")

        # For each recipient, generate a unique tracking URL
        for subscriber in subscribers:
            try:
                # Generate the email hash for tracking
                email_hash = generate_email_hash(subscriber.email)
                tracking_url = f"https://research.pretoriusse.net/track/{email_hash}"

                # Update the subscriber's email_hash in the database
                subscriber.email_hash = email_hash
                db_queries.update_subscriber(
                    subscriber.id, {"email_hash": email_hash}
                )  # Assuming you have this method
                try:
                    # Render the HTML content with the tracking URL for this specific recipient
                    html_content = template.render(
                        top_bottom_data=top_bottom_data,
                        sorted_stock_data=sorted_stock_data,
                        summary_report_url=summary_report_url,
                        detailed_report_url=subscriber_urls[
                            f"{subscriber.id}_detailed"
                        ],
                        tracking_url=tracking_url,  # Include the tracking URL
                    )
                except KeyError as ex:
                    # Render the HTML content with the tracking URL for this specific recipient
                    html_content = template.render(
                        top_bottom_data=top_bottom_data,
                        sorted_stock_data=sorted_stock_data,
                        summary_report_url=summary_report_url,
                        detailed_report_url="https://research.pretoriusse.net/reports",
                        tracking_url=tracking_url,  # Include the tracking URL
                    )

                # Create a new email message for each recipient
                message = MIMEMultipart()
                message["From"] = formataddr(("Stock Bot", EMAIL_ADDRESS))
                message["To"] = formataddr((subscriber.name, subscriber.email))
                message["Subject"] = subject

                # Attach the HTML content
                message.attach(MIMEText(html_content, "html"))

                # Send the email
                with smtplib.SMTP(SERVER_ADDRESS, SERVER_PORT) as server:
                    server.starttls()
                    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                    server.send_message(message)

                logger.info(f"Email sent successfully to {subscriber.email}")

            except Exception as ex:
                logger.error(f"Failed to send email to {subscriber.email}: {ex}")

    except SQLAlchemyError as db_error:
        logger.error(f"Database error: {db_error}")
    except Exception as ex:
        logger.error(f"Failed to send emails: {ex}")


def daily_job() -> None:
    """Run the complete daily close-price job once.

    Order of operations: ZAR charts → per-ticker Bollinger charts →
    predictions/indicators for the whole universe → RSI benchmark
    comparisons → CSV snapshot + ``close_runs`` upload → summary, detailed
    and per-subscriber reports. Any uncaught error aborts the run (it is
    logged, and cached models/GPU memory are always released).
    """
    while True:
        try:
            start_time = datetime.now()
            execute_time = datetime.today().strftime("%Y-%m-%d %H:%M")
            today = datetime.today().strftime("%Y-%m-%d")

            logger.info(Fore.YELLOW + "Starting close daily job" + Fore.RESET)
            try:
                process_zar_bollinger()
            except Exception as ex:
                logger.error(Fore.RED + f"Error processing ZAR Bollinger:")
                logger.error(ex)
                logger.error(Fore.RESET)

            generate_bollinger_and_overbought_oversold_close()

            for direc in DIRECTORIES:
                os.makedirs(direc, exist_ok=True)

            hparams = ModelHyperParams(
                lstm_units=400,
                dropout=0.3,
                epochs=20 if DEBUGGING else 200,
            )

            fetch_result = fetch_data(hparams)
            stock_data = fetch_result.stocks
            stock_images = fetch_result.images
            total_value_next_week = fetch_result.total_value_next_week
            total_value_next_month = fetch_result.total_value_next_month

            logger.debug(Fore.GREEN + "Data fetched and predictions done." + Fore.RESET)

            stock_data = add_close_rsi_comparisons(stock_data)

            stock_data.to_csv(
                os.path.join("runs", f"{execute_time.replace(':', '')}_close.csv"),
                index=False,
            )
            stock_data.to_csv(
                os.path.join(
                    "data", "runs", f"{execute_time.replace(':', '')}_close.csv"
                ),
                index=False,
            )

            if not DEBUGGING:
                try:
                    # upload_close_runs is the correct helper — the previous
                    # call to a non-existent db_queries.upload_close raised an
                    # AttributeError that was silently swallowed here, so runs
                    # were never uploaded.
                    db_queries.upload_close_runs(
                        os.path.join(
                            "data", "runs", f"{execute_time.replace(':', '')}_close.csv"
                        )
                    )
                    logger.debug(
                        Fore.GREEN + "Uploaded Close runs for today" + Fore.RESET
                    )

                except Exception as ex:
                    logger.error(ex)
                    logger.error(
                        Fore.RED
                        + "Close runs data already exists for today"
                        + Fore.RESET
                    )

            reports_dir = REPORTS_DIR
            os.makedirs(reports_dir, exist_ok=True)

            end_time = datetime.now()
            running_time = end_time - start_time
            minutes = round(running_time.seconds / 60, 2)

            logger.info(
                Fore.MAGENTA + f"\nTime Took:\t{minutes} minutes\n" + Fore.RESET
            )

            os.makedirs(os.path.join(reports_dir, f"{today}"), exist_ok=True)
            attachment_urls: list[str] = []

            # Tickers skipped during fetch/prediction (the `continue` paths in
            # fetch_data) leave their prediction/metric cells unfilled, which can
            # leave these columns as object dtype mixing floats with non-numeric
            # sentinels. Coerce them to numeric before any report code rounds or
            # ranks them (round() and nlargest() both choke on object dtype).
            numeric_value_cols = [
                "Next Week Prediction",
                "Next Month Prediction",
                "Z-Score",
                "Current Price",
                "Current Value",
                "Overbought_Oversold",
                "Overbought_Oversold_Value",
                "Sentiment Score",
                "MA24",
                "MA55",
                "Volume",
            ] + RISK_METRIC_COLS
            for col in numeric_value_cols:
                if col in stock_data.columns:
                    stock_data[col] = pd.to_numeric(
                        stock_data[col], errors="coerce"
                    ).fillna(0)

            # Identify numeric columns in the stock data to round to 2 decimal places.
            numeric_cols = stock_data.select_dtypes(include=["number"]).columns

            # Round only these columns to 2 decimal places
            stock_data[numeric_cols] = stock_data[numeric_cols].round(2)

            if SUMMARY_REPORT:
                summary_pdf_filename = os.path.join(
                    reports_dir, f"{today}", "close_summary.pdf"
                )
                create_detailed_pdf(
                    stock_data,
                    stock_images,
                    summary_pdf_filename,
                    total_value_next_week,
                    total_value_next_month,
                    summary_report=True,
                    today=today,
                )
                # summary_url = upload_to_spaces(summary_pdf_filename, SPACES_KEY, SPACES_SECRET, SPACES_BUCKET, SPACES_REGION, SPACES_URL, today)
                # attachment_urls.append(summary_url)

            detailed_pdf_filename = os.path.join(
                reports_dir, f"{today}", "close_detailed.pdf"
            )
            create_detailed_pdf(
                stock_data,
                stock_images,
                detailed_pdf_filename,
                total_value_next_week,
                total_value_next_month,
                summary_report=False,
                today=today,
            )
            # detailed_url = upload_to_spaces(detailed_pdf_filename, SPACES_KEY, SPACES_SECRET, SPACES_BUCKET, SPACES_REGION, SPACES_URL, today)
            # attachment_urls.append(detailed_url)

            subscribers = db_queries.fetch_active_subscribers()
            subscriber_urls: dict[str, str] = {}
            logger.info(f"Fetched {len(subscribers)} active subscribers.")

            for subscriber in subscribers:
                user_pdf_filename = os.path.join(
                    reports_dir, f"{today}", f"user_{subscriber.id}_close_detailed.pdf"
                )
                create_user_detailed_pdf(
                    stock_data,
                    stock_images,
                    user_pdf_filename,
                    total_value_next_week,
                    total_value_next_month,
                    subscriber=subscriber,
                    today=today,
                )
                # subscriber_urls[f'{subscriber.id}_detailed'] = upload_to_spaces(user_pdf_filename, SPACES_KEY, SPACES_SECRET, SPACES_BUCKET, SPACES_REGION, SPACES_URL, today)

            logger.debug(Fore.GREEN + "PDF created and uploaded" + Fore.RESET)

            # Prepare top and bottom data for the email
            top_bottom_data: TopBottomData = {
                "Z_Score": {
                    "top_10": stock_data.nlargest(10, "Z-Score").to_dict(
                        orient="records"
                    ),
                    "bottom_10": stock_data.nsmallest(10, "Z-Score").to_dict(
                        orient="records"
                    ),
                },
                "Next_Week_Prediction_Change": {
                    "top_10": stock_data.nlargest(10, "Next Week Prediction").to_dict(
                        orient="records"
                    ),
                    "bottom_10": stock_data.nsmallest(
                        10, "Next Week Prediction"
                    ).to_dict(orient="records"),
                },
                "Next_Month_Prediction_Change": {
                    "top_10": stock_data.nlargest(10, "Next Month Prediction").to_dict(
                        orient="records"
                    ),
                    "bottom_10": stock_data.nsmallest(
                        10, "Next Month Prediction"
                    ).to_dict(orient="records"),
                },
                "Overbought_Oversold_Value": {
                    "top_10": stock_data.nlargest(
                        10, "Overbought_Oversold_Value"
                    ).to_dict(orient="records"),
                    "bottom_10": stock_data.nsmallest(
                        10, "Overbought_Oversold_Value"
                    ).to_dict(orient="records"),
                },
                "SECTOR_RSI_1M": {
                    "top_10": stock_data.nlargest(10, "SECTOR RSI 1M").to_dict(
                        orient="records"
                    ),
                    "bottom_10": stock_data.nsmallest(10, "SECTOR RSI 1M").to_dict(
                        orient="records"
                    ),
                },
                "SECTOR_RSI_3M": {
                    "top_10": stock_data.nlargest(10, "SECTOR RSI 3M").to_dict(
                        orient="records"
                    ),
                    "bottom_10": stock_data.nsmallest(10, "SECTOR RSI 3M").to_dict(
                        orient="records"
                    ),
                },
                "SECTOR_RSI_6M": {
                    "top_10": stock_data.nlargest(10, "SECTOR RSI 6M").to_dict(
                        orient="records"
                    ),
                    "bottom_10": stock_data.nsmallest(10, "SECTOR RSI 6M").to_dict(
                        orient="records"
                    ),
                },
                "MARKET_RSI_1M": {
                    "top_10": stock_data.nlargest(10, "MARKET RSI 1M").to_dict(
                        orient="records"
                    ),
                    "bottom_10": stock_data.nsmallest(10, "MARKET RSI 1M").to_dict(
                        orient="records"
                    ),
                },
                "MARKET_RSI_3M": {
                    "top_10": stock_data.nlargest(10, "MARKET RSI 3M").to_dict(
                        orient="records"
                    ),
                    "bottom_10": stock_data.nsmallest(10, "MARKET RSI 3M").to_dict(
                        orient="records"
                    ),
                },
                "MARKET_RSI_6M": {
                    "top_10": stock_data.nlargest(10, "MARKET RSI 6M").to_dict(
                        orient="records"
                    ),
                    "bottom_10": stock_data.nsmallest(10, "MARKET RSI 6M").to_dict(
                        orient="records"
                    ),
                },
                "MA24": {
                    "top_10": stock_data.nlargest(10, "MA24").to_dict(orient="records"),
                    "bottom_10": stock_data.nsmallest(10, "MA24").to_dict(
                        orient="records"
                    ),
                },
                "MA55": {
                    "top_10": stock_data.nlargest(10, "MA55").to_dict(orient="records"),
                    "bottom_10": stock_data.nsmallest(10, "MA55").to_dict(
                        orient="records"
                    ),
                },
            }

            # Get the unique tickers that are in any top 10 or bottom 10 list
            unique_tickers: set[str] = set()
            for key in top_bottom_data:
                unique_tickers.update(
                    [item["code"] for item in top_bottom_data[key]["top_10"]]
                )
                unique_tickers.update(
                    [item["code"] for item in top_bottom_data[key]["bottom_10"]]
                )

            # Filter the stock_data to include only the rows with the unique tickers
            sorted_stock_data = stock_data[stock_data["code"].isin(unique_tickers)]
            summary_url = ""
            detailed_url = ""

            # Send the email with the report links and the filtered data
            """send_email(
                subject=f'Daily Stock Report {today} - Close',
                summary_report_url=summary_url,
                detailed_report_url=detailed_url,
                top_bottom_data=top_bottom_data,
                sorted_stock_data=sorted_stock_data.to_dict(orient='records'),
                subscriber_urls=subscriber_urls
            )
            """
            logger.info("Job completed" + Fore.RESET)
            break

        except Exception as ex:
            logger.error(Fore.RED + "Error occured in Close job.\n" + Fore.RESET)
            logger.error(ex)
            break

        finally:
            # Release all cached Keras models and free GPU memory once the run ends
            _model_cache.clear()
            clear_session()
            gc.collect()


def setup_scheduler() -> None:
    """Block forever, re-running ``daily_job`` every day at 06:08."""
    schedule.every().day.at("06:08").do(daily_job)
    while True:
        schedule.run_pending()
        time.sleep(15)


if __name__ == "__main__":
    daily_job()
    setup_scheduler()
