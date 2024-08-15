
import os
import time
import json
import gc
import threading
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from keras.api import Sequential
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.backend import clear_session # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    from ..assets import database_queries as db_queries  # Importing database queries
except ImportError:
    from assets import database_queries as db_queries
import re
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')



def make_dates_timezone_naive(data):
    try:
        # Convert all datetime objects to timezone-naive
        data['date'] = pd.to_datetime(data['date']).dt.tz_localize(None)
        return data
    except KeyError:
        # Convert all datetime objects to timezone-naive
        data['date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
        return data


def sanitize_ticker(ticker):
    # Replace or remove characters that are not alphanumeric or dot
    sanitized_ticker = re.sub(r'[^A-Za-z0-9.]', '', ticker)
    return sanitized_ticker


# Training function (same as in the original code)
def calculate_accuracy(y_true, y_pred, tolerance=0.05):
    """Calculate the percentage of predictions within a tolerance of the actual values."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.mean(np.abs((y_true - y_pred) / y_true) <= tolerance)
    return accuracy * 100

def train_new_model(X, y, model_dir, model_path, hparams, sanitized_ticker):
    clear_session()
    os.makedirs(model_dir, exist_ok=True)
    if os.path.exists(model_path):
        model: Sequential = load_model(model_path)
    else:
        model: Sequential = Sequential()
    model.add(LSTM(units=hparams['HP_LSTM_UNITS'], return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(hparams['HP_DROPOUT']))
    model.add(LSTM(units=hparams['HP_LSTM_UNITS']))
    model.add(Dropout(hparams['HP_DROPOUT']))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(X, y, epochs=hparams['HP_EPOCHS'], batch_size=64, validation_split=0.1)
    model.save(model_path)

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred, tolerance=0.05)

    model_metadata = {
        'last_trained_date': str(pd.to_datetime('today')),
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        'r2_score': r2,
        'predictions_within_5_percent': accuracy
    }

    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump(model_metadata, f)

    logger.info(f"Model Performance for {sanitized_ticker}: MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    del model
    gc.collect()
    clear_session()

def load_model_metadata(model_dir):
    metadata_path = os.path.join(model_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    
    X = np.array(X)
    y = np.array(y)
    if len(X.shape) == 2:
        X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y

def check_and_train_model(ticker, hparams, seq_length=60):
    logger.info(f"Checking if training is required for {ticker}")

    # Fetch historical data from the database
    starttime_dt = datetime.now() - timedelta(days=4015)
    start_date = starttime_dt.strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    hist = db_queries.get_ticker_from_db_with_date_select(ticker, start_date, end_date)
    hist = make_dates_timezone_naive(hist)
    hist.reset_index(inplace=True)
    hist['date'] = pd.to_datetime(hist['date'])

    scaler = MinMaxScaler()
    hist['close'] = scaler.fit_transform(hist[['close']])
    X, y = create_sequences(hist['close'].values, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sanitized_ticker = sanitize_ticker(ticker)
    model_dir = os.path.join('models', sanitized_ticker)
    model_path = os.path.join(model_dir, f'{sanitized_ticker}_Close_Model.keras')

    last_date_in_data = hist['date'].max()

    model_metadata = load_model_metadata(model_dir)

    if os.path.exists(model_path) and model_metadata and model_metadata.get('last_trained_date'):
        last_trained_date = pd.to_datetime(model_metadata['last_trained_date'])
        
        if last_date_in_data <= last_trained_date:
            logger.info(f"No new data since last training on {last_trained_date}. Skipping training.")
        else:
            logger.info(f"New data available. Retraining model.")
            hist = hist[hist['date'] > last_trained_date]
            X, y = create_sequences(hist['close'].values, seq_length)
            train_new_model(X, y, model_dir, model_path, hparams, sanitized_ticker)
    else:
        logger.info(f"No existing model found. Training a new model.")
        train_new_model(X_train, y_train, model_dir, model_path, hparams, sanitized_ticker)

def run_training_loop(hparams):
    df: pd.DataFrame = db_queries.fetch_stock_and_commodity_universe_from_db()
    

    while True:
        threads = []
        for index, row in df.iterrows():
            if "=" in row['code']:
                continue
            thread = threading.Thread(target=check_and_train_model, args=(row['code'], hparams))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
            
        logger.info("Completed a full training check cycle. Sleeping for 1 hour.")
        time.sleep(1722800)

if __name__ == "__main__":
    hparams = {
        'HP_LSTM_UNITS': 400,
        'HP_DROPOUT': 0.3,
        'HP_EPOCHS': 200
    }
    run_training_loop(hparams)