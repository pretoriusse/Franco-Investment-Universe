import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import time
import json
import gc
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from keras.api import Sequential
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.backend import clear_session  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import re
import schedule  # Import the schedule library
import subprocess
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

try:
    from ..assets import database_queries as db_queries  # Importing database queries
except ImportError:
    from assets import database_queries as db_queries

try:
    from ..assets.const import hparams  # Importing hparams
except ImportError:
    from assets.const import hparams

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set TensorFlow to use GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs.")
    except RuntimeError as e:
        logger.error(f"Error setting GPU memory growth: {e}")


class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, X, y):
        super(PredictionCallback, self).__init__()
        self.X = X
        self.y = y
        self.predictions = []

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.X)
        self.predictions.append(preds)


def plot_actual_vs_predicted(y_true, y_pred, ticker, model_dir):
    """Creates and saves a plot comparing actual vs predicted values."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='orange')
    plt.title(f'Actual vs Predicted Adjusted Close for {ticker}')
    plt.xlabel('Time Steps')
    plt.ylabel('Adjusted Close Price')
    plt.legend()

    # Construct the file path
    sanitized_ticker = sanitize_ticker(ticker)
    file_path = os.path.join(model_dir.replace('models', 'plots'), f'adjusted_close_training_vs_actual_{hparams["HP_EPOCHS"]}_epochs_{hparams["HP_DROPOUT"]}_dropout_{hparams["HP_LSTM_UNITS"]}_units.jpg').replace(".JO", '')
    print(file_path)

    plt.savefig(file_path)
    plt.close()  # Close the plot to free up memory


def pull_and_merge_trained_models(branch_name="main"):
    try:
        # Step 1: Stash any local changes
        subprocess.run(["git", "stash"], check=True)

        # Step 2: Pull the latest changes from the remote branch
        subprocess.run(["git", "pull", "origin", branch_name], check=True)

        # Step 3: Merge the trained models
        model_dir = "models/"
        subprocess.run(["git", "add", model_dir], check=True)
        today = datetime.now().strftime("%d/%m/%Y")
        commit_message = f"Merge of trained models {today}"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        # Step 4: Push the changes back to the remote branch
        subprocess.run(["git", "push", "origin", branch_name], check=True)

        # Step 5: Apply the stashed changes back
        subprocess.run(["git", "stash", "pop"], check=True)

        print("Successfully pulled, merged, and pushed trained models.")

    except subprocess.CalledProcessError as e:
        print(f"Error during git operation: {e}")


def make_dates_timezone_naive(data):
    data['date'] = pd.to_datetime(data['date']).dt.tz_localize(None)
    return data


def sanitize_ticker(ticker):
    sanitized_ticker = re.sub(r'[^A-Za-z0-9.]', '', ticker)
    return sanitized_ticker


def calculate_accuracy(y_true, y_pred, tolerance=0.05):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.mean(np.abs((y_true - y_pred) / y_true) <= tolerance)
    return accuracy * 100


def clear_gpu_memory():
    """Clears GPU memory to prevent memory overflow errors."""
    clear_session()  # Clear TensorFlow session
    tf.keras.backend.clear_session()  # Clear Keras session
    tf.compat.v1.reset_default_graph()  # Reset TensorFlow graph
    gc.collect()  # Trigger garbage collection


# Modify the train_new_model function to include plotting
def train_new_model(X, y, model_dir, model_path, hparams, sanitized_ticker):
    logger.info("Started training for %s", sanitized_ticker)
    print("Started training for %s", sanitized_ticker)
    clear_gpu_memory()  # Clear GPU memory before starting new training
    os.makedirs(model_dir, exist_ok=True)

    model = Sequential()
    model.add(LSTM(units=hparams['HP_LSTM_UNITS'], return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(hparams['HP_DROPOUT']))
    model.add(LSTM(units=hparams['HP_LSTM_UNITS']))
    model.add(Dropout(hparams['HP_DROPOUT']))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Initialize callback
    prediction_callback = PredictionCallback(X, y)

    history = model.fit(X, y, epochs=hparams['HP_EPOCHS'], batch_size=128, validation_split=0.05, callbacks=[prediction_callback])
    
    if os.path.exists(model_path):
        os.remove(model_path)

    model.save(model_path)

    # Plot actual vs predicted for the final predictions
    y_pred = model.predict(X)
    plot_actual_vs_predicted(y, y_pred, sanitized_ticker, model_dir)

    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred, tolerance=0.05)
    accuracy10 = calculate_accuracy(y, y_pred, tolerance=0.10)
    accuracy15 = calculate_accuracy(y, y_pred, tolerance=0.15)
    accuracy20 = calculate_accuracy(y, y_pred, tolerance=0.20)

    model_metadata = {
        'last_trained_date': str(pd.to_datetime('today')),
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        'r2_score': r2,
        'predictions_within_5_percent': accuracy,
        'predictions_within_10_percent': accuracy10,
        'predictions_within_15_percent': accuracy15,
        'predictions_within_20_percent': accuracy20
    }

    # Ensure the model directory exists before saving the metadata
    os.makedirs(model_dir, exist_ok=True)

    # Save metadata JSON
    with open(os.path.join(model_dir, 'adjusted_close_metadata.json'), 'w+') as f:
        json.dump(model_metadata, f)

    logger.info(f"Model Performance for {sanitized_ticker}: MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    del model
    clear_gpu_memory()


def load_model_metadata(model_dir):
    metadata_path = os.path.join(model_dir, 'adjusted_close_metadata.json')
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
    
    # Debugging: Check if sequences are being created correctly
    if len(X.shape) == 2:
        logger.warning(f"Data for sequences has only 2 dimensions: {X.shape}. Reshaping to add the third dimension.")
        X = X.reshape((X.shape[0], X.shape[1], 1))

    logger.info(f"Sequences created with shape: X={X.shape}, y={y.shape}")
    
    return X, y


def check_and_train_model(ticker, hparams, seq_length=60):
    logger.info(f"Checking if training is required for {ticker}")

    starttime_dt = datetime.now() - timedelta(days=2555)
    start_date = starttime_dt.strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    hist = db_queries.get_ticker_from_db_with_date_select(ticker, start_date, end_date)
    hist = make_dates_timezone_naive(hist)
    hist.reset_index(inplace=True)
    hist['date'] = pd.to_datetime(hist['date'])
    
    logger.info(f"Fetched {len(hist)} rows of data for {ticker} from {start_date} to {end_date}.")

    scaler = MinMaxScaler()
    hist['Adj Close'] = scaler.fit_transform(hist[['Adj Close']])
    
    if len(hist) < seq_length:
        logger.error(f"Not enough data to create sequences for {ticker}.")
        return

    X, y = create_sequences(hist['Adj Close'].values, seq_length)

    if X.shape[0] == 0 or y.shape[0] == 0:
        logger.error(f"Not enough data to create sequences for {ticker}. Skipping training.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sanitized_ticker = sanitize_ticker(ticker)
    model_dir = os.path.join('models', sanitized_ticker)
    model_path = os.path.join(model_dir, f'{sanitized_ticker}_Adjusted_Close_Model.keras')

    last_date_in_data = hist['date'].max()

    model_metadata = load_model_metadata(model_dir)

    if os.path.exists(model_path) and model_metadata and model_metadata.get('last_trained_date'):
        last_trained_date = pd.to_datetime(model_metadata['last_trained_date'])
        
        if last_date_in_data <= last_trained_date:
            logger.info(f"No new data since last training on {last_trained_date}. Skipping training.")
        else:
            logger.info(f"New data available. Retraining model with existing and new data.")

            # Load existing model
            model = load_model(model_path)
            
            # Retrain with the existing and new data
            model.fit(X_train, y_train, epochs=hparams['HP_EPOCHS'], batch_size=128, validation_split=0.05)

            # Evaluate the model on the test set
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            accuracy = calculate_accuracy(y_test, y_pred, tolerance=0.05)
            accuracy10 = calculate_accuracy(y_test, y_pred, tolerance=0.10)
            accuracy15 = calculate_accuracy(y_test, y_pred, tolerance=0.15)
            accuracy20 = calculate_accuracy(y_test, y_pred, tolerance=0.20)

            # Save the retrained model
            if os.path.exists(model_path):
                os.remove(model_path)
            model.save(model_path)

            # Update and save model metadata
            model_metadata = {
                'last_trained_date': str(pd.to_datetime('today')),
                'mean_squared_error': mse,
                'mean_absolute_error': mae,
                'r2_score': r2,
                'predictions_within_5_percent': accuracy,
                'predictions_within_10_percent': accuracy10,
                'predictions_within_15_percent': accuracy15,
                'predictions_within_20_percent': accuracy20
            }

            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, 'adjusted_close_metadata.json'), 'w+') as f:
                json.dump(model_metadata, f)
                
            with open(os.path.join(model_dir, 'adjusted_close_metadata.json'), 'w+') as f:
                json.dump(model_metadata, f)

            logger.info(f"Model retrained and saved for {sanitized_ticker}. MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            del model
            clear_gpu_memory()  # Clear GPU memory after retraining
    else:
        logger.info(f"No existing model found. Training a new model.")
        train_new_model(X_train, y_train, model_dir, model_path, hparams, sanitized_ticker)


def run_training_loop(hparams):
    df: pd.DataFrame = db_queries.fetch_stock_universe_from_db()

    for index, row in df.iterrows():
        if "=" in row['code'] or row['commodity'] or row['code'] in ['%5EJ300.JO', 'STXFIN.JO', 'STXCAP.JO', 'STXRES.JO']:
            continue
        sanitized_ticker = sanitize_ticker(row['code'])
        model_dir = os.path.join('models', sanitized_ticker)
        model_path = os.path.join(model_dir, f'{sanitized_ticker}_Adjusted_Close_Model.keras')
        try:
            check_and_train_model(row['code'], hparams)
        except Exception as e:
            logger.error(f"Error occurred for {row['code']}: {e}")
            continue
        
    logger.info("Completed a full training check cycle.")


def job():
    try:
        run_training_loop(hparams)
    except KeyboardInterrupt:
        clear_gpu_memory()


if __name__ == "__main__":
    job()
    schedule.every().day.at("18:30").do(job)
    
    while True:
        schedule.run_pending()
        time.sleep(15)  # Wait for the next scheduled task
