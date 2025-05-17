import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import json
import gc
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from keras.api import Sequential
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    LSTM, Dense, Dropout, GRU, Bidirectional,
    Conv1D, MaxPooling1D, Flatten, Attention
)
from tensorflow.keras.backend import clear_session # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import re
import schedule
import subprocess
import tensorflow as tf
import matplotlib.pyplot as plt
import ta  # Technical Analysis library
import optuna  # Hyperparameter optimization

# Attempt to import database queries and hyperparameters
try:
    from assets import database_queries as db_queries  # Relative import
except ImportError:
    from ..assets import database_queries as db_queries

try:
    from assets.const import hparams  # Relative import
except ImportError:
    from ..assets.const import hparams

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

# Callback Classes
class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, X, y):
        super(PredictionCallback, self).__init__()
        self.X = X
        self.y = y
        self.predictions = []

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.X)
        self.predictions.append(preds)


class AccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val, tolerance=0.10, target_accuracy=90.0):
        super(AccuracyCallback, self).__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.tolerance = tolerance
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val)
        accuracy = calculate_accuracy(self.y_val, y_pred, tolerance=self.tolerance)
        logger.info(f'Validation Accuracy within {self.tolerance*100}%: {accuracy:.2f}%')
        
        accuracy20 = calculate_accuracy(self.y_val, y_pred, tolerance=0.2)
        logger.info(f'Validation Accuracy within {0.2*100}%: {accuracy20:.2f}%')
        
        accuracy25 = calculate_accuracy(self.y_val, y_pred, tolerance=0.25)
        logger.info(f'Validation Accuracy within {0.25*100}%: {accuracy25:.2f}%')
        
        accuracy50 = calculate_accuracy(self.y_val, y_pred, tolerance=0.5)
        logger.info(f'Validation Accuracy within {0.5*100}%: {accuracy50:.2f}%')

        if accuracy >= self.target_accuracy:
            logger.info(f'Target validation accuracy of {self.target_accuracy}% achieved. Stopping training.')
            self.model.stop_training = True

def make_dates_timezone_naive(data):
    """Removes timezone information from the 'date' column."""
    data['date'] = pd.to_datetime(data['date']).dt.tz_localize(None)
    return data

# Utility Functions
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
    plots_dir = model_dir.replace('models', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    file_path = os.path.join(
        plots_dir,
        f'adjusted_close_training_vs_actual_{hparams["HP_EPOCHS"]}_epochs_'
        f'{hparams["HP_DROPOUT"]}_dropout_{hparams["HP_LSTM_UNITS"]}_units.jpg'
    ).replace(".JO", '')
    logger.info(f"Saving plot to: {file_path}")

    plt.savefig(file_path)
    plt.close()  # Close the plot to free up memory


def plot_residuals(y_true, y_pred, ticker, model_dir):
    """Creates and saves a residuals plot."""
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='red')
    plt.title(f'Residuals for {ticker}')
    plt.xlabel('Predicted Adjusted Close')
    plt.ylabel('Residuals')

    # Construct the file path
    sanitized_ticker = sanitize_ticker(ticker)
    plots_dir = model_dir.replace('models', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    file_path = os.path.join(
        plots_dir,
        f'residuals_{sanitized_ticker}.jpg'
    )
    logger.info(f"Saving residuals plot to: {file_path}")

    plt.savefig(file_path)
    plt.close()


def sanitize_ticker(ticker):
    """Sanitizes the ticker symbol by removing unwanted characters."""
    sanitized_ticker = re.sub(r'[^A-Za-z0-9.]', '', ticker).replace('.JO', '')
    return sanitized_ticker


def calculate_accuracy(y_true, y_pred, tolerance=0.05):
    """
    Calculates the percentage of predictions within the specified tolerance.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        tolerance (float): Tolerance level (e.g., 0.10 for 10%).

    Returns:
        float: Accuracy percentage.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_diff = np.abs((y_true - y_pred) / y_true)
        relative_diff = np.where(np.isfinite(relative_diff), relative_diff, 0)
    accuracy = np.mean(relative_diff <= tolerance)
    return accuracy * 100


def remove_outliers(df, column, threshold=3):
    """Remove outliers based on Z-score."""
    from scipy import stats
    z_scores = stats.zscore(df[column])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < threshold)
    return df[filtered_entries]


def handle_missing_values(df):
    """Handle missing values by forward filling."""
    df = df.ffill()
    df = df.dropna()
    return df


def preprocess_data(hist):
    """Preprocess the historical data."""
    # Remove outliers
    hist = remove_outliers(hist, 'Adj Close')

    # Remove non-positive values
    hist = hist[hist['Adj Close'] > 0]

    # Handle missing values
    hist = handle_missing_values(hist)

    # Ensure 'High', 'Low', 'Adj Close', 'Volume' columns exist and are correctly named
    required_columns = {'High', 'Low', 'Adj Close', 'Volume'}
    missing_columns = required_columns - set(hist.columns)
    if missing_columns:
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    # Feature Engineering
    hist['MA50'] = ta.trend.sma_indicator(hist['Adj Close'], window=50)
    hist['MA200'] = ta.trend.sma_indicator(hist['Adj Close'], window=200)
    hist['RSI'] = ta.momentum.rsi(hist['Adj Close'], window=14)
    hist['MACD'] = ta.trend.macd(hist['Adj Close'])
    hist['MACD_DIFF'] = ta.trend.macd_diff(hist['Adj Close'])
    hist['Bollinger_High'] = ta.volatility.bollinger_hband(hist['Adj Close'], window=20)
    hist['Bollinger_Low'] = ta.volatility.bollinger_lband(hist['Adj Close'], window=20)
    hist['ATR'] = ta.volatility.average_true_range(hist['High'], hist['Low'], hist['Adj Close'], window=14)
    hist['OBV'] = ta.volume.on_balance_volume(hist['Adj Close'], hist['Volume'])

    # Corrected Stochastic Oscillator calculation using the StochasticOscillator class
    stochastic = ta.momentum.StochasticOscillator(
        high=hist['High'],
        low=hist['Low'],
        close=hist['Adj Close'],
        window=14,
        smooth_window=3
    )
    hist['Stochastic_K'] = stochastic.stoch()
    hist['Stochastic_D'] = stochastic.stoch_signal()

    hist['day_of_week'] = hist['date'].dt.dayofweek
    hist['month'] = hist['date'].dt.month
    hist['quarter'] = hist['date'].dt.quarter

    # Lag Features
    for lag in range(1, 4):
        hist[f'lag_{lag}'] = hist['Adj Close'].shift(lag)

    # Rolling Statistics
    hist['rolling_mean_5'] = hist['Adj Close'].rolling(window=5).mean()
    hist['rolling_std_5'] = hist['Adj Close'].rolling(window=5).std()
    hist['rolling_mean_10'] = hist['Adj Close'].rolling(window=10).mean()
    hist['rolling_std_10'] = hist['Adj Close'].rolling(window=10).std()

    # Drop NaNs created by indicators and lag features
    hist = hist.dropna()

    return hist


def build_bidirectional_gru_model(input_shape, hparams):
    """Builds a Bidirectional GRU model with regularization."""
    model = Sequential()
    model.add(Bidirectional(
        GRU(units=hparams['HP_GRU_UNITS'], return_sequences=True,
            kernel_regularizer=l2(0.001)), input_shape=input_shape))
    model.add(Dropout(hparams['HP_DROPOUT']))
    model.add(Bidirectional(
        GRU(units=hparams['HP_GRU_UNITS'], kernel_regularizer=l2(0.001))))
    model.add(Dropout(hparams['HP_DROPOUT']))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def build_cnn_lstm_model(input_shape, hparams):
    """Builds a CNN-LSTM hybrid model."""
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=hparams['HP_LSTM_UNITS'], return_sequences=True))
    model.add(Dropout(hparams['HP_DROPOUT']))
    model.add(LSTM(units=hparams['HP_LSTM_UNITS']))
    model.add(Dropout(hparams['HP_DROPOUT']))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def build_model(input_shape, hparams):
    """Builds the LSTM model with regularization."""
    model = Sequential()
    model.add(LSTM(units=hparams['HP_LSTM_UNITS'], return_sequences=True, kernel_regularizer=l2(0.001), input_shape=input_shape))
    model.add(Dropout(hparams['HP_DROPOUT']))
    model.add(LSTM(units=hparams['HP_LSTM_UNITS'], kernel_regularizer=l2(0.001)))
    model.add(Dropout(hparams['HP_DROPOUT']))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def load_model_metadata(model_dir):
    """Loads model metadata from a JSON file."""
    metadata_path = os.path.join(model_dir, 'adjusted_close_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def create_sequences(data, seq_length):
    """Creates sequences of data for time series forecasting."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length][0])  # Assuming 'Adj Close' is the first feature

    X = np.array(X)
    y = np.array(y)

    # Debugging: Check if sequences are being created correctly
    if len(X.shape) == 2:
        logger.warning(f"Data for sequences has only 2 dimensions: {X.shape}. Reshaping to add the third dimension.")
        X = X.reshape((X.shape[0], X.shape[1], 1))

    logger.info(f"Sequences created with shape: X={X.shape}, y={y.shape}")

    return X, y


def load_and_scale_data(hist):
    """Loads and scales the data."""
    scaler = MinMaxScaler()
    hist_scaled = scaler.fit_transform(hist)
    return hist_scaled, scaler


# Hyperparameter Optimization Function
def optimize_hyperparameters(ticker, hparams, X_train, y_train, X_val, y_val):
    """Optimize hyperparameters using Optuna for each ticker."""

    def objective(trial):
        # Suggest hyperparameters
        hp_lstm_units = trial.suggest_int('HP_LSTM_UNITS', 50, 300)
        hp_gru_units = trial.suggest_int('HP_GRU_UNITS', 50, 300)
        hp_dropout = trial.suggest_float('HP_DROPOUT', 0.1, 0.5)
        hp_epochs = trial.suggest_int('HP_EPOCHS', 100, 800)
        hp_batch_size = trial.suggest_categorical('HP_BATCH_SIZE', [64, 128, 256])

        # Update hyperparameters for this trial
        current_hparams = hparams.copy()
        current_hparams['HP_LSTM_UNITS'] = hp_lstm_units
        current_hparams['HP_GRU_UNITS'] = hp_gru_units
        current_hparams['HP_DROPOUT'] = hp_dropout
        current_hparams['HP_EPOCHS'] = hp_epochs
        current_hparams['HP_BATCH_SIZE'] = hp_batch_size

        # Build and train the model
        model = build_bidirectional_gru_model((X_train.shape[1], X_train.shape[2]), current_hparams)

        # Initialize callbacks
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        accuracy_callback = AccuracyCallback(
            X_val, y_val, tolerance=0.10, target_accuracy=90.0)

        history = model.fit(
            X_train, y_train,
            epochs=current_hparams['HP_EPOCHS'],
            batch_size=current_hparams['HP_BATCH_SIZE'],
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr, accuracy_callback],
            verbose=0  # Suppress training logs for Optuna
        )

        # Evaluate the model
        y_pred = model.predict(X_val)
        accuracy10 = calculate_accuracy(y_val, y_pred, tolerance=0.10)

        # Clear GPU memory
        del model
        clear_gpu_memory()

        return 100 - accuracy10  # Optuna minimizes the objective

    # Create Optuna study and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial), n_trials=hparams.get('OPTUNA_TRIALS', 50))

    best_params = study.best_params
    logger.info(f"Best hyperparameters found for {ticker}: {best_params}")

    # Save best hyperparameters in JSON for future reference
    sanitized_ticker = sanitize_ticker(ticker)
    model_dir = os.path.join('models', sanitized_ticker)
    os.makedirs(model_dir, exist_ok=True)
    hyperparams_path = os.path.join(model_dir, 'best_hyperparameters.json')

    try:
        with open(hyperparams_path, mode='w+') as f:
            json.dump(best_params, f)
        logger.info(f"Best hyperparameters saved for {ticker} at {hyperparams_path}")
    except Exception as e:
        logger.error(f"Error saving best hyperparameters for {ticker}: {e}")

    return best_params


def train_new_model(X_train, y_train, X_val, y_val, model_dir, model_path, hparams, sanitized_ticker):
    """Trains a new model and saves it along with metadata and plots."""
    logger.info("Started training for %s", sanitized_ticker)
    print(f"Started training for {sanitized_ticker}")
    clear_gpu_memory()  # Clear GPU memory before starting new training
    os.makedirs(model_dir, exist_ok=True)

    # Choose model architecture (GRU, LSTM, or CNN-LSTM)
    model = build_bidirectional_gru_model((X_train.shape[1], X_train.shape[2]), hparams)

    # Initialize callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=hparams.get('PATIENCE', 20), restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    accuracy_callback = AccuracyCallback(
        X_val, y_val, tolerance=0.10, target_accuracy=hparams.get('TARGET_ACCURACY', 90.0))

    history = model.fit(
        X_train, y_train,
        epochs=hparams['HP_EPOCHS'],
        batch_size=hparams.get('HP_BATCH_SIZE', 128),
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr, accuracy_callback]
    )

    logger.info(f"Saving model to: {model_path}")
    try:
        model.save(model_path)
        if os.path.exists(model_path):
            logger.info(f"Model saved successfully at {model_path}")
        else:
            logger.error(f"Model save failed for {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

    # Plot actual vs predicted for the final predictions
    y_pred = model.predict(X_val)
    plot_actual_vs_predicted(y_val, y_pred, sanitized_ticker, model_dir)
    plot_residuals(y_val, y_pred, sanitized_ticker, model_dir)

    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    accuracy = calculate_accuracy(y_val, y_pred, tolerance=0.05)
    accuracy10 = calculate_accuracy(y_val, y_pred, tolerance=0.10)
    accuracy15 = calculate_accuracy(y_val, y_pred, tolerance=0.15)
    accuracy20 = calculate_accuracy(y_val, y_pred, tolerance=0.20)

    model_metadata = {
        'last_trained_date': str(pd.to_datetime('today')),
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        'r2_score': r2,
        'predictions_within_5_percent': accuracy,
        'predictions_within_10_percent': accuracy10,
        'predictions_within_15_percent': accuracy15,
        'predictions_within_20_percent': accuracy20,
        'hyperparameters': hparams  # Save the hyperparameters used
    }

    # Save metadata JSON with "w+" mode to create the file if it doesn't exist
    metadata_path = os.path.join(model_dir, 'adjusted_close_metadata.json')
    logger.info(f"Saving model metadata to {metadata_path}")
    try:
        with open(metadata_path, mode='w+') as f:
            json.dump(model_metadata, f)
        logger.info(f"Model metadata saved successfully at {metadata_path}")
        
    except Exception as e:
        logger.error(f"Error saving model metadata: {e}")

    logger.info(f"Model Performance for {sanitized_ticker}: MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    del model
    clear_gpu_memory()


def check_and_train_model(ticker, hparams, seq_length=60):
    """Checks if training is required for a ticker and trains the model if needed."""
    logger.info(f"Checking if training is required for {ticker}")

    # Fetch historical data
    starttime_dt = datetime.now() - timedelta(days=2555)  # Approximately 7 years
    start_date = starttime_dt.strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    hist = db_queries.get_ticker_from_db_with_date_select(ticker, start_date, end_date)

    hist = hist.rename(columns={'low': 'Low', 'high': 'High', 'adj_close': 'Adj Close', 'volume': 'Volume'})

    if hist.empty:
        logger.info(f"No data found for {ticker} after fetching from DB.")
        return

    hist = make_dates_timezone_naive(hist)
    hist.reset_index(inplace=True)
    hist['date'] = pd.to_datetime(hist['date'])

    logger.info(f"Fetched {len(hist)} rows of data for {ticker} from {start_date} to {end_date}.")

    # Preprocess data
    hist = preprocess_data(hist)

    # Scale features
    features = ['Adj Close', 'MA50', 'MA200', 'RSI',
                'MACD', 'MACD_DIFF', 'Bollinger_High', 'Bollinger_Low',
                'ATR', 'OBV', 'Stochastic_K', 'Stochastic_D',
                'day_of_week', 'month', 'quarter',
                'lag_1', 'lag_2', 'lag_3',
                'rolling_mean_5', 'rolling_std_5',
                'rolling_mean_10', 'rolling_std_10']
    hist_scaled, scaler = load_and_scale_data(hist[features])

    if len(hist_scaled) < seq_length:
        logger.error(f"Not enough data to create sequences for {ticker}.")
        return

    X, y = create_sequences(hist_scaled, seq_length)

    if X.shape[0] == 0 or y.shape[0] == 0:
        logger.error(f"Not enough data to create sequences for {ticker}. Skipping training.")
        return

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

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
            logger.info(f"New data available since last training on {last_trained_date}. Retraining model.")

            # Load existing model
            try:
                model = load_model(model_path)
                logger.info(f"Loaded existing model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading existing model: {e}")
                return

            # Fine-tune existing model
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=20, restore_best_weights=True)
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
            accuracy_callback = AccuracyCallback(
                X_val, y_val, tolerance=0.10, target_accuracy=90.0)

            history = model.fit(
                X_train, y_train,
                epochs=10,  # Fine-tuning epochs
                batch_size=hparams.get('HP_BATCH_SIZE', 128),
                validation_data=(X_val, y_val),
                callbacks=[early_stop, reduce_lr, accuracy_callback]
            )

            # Evaluate the model on the validation set
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            accuracy = calculate_accuracy(y_val, y_pred, tolerance=0.05)
            accuracy10 = calculate_accuracy(y_val, y_pred, tolerance=0.10)
            accuracy15 = calculate_accuracy(y_val, y_pred, tolerance=0.15)
            accuracy20 = calculate_accuracy(y_val, y_pred, tolerance=0.20)

            # Save the retrained model
            try:
                model.save(model_path)
                logger.info(f"Model retrained and saved successfully at {model_path}")
            except Exception as e:
                logger.error(f"Error saving retrained model: {e}")
                return

            # Update and save model metadata
            model_metadata.update({
                'last_trained_date': str(pd.to_datetime('today')),
                'mean_squared_error': mse,
                'mean_absolute_error': mae,
                'r2_score': r2,
                'predictions_within_5_percent': accuracy,
                'predictions_within_10_percent': accuracy10,
                'predictions_within_15_percent': accuracy15,
                'predictions_within_20_percent': accuracy20
            })

            metadata_path = os.path.join(model_dir, 'adjusted_close_metadata.json')
            logger.info(f"Saving updated model metadata to {metadata_path}")
            try:
                with open(metadata_path, mode='w+') as f:
                    json.dump(model_metadata, f)

                logger.info(f"Model metadata updated successfully at {metadata_path}")
            except Exception as e:
                logger.error(f"Error saving updated model metadata: {e}")

            # Plot actual vs predicted and residuals
            plot_actual_vs_predicted(y_val, y_pred, sanitized_ticker, model_dir)
            plot_residuals(y_val, y_pred, sanitized_ticker, model_dir)

            logger.info(f"Model retrained for {sanitized_ticker}. MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            del model
            clear_gpu_memory()
    else:
        logger.info(f"No existing model found for {ticker}. Training a new model.")

        # Optimize hyperparameters using Optuna
        if not os.path.exists(os.path.join(model_dir, 'best_hyperparameters.json')):
            best_hparams = optimize_hyperparameters(ticker, hparams, X_train, y_train, X_val, y_val)
            hparams.update(best_hparams)
        
        else:
            with open(file=os.path.join(model_dir, 'best_hyperparameters.json'), mode='r') as hparam_file:
                hparams.update(json.load(hparam_file))

        # Train the model with optimized hyperparameters
        train_new_model(X_train, y_train, X_val, y_val, model_dir, model_path, hparams, sanitized_ticker)



def run_training_loop(hparams):
    """Runs the training loop for all tickers in the stock universe."""
    df: pd.DataFrame = db_queries.fetch_stock_universe_from_db()

    for index, row in df.iterrows():
        # Skip certain tickers based on specific conditions
        if "=" in row['code'] or row['commodity'] or row['code'] in ['%5EJ300.JO', 'STXFIN.JO', 'STXCAP.JO', 'STXRES.JO']:
            continue
        sanitized_ticker = sanitize_ticker(row['code'])
        model_dir = os.path.join('models', sanitized_ticker)
        try:
            check_and_train_model(row['code'], hparams)
        except Exception as e:
            logger.error(f"Error occurred for {row['code']}: {e}")
            continue

    logger.info("Completed a full training check cycle.")


def job():
    """Scheduled job to run the training loop."""
    try:
        run_training_loop(hparams)
    except KeyboardInterrupt:
        clear_gpu_memory()


def clear_gpu_memory():
    """Clears GPU memory to prevent memory overflow errors."""
    clear_session()  # Clear TensorFlow session
    tf.keras.backend.clear_session()  # Clear Keras session
    tf.compat.v1.reset_default_graph()  # Reset TensorFlow graph
    gc.collect()  # Trigger garbage collection


# Main Execution
if __name__ == "__main__":
    # Initial training run
    job()

    # Schedule the job to run daily at 18:30
    schedule.every().day.at("18:30").do(job)

    logger.info("Training scheduler started. Waiting for scheduled jobs...")

    while True:
        schedule.run_pending()
        time.sleep(15)  # Wait for the next scheduled task
