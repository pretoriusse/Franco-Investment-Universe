import itertools
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)  # Change to DEBUG for detailed logs
logger = logging.getLogger(__name__)

from .const import DB_PARAMS_EXCELDATA as DB_PARAMS

# Create the database engine
engine = create_engine(
    f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@"
    f"{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
)

def fetch_data():
    query = "SELECT * FROM public.aipcharts_chart_data;"
    try:
        df = pd.read_sql_query(query, engine)
        logger.info("Data fetched successfully from the database.")
        return df
    except Exception as e:
        logger.error(f"An error occurred while fetching data: {e}")
        return None

def process_data(df):
    try:
        # Convert 'date' column to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values(by=['ticker', 'date'])
        df = df[df['date'] > '2015-06-30']
        logger.info(f"Data after date filtering: {df.shape[0]} rows, {df.shape[1]} columns.")
        
        # Log the min and max dates
        min_date = df['date'].min()
        max_date = df['date'].max()
        logger.info(f"Date range in data: {min_date.date()} to {max_date.date()}")

        # Handle missing and zero values for specified features
        required_columns = ['P_Sales_ps', 'P_Ebitda_ps', 'P_Op_profit_ps', 'P_EPS',
                            'P_FCF_ps', 'P_DPS', 'P_NAV', 'P_NTAV']
        df[required_columns] = df[required_columns].replace(0, np.nan)
        imputer = SimpleImputer(strategy='median')
        df[required_columns] = imputer.fit_transform(df[required_columns])
        logger.info(f"Data after handling zeros and imputing missing values: {df.shape[0]} rows, {df.shape[1]} columns.")

        # Encode 'ticker' if needed
        if 'ticker' in df.columns:
            le = LabelEncoder()
            df['ticker_encoded'] = le.fit_transform(df['ticker'])
        else:
            logger.error("The 'ticker' column is missing from the data.")
            return None

        # Compute price movement
        df['price_movement'] = df.groupby('ticker')['adj_close'].pct_change()
        df = df.dropna(subset=['price_movement'])
        logger.info(f"Data after computing price movement: {df.shape[0]} rows, {df.shape[1]} columns.")

        return df
    except Exception as e:
        logger.error(f"An error occurred during data processing: {e}")
        return None

def define_time_windows(df, window_years=5, end_gap_years=2, step_months=1):
    """
    Define sliding time windows.

    Args:
        df (pd.DataFrame): The dataframe with a 'date' column.
        window_years (int): Length of each window in years.
        end_gap_years (int): Minimum gap from today to end the window.
        step_months (int): Number of months to slide the window each step.

    Returns:
        List of tuples: Each tuple contains (start_date, end_date) for a window.
    """
    end_date_limit = datetime.now() - relativedelta(years=end_gap_years)
    window_length = relativedelta(years=window_years)
    step = relativedelta(months=step_months)
    
    # Latest possible start date
    latest_start = end_date_limit - window_length
    earliest_start = df['date'].min()

    windows = []
    current_start = latest_start

    while current_start >= earliest_start:
        current_end = current_start + window_length
        if current_end <= end_date_limit:
            windows.append((current_start, current_end))
            logger.debug(f"Window added: {current_start.date()} to {current_end.date()}")
        else:
            logger.debug(f"Window skipped (end date beyond limit): {current_start.date()} to {current_end.date()}")
        current_start -= step

    logger.info(f"Defined {len(windows)} sliding time windows.")
    return windows

def evaluate_feature_pairs_window(df, features, y_column):
    best_r2 = -np.inf
    best_pair = None
    best_model = None
    best_coefficients = None

    # Generate all possible pairs of features
    feature_pairs = list(itertools.combinations(features, 2))
    logger.info(f"Evaluating {len(feature_pairs)} feature pairs.")

    for pair in feature_pairs:
        X = df[list(pair)].values
        y = df[y_column].values

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit linear regression
        reg = LinearRegression()
        reg.fit(X_scaled, y)
        y_pred = reg.predict(X_scaled)

        # Calculate R-squared
        r2 = r2_score(y, y_pred)

        # Update best pair if necessary
        if r2 > best_r2:
            best_r2 = r2
            best_pair = pair
            best_model = reg
            best_coefficients = reg.coef_

    return best_pair, best_r2, best_model, best_coefficients

def plot_best_pair(df, best_pair, best_model, y_column, plot_dir, window=None):
    try:
        X = df[list(best_pair)].values
        y = df[y_column].values

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Predict
        y_pred = best_model.predict(X_scaled)

        # Calculate coefficients in original scale
        coef = best_model.coef_
        intercept = best_model.intercept_
        # To get coefficients in original scale:
        coef_original = coef / scaler.scale_
        intercept_original = intercept - np.sum((coef * scaler.mean_) / scaler.scale_)

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y, y=y_pred, alpha=0.5, color='red', label='Predictions')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], color='black', linewidth=2, label='Ideal Fit')

        plt.title(f'Actual vs Predicted Price Movement using {best_pair}'
                  + (f' ({window[0].date()} - {window[1].date()})' if window else ''))
        plt.xlabel('Actual Price Movement (%)')
        plt.ylabel('Predicted Price Movement (%)')
        plt.legend()
        plt.grid(True)

        # Save the plot
        save_path = os.path.join(plot_dir, 'Best_Pair_Regression_Plot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Best pair regression plot saved successfully at {save_path}.")

    except Exception as e:
        logger.error(f"An error occurred while plotting the best pair regression: {e}")

def main():
    # Fetch and process data
    df = fetch_data()
    if df is None:
        return

    df = process_data(df)
    if df is None:
        return

    # Define features and target
    features = ['P_Sales_ps', 'P_Ebitda_ps', 'P_Op_profit_ps', 'P_EPS',
                'P_FCF_ps', 'P_DPS', 'P_NAV', 'P_NTAV']
    y_column = 'price_movement'

    # Define time windows
    windows = define_time_windows(df, window_years=5, end_gap_years=2, step_months=1)

    if not windows:
        logger.warning("No time windows defined with the current parameters.")
        # Optionally, use the entire dataset as a single window
        windows = [(df['date'].min(), df['date'].max())]
        logger.info("Using the entire dataset as a single window.")

    qualifying_windows = []

    for window_start, window_end in windows:
        # Filter data for the current window
        window_df = df[(df['date'] >= window_start) & (df['date'] <= window_end)]
        logger.info(f"Evaluating window: {window_start.date()} to {window_end.date()} with {window_df.shape[0]} records.")

        if window_df.empty:
            logger.warning(f"No data in window: {window_start.date()} to {window_end.date()}. Skipping.")
            continue

        # Evaluate feature pairs in the current window
        best_pair, best_r2, best_model, best_coefficients = evaluate_feature_pairs_window(window_df, features, y_column)
        
        logger.info(f"Window {window_start.date()} - {window_end.date()}: Best R2 = {best_r2:.4f} with pair {best_pair}")

        # Check if the best R2 meets the threshold
        if best_r2 >= 0.75:
            qualifying_windows.append({
                'window_start': window_start,
                'window_end': window_end,
                'best_pair': best_pair,
                'best_r2': best_r2,
                'best_coefficients': best_coefficients,
                'best_model': best_model
            })
            # Uncomment the following line to stop after finding the first qualifying window
            # break

    if qualifying_windows:
        for idx, qw in enumerate(qualifying_windows, 1):
            print(f"\nQualifying Window {idx}:")
            print(f"Feature Pair: {qw['best_pair']} with R-squared: {qw['best_r2']:.4f}")
            print(f"Time Period: {qw['window_start'].date()} to {qw['window_end'].date()}")
            print(f"Coefficients: {qw['best_coefficients']}")

            # Define plot directory
            ticker = df['ticker'].iloc[0].replace('.JO', '') if 'ticker' in df.columns else 'Unknown'
            plot_dir = os.path.join('plots', f'Qualifying_Window_{idx}', ticker, 'metrics')
            os.makedirs(plot_dir, exist_ok=True)

            # Plot the best pair regression
            plot_best_pair(
                df[(df['date'] >= qw['window_start']) & (df['date'] <= qw['window_end'])],
                qw['best_pair'],
                qw['best_model'],
                y_column,
                plot_dir,
                window=(qw['window_start'], qw['window_end'])
            )
    else:
        logger.error("No qualifying feature pairs found across all time windows with R-squared >= 75%.")

if __name__ == "__main__":
    main()