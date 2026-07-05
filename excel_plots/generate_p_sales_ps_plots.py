# generate_scatter_plot.py

import matplotlib

matplotlib.use("Agg")  # Set the backend to Agg for non-interactive plotting

import os
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from assets.const import DB_PARAMS_EXCELDATA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def fetch_data():
    """
    Connects to the PostgreSQL database and fetches data from the specified table.

    Returns:
        pd.DataFrame: DataFrame containing the fetched data.
    """
    try:
        # Construct the PostgreSQL connection URL
        connection_url = (
            f"postgresql://{DB_PARAMS_EXCELDATA['user']}:{DB_PARAMS_EXCELDATA['password']}"
            f"@{DB_PARAMS_EXCELDATA['host']}:{DB_PARAMS_EXCELDATA['port']}/{DB_PARAMS_EXCELDATA['dbname']}"
        )

        # Create the SQLAlchemy engine
        engine = create_engine(connection_url)
        print("Database connection established.")

        # Define the SQL query
        query = "SELECT * FROM public.aipcharts_chart_data;"

        # Execute the query and read data into a pandas DataFrame
        df = pd.read_sql(query, engine)
        print("Data fetched successfully.")

        # Print columns for verification
        print("DataFrame Columns:", df.columns.tolist())

        return df

    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return None


def process_data(df):
    """
    Processes the fetched data to prepare for plotting.

    Args:
        df (pd.DataFrame): The raw DataFrame fetched from the database.

    Returns:
        pd.DataFrame: Processed DataFrame ready for plotting.
        pd.DataFrame: DataFrame containing main time period ranges.
    """
    try:
        # Identify the 'date' column (case-insensitive)
        date_column = None
        for col in df.columns:
            if col.lower() == "date":
                date_column = col
                break
        if date_column is None:
            raise ValueError("The DataFrame does not contain a 'date' column.")

        # Convert 'date' column to datetime
        df[date_column] = pd.to_datetime(df[date_column])

        # Sort the DataFrame by date
        df = df.sort_values(by=date_column).reset_index(drop=True)

        # Ensure 'P_Sales_ps' exists
        p_sales_column = "P_Sales_ps"
        if p_sales_column not in df.columns:
            # Attempt to find 'P_Sales_ps' with different cases or slight variations
            possible_columns = [
                col for col in df.columns if col.lower() == "p_sales_ps".lower()
            ]
            if possible_columns:
                p_sales_column = possible_columns[0]
                df.rename(columns={p_sales_column: "P_Sales_ps"}, inplace=True)
                print(f"Renamed column '{p_sales_column}' to 'P_Sales_ps'.")
            else:
                raise ValueError(
                    "The DataFrame does not contain a 'P_Sales_ps' column."
                )

        # Ensure 'P__change_24m' exists
        p_change_column = "P__change_24m"
        if p_change_column not in df.columns:
            # Attempt to find 'P__change_24m' with different cases or slight variations
            possible_columns = [
                col for col in df.columns if col.lower() == "p__change_24m".lower()
            ]
            if possible_columns:
                p_change_column = possible_columns[0]
                df.rename(columns={p_change_column: "P__change_24m"}, inplace=True)
                print(f"Renamed column '{p_change_column}' to 'P__change_24m'.")
            else:
                raise ValueError(
                    "The DataFrame does not contain a 'P__change_24m' column."
                )

        # Convert 'P_Sales_ps' and 'P__change_24m' to numeric, coercing errors
        df["P_Sales_ps"] = pd.to_numeric(df["P_Sales_ps"], errors="coerce")
        df["P__change_24m"] = pd.to_numeric(df["P__change_24m"], errors="coerce")

        # Drop rows with NaN in 'P_Sales_ps' or 'P__change_24m'
        initial_count = len(df)
        df = df.dropna(subset=["P_Sales_ps", "P__change_24m"]).reset_index(drop=True)
        final_count = len(df)
        print(
            f"Dropped {initial_count - final_count} rows due to NaN in 'P_Sales_ps' or 'P__change_24m'."
        )

        # Find the first non-zero 'P_Sales_ps' and add one year to that date
        non_zero_eps_date = df[df["P_Sales_ps"] > 0][date_column].min()
        if pd.isna(non_zero_eps_date):
            raise ValueError("No non-zero 'P_Sales_ps' values found in the data.")

        # Add one year to the date
        cutoff_date = non_zero_eps_date + pd.DateOffset(years=1)
        print(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")

        # Filter the data to include only rows with a date greater than the cutoff date
        df = df[df[date_column] > cutoff_date].reset_index(drop=True)
        print(
            f"Filtered data to include only dates after {cutoff_date.strftime('%Y-%m-%d')}."
        )

        # Remove data points where P_Sales_ps <= 0 to eliminate clutter around 0
        before_filter = len(df)
        df = df[df["P_Sales_ps"] > 0].reset_index(drop=True)
        after_filter = len(df)
        print(f"Removed {before_filter - after_filter} rows where 'P_Sales_ps' <= 0.")

        # Remove data points where P__change_24m == 0% to eliminate clutter on the horizontal axis
        before_pchange_filter = len(df)
        df = df[df["P__change_24m"] != 0].reset_index(drop=True)
        after_pchange_filter = len(df)
        print(
            f"Removed {before_pchange_filter - after_pchange_filter} rows where 'P__change_24m' == 0%."
        )

        # Split the data into two main sections based on the median date
        median_date = df[date_column].median()
        df["Main Set"] = df[date_column].apply(lambda x: 1 if x <= median_date else 2)
        print(
            f"Data split into two main sets based on median date: {median_date.strftime('%Y-%m-%d')}"
        )

        # Within each main set, split into five sub-sets using quantiles
        df["Sub Set"] = df.groupby("Main Set")["P_Sales_ps"].transform(
            lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
        )

        # Handle cases where a main set might have less than 5 unique values
        if df["Sub Set"].isnull().any():
            print(
                "Warning: Some sub-sets could not be created due to insufficient unique values."
            )
            df["Sub Set"] = df["Sub Set"].fillna(-1).astype(int)

        # Create a combined 'Set' identifier for coloring (1-10)
        df["Set"] = (
            (df["Main Set"] - 1) * 5 + df["Sub Set"] + 1
        )  # Sets 1-5 for Main Set 1 and 6-10 for Main Set 2

        # Define color palette with 10 distinct colors
        color_map = plt.get_cmap("tab10")
        df["Color"] = df["Set"].apply(
            lambda x: color_map((x - 1) % 10)
        )  # Adjusted for 0-based indexing

        # Create a mapping for the two main sets with date ranges included
        time_period_ranges_2 = (
            df.groupby("Main Set")[date_column].agg(["min", "max"]).reset_index()
        )
        time_period_ranges_2["Time Period"] = time_period_ranges_2.apply(
            lambda row: (
                f"Main Period {row['Main Set']} ({row['min'].strftime('%Y-%m-%d')} to {row['max'].strftime('%Y-%m-%d')})"
            ),
            axis=1,
        )
        time_period_ranges_2 = time_period_ranges_2[
            ["Main Set", "Time Period", "min", "max"]
        ]

        print("Data processing complete.")
        print(time_period_ranges_2)

        return df, time_period_ranges_2

    except Exception as e:
        print(f"An error occurred during data processing: {e}")
        return None, None


def perform_regression(df, time_period_ranges_2, x_column, y_column):
    """
    Performs linear regression for each defined main time period.

    Args:
        df (pd.DataFrame): The processed DataFrame.
        time_period_ranges_2 (pd.DataFrame): DataFrame containing main time period ranges.
        x_column (str): The column name to be used as the independent variable.
        y_column (str): The column name to be used as the dependent variable.

    Returns:
        dict: Dictionary containing regression models, R² scores, and predictions per period.
    """
    try:
        regression_results = {}
        for _, row in time_period_ranges_2.iterrows():
            period = row["Time Period"]
            main_set = row["Main Set"]
            period_df = df[df["Main Set"] == main_set]

            # Ensure there are enough data points
            if len(period_df) < 2:
                print(f"Not enough data points for {period} to perform regression.")
                continue

            X = period_df[[x_column]].values
            y = period_df[
                y_column
            ].values  # Assuming 'P__change_24m' is already in percentage

            # Fit linear regression
            reg = LinearRegression()
            reg.fit(X, y)
            y_pred = reg.predict(X)
            r_squared = r2_score(y, y_pred)

            regression_results[main_set] = {
                "model": reg,
                "r_squared": r_squared,
                "X": X,
                "y_pred": y_pred,
                "Time Period": period,
            }

            print(f"Regression for {period}: R² = {r_squared:.2f}")

        return regression_results

    except Exception as e:
        print(f"An error occurred during regression analysis: {e}")
        return {}


def generate_graph(
    df, time_period_ranges_2, regression_results, x_column, y_column, title
):
    """
    Generates a scatter plot with regression lines for main periods and data points colored by sub-sets.

    Args:
        df (pd.DataFrame): The processed DataFrame.
        time_period_ranges_2 (pd.DataFrame): DataFrame containing main time period ranges.
        regression_results (dict): Dictionary containing regression models and results.
        x_column (str): The column name to be used as the independent variable.
        y_column (str): The column name to be used as the dependent variable.
        title (str): The title of the plot.
    """
    try:
        plt.figure(figsize=(16, 10))

        # Scatter plot for each sub-set with distinct colors
        for set_number in sorted(df["Set"].unique()):
            subset = df[df["Set"] == set_number]
            if subset.empty:
                continue  # Skip empty subsets
            plt.scatter(
                subset[x_column],
                subset[y_column],
                alpha=0.7,
                edgecolors="w",
                s=100,
                label=f"Set {set_number}",
                color=subset["Color"].iloc[0],
            )

        # Plot regression lines for main periods
        for main_set, results in regression_results.items():
            reg = results["model"]
            r_squared = results["r_squared"]
            X = results["X"]
            y_pred = results["y_pred"]
            period = results["Time Period"]

            color = df[df["Main Set"] == main_set]["Color"].iloc[0]

            plt.plot(
                X.flatten(),
                y_pred,
                color=color,
                linewidth=2,
                label=f"{period} Regression (R² = {r_squared:.2f})",
            )

        # Add a vertical line at the latest P_Sales_ps value
        latest_value = df[x_column].iloc[-1]
        plt.axvline(
            x=latest_value,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Latest {title} = {latest_value:.2f}",
        )

        plt.title(f"Scatter Plot of {title} vs P_Sales_ps by Time Period", fontsize=18)
        plt.xlabel("P_Sales_ps", fontsize=14)
        plt.ylabel("P__change_24m (%)", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.5)

        # Adjust legend to avoid overlapping
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize="small",
            title="Sets and Regressions",
        )
        plt.tight_layout()

        # Handle both 'Ticker' and 'ticker' column names
        ticker_column = None
        for col in df.columns:
            if col.lower() == "ticker":
                ticker_column = col
                break
        if ticker_column is None:
            raise ValueError("The DataFrame does not contain a 'Ticker' column.")

        ticker = df[ticker_column].iloc[0].replace(".JO", "")
        save_dir = os.path.join("plots", ticker, "excel_plots")
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

        # Save the plot
        save_path = os.path.join(save_dir, "P_Sales_ps_Scatter_Plot.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

        # Display the plot
        # plt.show()

    except Exception as e:
        print(f"An error occurred while generating the graph: {e}")


def main():
    """
    Main function to fetch data, process it, perform regression, and generate the graph.
    """
    # Define the columns to plot
    x_column = "P_Sales_ps"  # Independent variable
    y_column = "P__change_24m"  # Dependent variable
    title = "P_Sales_ps"  # Title for the x-axis

    # Fetch data from the database
    df = fetch_data()
    if df is None:
        return

    # Process the data
    df, time_period_ranges_2 = process_data(df)
    if df is None or time_period_ranges_2 is None:
        return

    # Ensure the X and Y columns exist
    if x_column not in df.columns:
        print(f"The specified X column '{x_column}' does not exist in the DataFrame.")
        print("Available columns:", df.columns.tolist())
        return
    if y_column not in df.columns:
        print(f"The specified Y column '{y_column}' does not exist in the DataFrame.")
        print("Available columns:", df.columns.tolist())
        return

    # Perform regression analysis on the two main sections
    regression_results = perform_regression(
        df, time_period_ranges_2, x_column, y_column
    )

    # Generate the scatter plot with regression lines
    generate_graph(
        df, time_period_ranges_2, regression_results, x_column, y_column, title
    )


if __name__ == "__main__":
    main()
