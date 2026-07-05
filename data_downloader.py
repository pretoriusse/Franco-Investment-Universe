"""Daily market-data ingestion entry point.

Runs the price/ZAR/commodity uploaders, then the per-ticker and macro
news-sentiment pipelines, and finally refreshes the materialized views.
Scheduled to run every day at 17:10.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import psycopg2
import schedule
from psycopg2 import sql

from assets import const, fetch_daily_commodity_data, upload_history, zar_process
from assets.macro_sentiment import run_daily_macro_sentiment_pipeline
from assets.news_sentiment import run_daily_sentiment_pipeline


def update_materialized_views() -> None:
    conn = None
    try:
        # Establish a connection to the database
        conn = psycopg2.connect(**const.DB_PARAMS)

        with conn.cursor() as cursor:
            # Query to get all materialized views in the current schema
            cursor.execute("""
                SELECT matviewname
                FROM pg_matviews
                WHERE schemaname = 'public';
            """)
            views = cursor.fetchall()

            # Refresh each materialized view
            for view in views:
                view_name = view[0]
                cursor.execute(
                    sql.SQL("REFRESH MATERIALIZED VIEW {};").format(
                        sql.Identifier(view_name)
                    )
                )
                print(f"Refreshed materialized view: {view_name}")

        # Commit the transaction
        conn.commit()

    except Exception as e:
        print(f"An error occurred while refreshing materialized views: {e}")

    finally:
        # Ensure the connection is closed
        if conn is not None:
            conn.close()


def main() -> None:
    threads: list[threading.Thread] = []

    # Function to start and append a thread
    def start_thread(target: Callable[[], None], name: str) -> None:
        thread = threading.Thread(target=target, name=name)
        thread.start()
        threads.append(thread)
        print(f"{name} thread started")

    start_thread(zar_process.process_zar, "Process ZAR")
    start_thread(upload_history.main, "Upload History")
    # start_thread(dividends.main, 'Dividend Upload')
    start_thread(fetch_daily_commodity_data.main, "Commodity Upload")

    for thread in threads:
        thread.join()

    upload_history.main()
    zar_process.process_zar()
    # dividends.main()
    fetch_daily_commodity_data.main()

    # Fetch and store news sentiment for all tickers after market close
    try:
        from assets import database_queries as db_queries

        universe = db_queries.fetch_stock_and_commodity_universe_from_db()
        tickers = universe["code"].dropna().tolist()
        if tickers:
            run_daily_sentiment_pipeline(tickers)
    except Exception as e:
        print(f"Sentiment pipeline error: {e}")

    # Fetch and store macro/thematic sentiment (broad + geopolitical news).
    try:
        run_daily_macro_sentiment_pipeline()
    except Exception as e:
        print(f"Macro sentiment pipeline error: {e}")

    # Call the function to update materialized views
    update_materialized_views()


def setup_scheduler() -> None:
    # Schedule the main function to run every day at 17:30
    schedule.every().day.at("17:10").do(main)
    while True:
        schedule.run_pending()
        time.sleep(15)


if __name__ == "__main__":
    main()
    setup_scheduler()
