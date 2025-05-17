import schedule
from assets import upload_history, zar_process, dividends, fetch_daily_commodity_data, const
import time
import threading
import psycopg2
from psycopg2 import sql

def update_materialized_views():
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
                cursor.execute(sql.SQL("REFRESH MATERIALIZED VIEW {};").format(sql.Identifier(view_name)))
                print(f"Refreshed materialized view: {view_name}")

        # Commit the transaction
        conn.commit()
    
    except Exception as e:
        print(f"An error occurred while refreshing materialized views: {e}")
    
    finally:
        # Ensure the connection is closed
        if conn:
            conn.close()

def main():
    threads = []

    # Function to start and append a thread
    def start_thread(target, name):
        thread = threading.Thread(target=target, name=name)
        thread.start()
        threads.append(thread)
        print(f"{name} thread started")

    start_thread(zar_process.process_zar, 'Process ZAR')
    start_thread(upload_history.main, 'Upload History')
    #start_thread(dividends.main, 'Dividend Upload')
    start_thread(fetch_daily_commodity_data.main, 'Commodity Upload')
    
    for thread in threads:
        thread.join()

    upload_history.main()
    zar_process.process_zar()
    #dividends.main()
    fetch_daily_commodity_data.main()

    # Call the function to update materialized views
    update_materialized_views()

def setup_scheduler():
    # Schedule the main function to run every day at 17:30
    schedule.every().day.at("17:10").do(main)
    while True:
        schedule.run_pending()
        time.sleep(15)

if __name__ == '__main__':
    main()
    setup_scheduler()
