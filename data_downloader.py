import schedule
from assets import upload_history, zar_process, dividends, fetch_daily_commodity_data
import time
import threading

def main():
    threads = []

    # Function to start and append a thread
    def start_thread(target, name):
        thread = threading.Thread(target=target, name=name)
        thread.start()
        threads.append(thread)
        print(f"{name} thread started")

    #start_thread(zar_process.process_zar, 'Process ZAR')
    start_thread(upload_history.main, 'Upload History')
    #start_thread(dividends.main, 'Dividend Upload')
    #start_thread(fetch_daily_commodity_data.main, 'Commodity Upload')

    for thread in threads:
        thread.join()

def setup_scheduler():
    schedule.every().day.at("17:30").do(main)
    while True:
        schedule.run_pending()
        time.sleep(15)


if __name__ == '__main__':
    main()
    setup_scheduler()