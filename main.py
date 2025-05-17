import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow logging suppression

from adjusted_close_report import daily_job as adj_close_main
from close_report import daily_job as close_main
import threading
import schedule
import time

def main():
    # Start Adjusted Close job
    """adj_close_thread = threading.Thread(target=adj_close_main, name='Adjusted Close Main Thread')
    adj_close_thread.start()

    # Start Close job 20 minutes after Adjusted Close
    close_thread = threading.Thread(target=close_main, name='Close Main Thread')
    close_thread.start()

    # Join both threads to ensure they complete before exiting
    adj_close_thread.join()
    close_thread.join()"""
    close_main()
    adj_close_main()

def setup_scheduler():
    schedule.every().day.at("06:00").do(main)
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == '__main__':
    main()
    setup_scheduler()
