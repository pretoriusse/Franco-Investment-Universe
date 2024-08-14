from adjusted_close_testing import daily_job as adj_close_main
from close_prediction_testing import daily_job as close_main
import threading
import schedule
import time

def main():
    adj_thread = threading.Thread(target=adj_close_main, name='Adjusted Close Thread')
    close_thread = threading.Thread(target=close_main, name='Close Thread')

    adj_thread.start()
    close_thread.start()

    adj_thread.join()
    close_thread.join()

def setup_scheduler():
    schedule.every().day.at("00:10").do(main)
    while True:
        schedule.run_pending()
        time.sleep(15)


if __name__ == '__main__':
    main()
    setup_scheduler()