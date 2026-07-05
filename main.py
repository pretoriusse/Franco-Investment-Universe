import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TensorFlow logging suppression

import logging
import time

import schedule

from adjusted_close_report import daily_job as adj_close_main
from close_report import daily_job as close_main

logger = logging.getLogger(__name__)


def main():
    # Run the two jobs sequentially, NOT in parallel threads. Each job loads
    # per-ticker LSTM models onto the GPU; running both at once would roughly
    # double peak VRAM use and OOM the ~6-8 GB free on the RTX 3060. Sequential
    # execution keeps the GPU within budget — each job frees its models as it goes.
    #
    # Both calls are guarded so a failure in one job (or a transient DB/GPU
    # error) can't kill the scheduler loop and stop tomorrow's run.
    for label, job in (("close", close_main), ("adjusted close", adj_close_main)):
        try:
            job()
        except Exception:
            logger.exception("%s daily job failed", label)


def setup_scheduler():
    schedule.every().day.at("06:00").do(main)
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
    setup_scheduler()
