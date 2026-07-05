"""Register report files found on disk in the ``htmlwebview`` table.

The daily report jobs write HTML/PDF files under REPORTS_DIR but nothing
loads them into the webapp DB, so the frontend archive stays empty. This
module scans the reports directory and inserts any run that isn't registered
yet. ``start_report_sync`` (called from ``app.py``) runs a sync at startup
and then daily at 07:00 in a daemon thread.

Idempotent: rows are keyed on the detailed-HTML path, which carries a unique
constraint, so re-scans (and concurrent gunicorn workers) are harmless.
"""

import logging
import os
import platform
import re
import threading
import time
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from assets.const import DB_PARAMS_WEBAPP
from webapp.models import HTMLWebView, Subscribers

logger = logging.getLogger(__name__)

# REPORTS_DIR env var overrides everything. Otherwise prod (FLASK_ENV=
# production, same switch config.py uses for PayFast) reads the mounted
# backup share like close_report does, and dev reads the repo-local reports/
# — the mount can exist on the dev box too, so existence is no dev/prod test.
if os.environ.get("REPORTS_DIR"):
    REPORTS_DIR = os.environ["REPORTS_DIR"]
elif os.getenv("FLASK_ENV") == "production":
    REPORTS_DIR = (
        os.path.join(os.path.expanduser("~"), "Shares", "Reports")
        if platform.system() == "Windows"
        else "/mnt/backups/Shares/Reports"
    )
else:
    REPORTS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
    )

# File-name prefix on disk → report_type stored in htmlwebview.
REPORT_TYPES = {"close": "Close", "adjusted_close": "Adjusted Close"}
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_engine = create_engine(
    f"postgresql://{DB_PARAMS_WEBAPP['user']}:{DB_PARAMS_WEBAPP['password']}"
    f"@{DB_PARAMS_WEBAPP['host']}:{DB_PARAMS_WEBAPP['port']}/{DB_PARAMS_WEBAPP['dbname']}"
)
_Session = sessionmaker(bind=_engine)


def sync_reports() -> int:
    """Scan REPORTS_DIR and insert missing report runs. Returns rows added."""
    if not os.path.isdir(REPORTS_DIR):
        logger.warning(f"Report sync: directory not found: {REPORTS_DIR}")
        return 0

    added = 0
    with _Session() as session:
        known = {p for (p,) in session.query(HTMLWebView.html_detailed_path)}
        valid_subscriber_ids = {i for (i,) in session.query(Subscribers.id)}
        if not valid_subscriber_ids:
            logger.warning("Report sync: no subscribers to attach reports to")
            return 0
        fallback_id = min(valid_subscriber_ids)

        for date_dir in sorted(os.listdir(REPORTS_DIR)):
            dir_path = os.path.join(REPORTS_DIR, date_dir)
            if not DATE_DIR_RE.match(date_dir) or not os.path.isdir(dir_path):
                continue
            run_date = datetime.strptime(date_dir, "%Y-%m-%d").date()
            files = os.listdir(dir_path)

            for prefix, report_type in REPORT_TYPES.items():
                detailed_html = os.path.join(dir_path, f"{prefix}_detailed.html")
                if not os.path.exists(detailed_html) or detailed_html in known:
                    continue

                def path_or(name: str, fallback: str = "") -> str:
                    p = os.path.join(dir_path, name)
                    return p if os.path.exists(p) else fallback

                # Attribute the run to whichever subscriber has a personalised
                # file in this folder; general-only runs go to the first
                # subscriber. Row paths always point at the general files so
                # no user's personalised report is served to everyone.
                user_re = re.compile(rf"^user_(\d+)_{prefix}_detailed")
                user_ids = sorted(
                    int(m.group(1))
                    for f in files
                    if (m := user_re.match(f))
                    and int(m.group(1)) in valid_subscriber_ids
                )
                subscriber_id = user_ids[0] if user_ids else fallback_id

                session.add(
                    HTMLWebView(
                        display_date=date_dir,
                        report_type=report_type,
                        html_summary_path=path_or(
                            f"{prefix}_summary.html", detailed_html
                        ),
                        html_detailed_path=detailed_html,
                        pdf_summary_path=path_or(f"{prefix}_summary.pdf"),
                        pdf_detailed_path=path_or(f"{prefix}_detailed.pdf"),
                        actual_run_date=run_date,
                        subscriber_id=subscriber_id,
                    )
                )
                try:
                    session.commit()
                    added += 1
                except SQLAlchemyError:
                    # Another worker registered it between our snapshot and now.
                    session.rollback()

    if added:
        logger.info(f"Report sync: registered {added} new report run(s)")
    return added


def _sync_loop() -> None:
    while True:
        try:
            sync_reports()
        except Exception:
            logger.exception("Report sync failed")
        now = datetime.now()
        next_run = now.replace(hour=7, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        time.sleep((next_run - now).total_seconds())


def start_report_sync() -> None:
    """Sync now, then daily at 07:00, in a background daemon thread."""
    threading.Thread(target=_sync_loop, daemon=True, name="report-sync").start()
