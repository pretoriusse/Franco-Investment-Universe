"""Tests for webapp.report_sync — disk scan → htmlwebview registration."""

import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import webapp.report_sync as report_sync
from webapp.models import Base, HTMLWebView, Subscribers


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _add_subscriber(Session, sid):
    with Session() as s:
        s.add(
            Subscribers(
                id=sid,
                email=f"u{sid}@example.com",
                name="u",
                password="x",
                subscription_date=datetime.date.today(),
                subscription_expiration_date=datetime.date.today(),
                id_number="123",
            )
        )
        s.commit()


def test_sync_registers_backlog_and_is_idempotent(
    tmp_path, monkeypatch, session_factory
):
    monkeypatch.setattr(report_sync, "_Session", session_factory)
    monkeypatch.setattr(report_sync, "REPORTS_DIR", str(tmp_path))
    _add_subscriber(session_factory, 1)
    _add_subscriber(session_factory, 7)

    # Full close run with a personalised file for subscriber 7
    day = tmp_path / "2025-07-01"
    day.mkdir()
    for f in [
        "close_summary.html",
        "close_detailed.html",
        "close_summary.pdf",
        "close_detailed.pdf",
        "user_7_close_detailed.html",
    ]:
        (day / f).write_text("x")
    # Adjusted run with only a detailed html and no user files
    day2 = tmp_path / "2025-06-30"
    day2.mkdir()
    (day2 / "adjusted_close_detailed.html").write_text("x")
    # Non-date junk in the reports dir is ignored
    (tmp_path / "sync.ffs_db").write_text("")

    assert report_sync.sync_reports() == 2
    assert report_sync.sync_reports() == 0  # re-scan adds nothing

    with session_factory() as s:
        rows = {r.report_type: r for r in s.query(HTMLWebView).all()}

    close = rows["Close"]
    assert close.display_date == "2025-07-01"
    assert close.subscriber_id == 7  # personalised file wins
    assert close.html_detailed_path.endswith("close_detailed.html")
    assert "user_" not in close.html_detailed_path  # general file is served

    adj = rows["Adjusted Close"]
    assert adj.subscriber_id == 1  # fallback: first subscriber
    assert adj.html_summary_path == adj.html_detailed_path  # summary fallback
    assert adj.pdf_detailed_path == ""


def test_sync_without_subscribers_adds_nothing(tmp_path, monkeypatch, session_factory):
    monkeypatch.setattr(report_sync, "_Session", session_factory)
    monkeypatch.setattr(report_sync, "REPORTS_DIR", str(tmp_path))
    day = tmp_path / "2025-07-01"
    day.mkdir()
    (day / "close_detailed.html").write_text("x")

    assert report_sync.sync_reports() == 0
