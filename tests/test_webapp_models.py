"""
Unit tests for webapp.models.Subscribers Flask-Login interface.

These properties drive access control (login.py checks subscription_paid and
relies on is_active implicitly through Flask-Login), so the expiry/blacklist
logic is worth pinning directly rather than only through the routes.
"""
from datetime import date, timedelta

from webapp.models import Subscribers


def _make_subscriber(**overrides):
    defaults = dict(
        subscription_expiration_date=date.today() + timedelta(days=30),
        black_listed=False,
    )
    defaults.update(overrides)
    return Subscribers(**defaults)


def test_is_active_true_when_not_expired_and_not_blacklisted():
    sub = _make_subscriber()
    assert sub.is_active is True


def test_is_active_false_when_subscription_expired():
    sub = _make_subscriber(subscription_expiration_date=date.today() - timedelta(days=1))
    assert sub.is_active is False


def test_is_active_false_when_blacklisted_even_if_not_expired():
    sub = _make_subscriber(black_listed=True)
    assert sub.is_active is False


def test_is_active_false_on_expiration_day_itself():
    # subscription_expiration_date == today is not "> today", so it expires
    # on its own expiration date rather than at the end of it.
    sub = _make_subscriber(subscription_expiration_date=date.today())
    assert sub.is_active is False


def test_get_id_returns_string():
    sub = _make_subscriber()
    sub.id = 42
    assert sub.get_id() == "42"


def test_is_authenticated_and_is_anonymous_are_fixed():
    sub = _make_subscriber()
    assert sub.is_authenticated is True
    assert sub.is_anonymous is False
