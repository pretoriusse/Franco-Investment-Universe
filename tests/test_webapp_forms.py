"""
Form-validation tests for webapp.forms, run against an in-memory SQLite DB.

RegistrationForm.__init__ queries Subscriptions directly from the database,
so it can't be tested as a pure unit -- these are small integration tests
using the flask_app fixture's throwaway SQLite DB instead of the real one.
"""

from webapp.forms import LoginForm, RegistrationForm
from webapp.models import Subscriptions, db


def _add_subscription(flask_app, name="Basic", cost=99.0, detail="Basic tier"):
    with flask_app.app_context():
        sub = Subscriptions(name=name, cost=cost, detail=detail)
        db.session.add(sub)
        db.session.commit()
        return sub.id


def test_login_form_requires_email_and_password(flask_app):
    with flask_app.test_request_context(method="POST", data={}):
        form = LoginForm()
        assert form.validate() is False
        assert "email" in form.errors
        assert "password" in form.errors


def test_login_form_valid_with_email_and_password(flask_app):
    with flask_app.test_request_context(
        method="POST", data={"email": "user@example.com", "password": "secret"}
    ):
        form = LoginForm()
        assert form.validate() is True


def test_registration_form_loads_subscription_choices_from_db(flask_app):
    sub_id = _add_subscription(flask_app)

    with flask_app.test_request_context(method="POST", data={}):
        form = RegistrationForm()
        assert (sub_id, "Basic") in form.subscription.choices


def test_registration_form_rejects_mismatched_passwords(flask_app):
    sub_id = _add_subscription(flask_app)

    with flask_app.test_request_context(
        method="POST",
        data={
            "email": "user@example.com",
            "name": "User",
            "id_number": "1234567890123",
            "password": "secret123",
            "confirm_password": "different",
            "subscription": str(sub_id),
        },
    ):
        form = RegistrationForm()
        assert form.validate() is False
        assert "confirm_password" in form.errors


def test_registration_form_valid_with_matching_passwords(flask_app):
    sub_id = _add_subscription(flask_app)

    with flask_app.test_request_context(
        method="POST",
        data={
            "email": "user@example.com",
            "name": "User",
            "id_number": "1234567890123",
            "password": "secret123",
            "confirm_password": "secret123",
            "subscription": str(sub_id),
        },
    ):
        form = RegistrationForm()
        assert form.validate() is True


def test_registration_form_rejects_invalid_email(flask_app):
    sub_id = _add_subscription(flask_app)

    with flask_app.test_request_context(
        method="POST",
        data={
            "email": "not-an-email",
            "name": "User",
            "id_number": "1234567890123",
            "password": "secret123",
            "confirm_password": "secret123",
            "subscription": str(sub_id),
        },
    ):
        form = RegistrationForm()
        assert form.validate() is False
        assert "email" in form.errors
