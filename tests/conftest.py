import pytest


@pytest.fixture
def flask_app():
    """Minimal Flask app + in-memory SQLite DB, isolated from the real webapp DB."""
    from flask import Flask
    from webapp.models import db

    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SECRET_KEY"] = "test-secret"
    app.config["WTF_CSRF_ENABLED"] = False
    db.init_app(app)

    with app.app_context():
        db.create_all()
        yield app
