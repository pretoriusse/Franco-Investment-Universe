"""Flask application configuration (web-app database + PayFast integration)."""

from __future__ import annotations

import os
from datetime import timedelta

from assets.const import DB_PARAMS_WEBAPP as DB_PARAMS


_IS_PROD = os.getenv("FLASK_ENV") == "production"


class Config:
    """Flask config object; attributes are read by ``app.config.from_object``."""

    # Session signing key. MUST be set via env in production — the hardcoded
    # fallback is dev-only; anyone who knows it can forge login sessions.
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-only-insecure-key-change-me")

    # Cookie hardening
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = "Lax"
    SESSION_COOKIE_SECURE: bool = _IS_PROD  # HTTPS-only cookies in production
    REMEMBER_COOKIE_HTTPONLY: bool = True
    REMEMBER_COOKIE_SECURE: bool = _IS_PROD
    SQLALCHEMY_DATABASE_URI: str = (
        f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}"
        f"@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    PAYFAST_MERCHANT_ID: str = "10035140"  # Use sandbox credentials if testing
    PAYFAST_MERCHANT_KEY: str = "0lnni1195i3g0"  # Use sandbox credentials if testing
    PAYFAST_PASSPHRASE: str = "Hgx51MLLePOz5eIno2b5EBw5OIibud3d"  # Optional
    PAYFAST_RETURN_URL: str = "https://research.pretoriusse.net/payment/success"
    PAYFAST_CANCEL_URL: str = "https://research.pretoriusse.net/payment/cancel"
    PAYFAST_NOTIFY_URL: str = "https://research.pretoriusse.net/payment/ipn"
    REMEMBER_COOKIE_NAME: str = "MarketWatchLoginCookie"
    COOKIE_DURATION: timedelta = timedelta(days=14)
    PAYFAST_VERSION: str = "1"
    PAYFAST_API_URL: str = "https://api.payfast.co.za"

    # Production uses the live PayFast endpoint; everything else uses sandbox.
    PAYFAST_URL: str = (
        "https://www.payfast.co.za/eng/process"
        if os.getenv("FLASK_ENV") == "production"
        else "https://sandbox.payfast.co.za/eng/process"
    )
