from assets.const import DB_PARAMS_WEBAPP as DB_PARAMS
import os
from datetime import timedelta

class Config:
    SECRET_KEY = 'supersecretkey'
    SQLALCHEMY_DATABASE_URI = f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    PAYFAST_MERCHANT_ID = '10035140'  # Use sandbox credentials if testing
    PAYFAST_MERCHANT_KEY = '0lnni1195i3g0'  # Use sandbox credentials if testing
    PAYFAST_PASSPHRASE = 'Hgx51MLLePOz5eIno2b5EBw5OIibud3d'  # Optional
    PAYFAST_RETURN_URL = 'https://research.pretoriusse.net/payment/success'
    PAYFAST_CANCEL_URL = 'https://research.pretoriusse.net/payment/cancel'
    PAYFAST_NOTIFY_URL = 'https://research.pretoriusse.net/payment/ipn'
    REMEMBER_COOKIE_NAME = 'MarketWatchLoginCookie'
    COOKIE_DURATION = timedelta(days=14)
    PAYFAST_VERSION = '1'
    PAYFAST_API_URL = 'https://api.payfast.co.za'

    # Check the environment variable to decide which URL to use
    if os.getenv('FLASK_ENV') == 'production':
        PAYFAST_URL = 'https://www.payfast.co.za/eng/process'  # Production URL
    else:
        PAYFAST_URL = 'https://sandbox.payfast.co.za/eng/process'  # Sandbox URL for testing
