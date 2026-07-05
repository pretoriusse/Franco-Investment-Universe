"""Application-wide constants and connection parameters."""

from __future__ import annotations

import os
from typing import Final

# Connection parameters for the market-data ("sharesdata") database.
DB_PARAMS: Final[dict[str, str]] = {
    "dbname": "sharesdata",
    "user": "postgres",
    "password": "Pr3t0r1u5",
    "host": "192.168.50.138",
    "port": "5432",
}

EMAIL_ADDRESS: Final[str] = str(
    os.environ.get("EMAIL_ADDRESS", "pretoriusspprt@gmail.com")
)
SERVER_ADDRESS: Final[str] = str(os.environ.get("SERVER_ADDRESS", "smtp.gmail.com"))
SERVER_PORT: Final[int] = int(os.environ.get("SERVER_PORT", "587"))
EMAIL_PASSWORD: Final[str] = str(os.environ.get("EMAIL_PASSWORD", "lhdrcfhkdnatnrlo"))

# Optional NewsAPI.org key for the macro-sentiment pipeline. Empty string
# disables that source (RSS + GDELT still run without any key).
NEWSAPI_KEY: Final[str] = str(os.environ.get("NEWSAPI_KEY", ""))

# Connection parameters for the web-application ("webapp") database.
DB_PARAMS_WEBAPP: Final[dict[str, str]] = {
    "dbname": "webapp",
    "user": "postgres",
    "password": "Pr3t0r1u5",
    "host": "127.0.0.1",
    "port": "5432",
}

# Connection parameters for the spreadsheet-import ("exceldata") database.
DB_PARAMS_EXCELDATA: Final[dict[str, str]] = {
    "dbname": "exceldata",
    "user": "postgres",
    "password": "Pr3t0r1u5",
    "host": "127.0.0.1",
    "port": "5432",
}

# Default model hyperparameters. Values are numeric (ints widen to float under
# the numeric tower) so the dict is typed ``dict[str, float]``; callers that
# need an int (e.g. epoch counts) cast at the use-site.
hparams: Final[dict[str, float]] = {
    "HP_LSTM_UNITS": 600,  # Number of LSTM units
    "HP_GRU_UNITS": 300,  # Number of GRU units
    "HP_DROPOUT": 0.2,  # Dropout rate
    "HP_EPOCHS": 800,  # Maximum number of epochs
    "TARGET_ACCURACY": 90.0,  # Desired validation accuracy within 10%
    "PATIENCE": 20,  # Patience for EarlyStopping
    "OPTUNA_TRIALS": 30,  # Number of Optuna trials
    "SEQ_LENGTH": 60,  # Sequence length for time series
}
