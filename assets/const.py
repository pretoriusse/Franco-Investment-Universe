"""CONSTANTS File"""
import os

DB_PARAMS = {
    'dbname': 'sharesdata',
    'user': 'postgres',
    'password': 'Pr3t0r1u5',
    'host': '192.168.50.138',
    'port': '5432'
}

EMAIL_ADDRESS = str(os.environ.get('EMAIL_ADDRESS', 'pretoriusspprt@gmail.com'))
SERVER_ADDRESS = str(os.environ.get('SERVER_ADDRESS', 'smtp.gmail.com'))
SERVER_PORT = int(os.environ.get('SERVER_PORT', '587'))
EMAIL_PASSWORD = str(os.environ.get('EMAIL_PASSWORD', 'lhdrcfhkdnatnrlo'))

DB_PARAMS_WEBAPP = {
    'dbname': 'webapp',
    'user': 'postgres',
    'password': 'Pr3t0r1u5',
    'host': '127.0.0.1',
    'port': '5432'
}

DB_PARAMS_EXCELDATA = {
    'dbname': 'exceldata',
    'user': 'postgres',
    'password': 'Pr3t0r1u5',
    'host': '127.0.0.1',
    'port': '5432'
}

hparams = {
    'HP_LSTM_UNITS': 600,       # Number of LSTM units
    'HP_GRU_UNITS': 300,        # Number of GRU units
    'HP_DROPOUT': 0.2,          # Dropout rate
    'HP_EPOCHS': 800,           # Maximum number of epochs
    'TARGET_ACCURACY': 90.0,    # Desired validation accuracy within 10%
    'PATIENCE': 20,             # Patience for EarlyStopping
    'OPTUNA_TRIALS':30,        # Number of Optuna trials
    'SEQ_LENGTH': 60,           # Sequence length for time series
}
