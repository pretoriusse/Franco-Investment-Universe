"""CONSTANTS File"""
import os

DB_PARAMS = {
    'dbname': 'sharesdata',
    'user': 'postgres',
    'password': 'Pr3t0r1u5',
    'host': '192.168.0.181',
    'port': '5432'
}

EMAIL_ADDRESS = str(os.environ.get('EMAIL_ADDRESS', 'pretoriusspprt@gmail.com'))
SERVER_ADDRESS = str(os.environ.get('SERVER_ADDRESS', 'smtp.gmail.com'))
SERVER_PORT = int(os.environ.get('SERVER_PORT', '587'))
EMAIL_PASSWORD = str(os.environ.get('EMAIL_PASSWORD', 'lhdrcfhkdnatnrlo'))