# Franco Investment Universe — CLAUDE.md

## Project Overview

Franco Investment Universe is a production investment-analysis platform for the
**Johannesburg Stock Exchange (JSE)**. It runs three loosely coupled subsystems:

1. **Daily ML report pipeline** — downloads fresh market data, runs LSTM/GRU price
   predictions, generates HTML + PDF reports, and emails them to subscribers.
2. **Flask web application** — subscriber portal at `https://research.pretoriusse.net`
   with user auth, subscription management (PayFast), portfolio tracking, and a
   report archive.
3. **Auxiliary data pipeline** — ingests ZAR forex rates, dividend data, and
   commodity futures (gold `GC=F`, copper `HG=F`) on a separate schedule.

The stock universe (`investment_universe.csv`) contains 100+ JSE-listed equities
plus commodity futures, classified by industry and sub-industry.

---

## Architecture

```
main.py (06:00 daily)
├── close_report.py        ← close-price LSTM predictions → HTML/PDF → email
└── adjusted_close_report.py ← adj-close LSTM predictions → HTML/PDF → email

data_downloader.py (17:10 daily)
├── assets/zar_process.py        ← ZAR/USD rate ingestion
├── assets/upload_history.py     ← portfolio transaction sync
├── assets/fetch_daily_commodity_data.py ← commodity OHLCV
└── DB materialized-view refresh

app.py (Flask, always-on)
├── /                 home / report list
├── /login /register  auth
├── /portfolio        portfolio tracker
├── /reports          report archive per subscriber
├── /pay_subscription /payment/* PayFast IPN/return/cancel
└── /manage_subscription admin panel
```

---

## Directory Map

```
Franco-Investment-Universe/
├── main.py                      # ML report scheduler entry point (06:00)
├── app.py                       # Flask web application (~21 KB)
├── close_report.py              # Close-price daily job (~68 KB)
├── adjusted_close_report.py     # Adj-close daily job (~69 KB)
├── data_downloader.py           # Auxiliary data pipeline (17:10)
├── create_tables.py             # One-off DB table creation helper
├── Management_GUI.py            # Desktop admin GUI (tkinter)
├── investment_universe.csv      # Master stock list (100+ JSE tickers)
├── db_config.json               # sharesdata DB connection (legacy reference)
├── requirements.txt             # Full dependency list (200+ packages)
├── webapp_requirements.txt      # Flask-only subset
│
├── assets/
│   ├── const.py                 # Global constants: DB params, email creds, ML hyperparams
│   ├── config.py                # Flask Config class (PayFast, session, SQLALCHEMY_URI)
│   ├── models.py                # SQLAlchemy ORM for sharesdata DB
│   ├── database_queries.py      # Query helpers used by report jobs (~26 KB)
│   ├── get_best_metrics_for_valuation.py
│   ├── technical_analysis.py    # TA indicators (RSI, MA, Z-score, etc.)
│   ├── dividends.py
│   ├── fetch_daily_commodity_data.py
│   ├── zar_process.py
│   ├── upload_history.py
│   └── payments.py              # PayFast API helpers
│
├── webapp/
│   ├── models.py                # SQLAlchemy ORM for webapp DB (Subscribers, etc.)
│   └── forms.py                 # Flask-WTF forms (Registration, Login)
│
├── training/
│   ├── close.py                 # LSTM training for close price
│   ├── adjusted_close.py        # LSTM training for adjusted close
│   ├── oldclosetraining.py      # Archived previous training approach
│   └── old_adjusted_training.py
│
├── models/                      # Saved Keras models — one sub-folder per ticker
│   └── {TICKER.JO}/
│       └── metadata.json        # Model accuracy, training date, hyperparams
│
├── plots/                       # Generated stock-price chart PNGs (~245 MB)
├── reports/                     # Generated HTML + PDF reports (~352 MB)
├── runs/                        # TensorBoard event files (~3 MB)
│
├── templates/                   # Flask Jinja2 templates
│   ├── base.html
│   ├── home.html / login.html / register.html
│   ├── portfolio.html / reports.html
│   └── subscriptions.html / pay_subscription.html / manage_subscription.html
│
├── static/                      # CSS, JS, images (~9 MB)
│   ├── css/ / js/ / images/
│   └── styles.css
│
├── email_template.html          # Jinja2 email body template
├── detailed_template.html       # Detailed report HTML template
├── summary_template.html        # Summary report HTML template
├── web_template.html            # Web-view report template
│
├── migrations/                  # Flask-Migrate / Alembic versions
│   └── versions/
│       ├── 4682726600ce_dummy.py
│       └── 99f6b5224502_add_volume_for_commodities.py
└── alembic/                     # Alembic env for sharesdata DB
```

---

## Tech Stack

| Role | Libraries |
|---|---|
| Web framework | Flask 3.0, Flask-Login, Flask-WTF, Flask-Migrate |
| ORM / DB | SQLAlchemy 2.0, psycopg2-binary, Alembic |
| Deep learning | TensorFlow 2.18, Keras 3.5 |
| Hyperparameter tuning | Optuna 4.2 |
| Market data | yfinance 0.2, fredapi 0.5 |
| Data processing | pandas 2.2, numpy 1.26, scipy 1.14 |
| Technical analysis | pandas_ta 0.3, scikit-learn 1.5 |
| Visualisation | matplotlib 3.9, Pillow 10.4 |
| Reporting | pdfkit 1.0 (wkhtmltopdf), PyPDF2 3.0, Jinja2 3.1 |
| Scheduling | schedule 1.2 |
| Email | smtplib / Gmail SMTP |
| Payment | PayFast REST API (South Africa) |
| Cloud storage | boto3 / AWS S3 |
| GUI | tkinter (Management_GUI.py) |
| Packaging | PyInstaller 6.11 |

---

## Databases

Three PostgreSQL databases share the same server at `192.168.50.138` (internal LAN).

| Database | ORM file | Purpose |
|---|---|---|
| `sharesdata` | `assets/models.py` | Core market data: stocks, OHLCV, runs, commodities, dividends |
| `webapp` | `webapp/models.py` | Web app: subscribers, subscriptions, portfolio, report paths |
| `exceldata` | (raw psycopg2 queries) | Excel/spreadsheet data imports |

### Key tables — sharesdata
- `stocks` — master stock list (code, name, industry, sub-industry)
- `adj_runs` / `close_runs` — daily prediction results (price, volume, RSI, MA, Z-score)
- `commodities` — daily OHLCV for futures tickers
- `dividends` — dividend events per stock

### Key tables — webapp
- `subscribers` — user accounts (email, hashed password, subscription dates, admin flag)
- `subscriptions` — subscription tiers (name, cost, features)
- `subscription_functions` — feature flags per subscription tier
- `portfolio_tracker` — per-user holdings (ticker, weight, comment)
- `portfolio_transaction_history` — buy/sell log
- `htmlwebview` — report file paths per subscriber per run date
- `referals` — referral codes

### Migrations
- `webapp` DB: managed via **Flask-Migrate** (`flask db migrate / upgrade`)
- `sharesdata` DB: managed via **Alembic** (`alembic upgrade head` from `alembic/`)

---

## Configuration

### assets/const.py
Central constants file. Defines:
- `DB_PARAMS` — sharesdata connection dict
- `DB_PARAMS_WEBAPP` — webapp connection dict
- `DB_PARAMS_EXCELDATA` — exceldata connection dict
- `EMAIL_ADDRESS`, `SERVER_ADDRESS`, `SERVER_PORT`, `EMAIL_PASSWORD` — read from env vars with hardcoded fallbacks
- `hparams` — ML hyperparameters (see ML section below)

### assets/config.py
Flask `Config` class. Defines:
- `SECRET_KEY` — session signing key (currently hardcoded)
- `SQLALCHEMY_DATABASE_URI` — built from `DB_PARAMS_WEBAPP`
- PayFast merchant credentials and URLs
- `FLASK_ENV=production` switches PayFast from sandbox to live

### db_config.json
Legacy connection file for the sharesdata DB. Some older scripts read this directly
instead of `const.py`.

### Environment variables (override const.py defaults)
```
EMAIL_ADDRESS      # sender Gmail address
EMAIL_PASSWORD     # Gmail app password
SERVER_ADDRESS     # SMTP host (default smtp.gmail.com)
SERVER_PORT        # SMTP port (default 587)
FLASK_ENV          # set to "production" for live PayFast
```

---

## Running the Application

### ML Report Pipeline (daily job)
```bash
# Run once immediately then schedule at 06:00 daily
python main.py

# Run a single subsystem directly
python close_report.py
python adjusted_close_report.py
```

### Auxiliary Data Pipeline
```bash
# Run once immediately then schedule at 17:10 daily
python data_downloader.py
```

### Flask Web Application
```bash
# Development
flask run --host=0.0.0.0 --port=5000

# Production (example with gunicorn)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Database Migrations (webapp DB)
```bash
flask db migrate -m "description"
flask db upgrade
```

### Database Migrations (sharesdata DB)
```bash
alembic upgrade head
```

### ML Model Training
```bash
python training/close.py
python training/adjusted_close.py
```

---

## ML Pipeline

### Hyperparameters (`assets/const.py → hparams`)
| Parameter | Value | Description |
|---|---|---|
| `HP_LSTM_UNITS` | 600 | LSTM layer size |
| `HP_GRU_UNITS` | 300 | GRU layer size |
| `HP_DROPOUT` | 0.2 | Dropout rate |
| `HP_EPOCHS` | 800 | Max epochs |
| `PATIENCE` | 20 | Early stopping patience |
| `TARGET_ACCURACY` | 90.0 | Target validation accuracy (%) |
| `OPTUNA_TRIALS` | 30 | Hyperparameter search trials |
| `SEQ_LENGTH` | 60 | Input sequence length (trading days) |

### Model Storage
Each trained model lives in `models/{TICKER.JO}/`:
- Keras SavedModel or `.h5` file
- `metadata.json` — records accuracy metrics, training date, best hyperparams

### Data Flow (daily)
1. `yfinance` downloads fresh OHLCV for each ticker in `investment_universe.csv`
2. `MinMaxScaler` normalises price sequences (seq_length=60)
3. Trained LSTM model produces next-week and next-month price predictions
4. `technical_analysis.py` computes RSI (1m/3m/6m), MA24, MA55, Z-score
5. Results written to `adj_runs` / `close_runs` tables
6. Jinja2 renders HTML report; `pdfkit` converts to PDF
7. SMTP sends HTML email with embedded charts to all active subscribers

---

## Coding Conventions

### Python style
- No type hints; plain Python 3 with docstrings mostly absent.
- Logging via `logging.basicConfig` with INFO level; `colorlog` / `colorama` used for
  console output in report scripts.
- Imports: stdlib → third-party → local `assets.*` / `webapp.*`. TensorFlow logging
  suppressed with `os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'` at the top of each ML file.
- Long scripts (`close_report.py`, `adjusted_close_report.py`) are self-contained monoliths
  with a `daily_job()` entry function called by `main.py`.
- Database sessions use a mix of SQLAlchemy ORM (Flask routes, `webapp/models.py`) and
  raw `psycopg2` (report scripts via `assets/database_queries.py`).
- `assets/database_queries.py` is the canonical place for reusable query logic — add new
  queries there rather than inline in report scripts.

### HTML / Jinja2
- Report templates use `{{ variable }}` syntax with Jinja2's `FileSystemLoader`.
- Email templates embed base64-encoded chart images as `<img src="data:image/png;base64,...">`.

### Flask routes (`app.py`)
- All routes in a single file; no Blueprints used.
- CSRF protection enabled globally via `flask_wtf.CSRFProtect`.
- Authentication via `flask_login`; `@login_required` decorator on protected routes.
- PayFast IPN handler at `/payment/ipn` validates MD5 signature before updating subscription.

---

## Security / Credential Hygiene

**Current state — credentials are hardcoded in source:**
- `assets/const.py` — PostgreSQL passwords, Gmail app password
- `assets/config.py` — PayFast merchant key, passphrase, Flask SECRET_KEY
- `db_config.json` — PostgreSQL password

**Do not commit new credentials to source.** The correct pattern (already partially in place
for email) is to read from environment variables with a local `.env` file excluded via
`.gitignore`. The `.gitignore` currently only excludes `data/`, `logs/`, and `*.pyc` — it
should also exclude `.env`.

Recommended remediation (outside this file's scope): move all secrets to env vars and load
via `python-dotenv`.

---

## Git Workflow

- `main` branch holds production code.
- Feature/fix branches are short-lived; commit messages observed follow a free-form style
  (e.g., `"automated backup 2024-08-14 08:06:03"`).
- Large binary artefacts (`models/`, `plots/`, `reports/`, `sharesdata.backup`) are committed
  directly — consider `.gitignore`-ing these or using Git LFS.

---

## Testing

No automated test suite exists. `pytest` is available in the environment.

To manually verify the ML pipeline:
```bash
# Run a single stock through the report job (requires live DB connection)
python close_report.py   # processes all tickers in investment_universe.csv
```

To verify the web app:
```bash
flask run
# Navigate to http://localhost:5000 and test login / report view
```

---

## Common Tasks

### Add a new JSE stock to the universe
1. Append a row to `investment_universe.csv` (ticker, name, industry, sub-industry, market cap).
2. Train a model for the new ticker: `python training/close.py` (edit the ticker list inside).
3. The next daily run of `close_report.py` / `adjusted_close_report.py` will pick it up automatically.

### Add a new subscriber via admin
Use `Management_GUI.py` (desktop GUI) or insert directly into the `subscribers` table in the
`webapp` DB with a hashed password (`werkzeug.security.generate_password_hash`).

### Refresh PostgreSQL materialized views manually
```bash
python -c "from data_downloader import update_materialized_views; update_materialized_views()"
```

### Check TensorBoard training runs
```bash
tensorboard --logdir runs/
```

### Install dependencies
```bash
pip install -r requirements.txt          # full environment
pip install -r webapp_requirements.txt   # Flask web app only
```
