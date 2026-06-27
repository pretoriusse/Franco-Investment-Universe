# Test Coverage Analysis

## Starting point

Before this change the repository had **zero automated tests**: no `tests/`
directory, no `pytest.ini`/`pyproject.toml` test config, and no CI workflow.
`pytest` was listed in `CLAUDE.md` as merely "available in the environment."
Every module mixes business logic with I/O (Postgres, yfinance, SMTP,
PayFast, TensorFlow/Keras) at the top of the call stack, which is the main
reason no tests existed — there was no seam to unit-test through.

## What this change adds

A `tests/` suite (33 tests, all passing) covering the modules that have
genuine pure-logic seams, plus the scaffolding to grow it:

| File | Covers |
|---|---|
| `tests/test_payments_signature.py` | `assets/payments.py::generate_signature` — PayFast MD5 signature: field ordering, URL-encoding, passphrase handling, empty-field skipping |
| `tests/test_technical_analysis.py` | `assets/technical_analysis.py::calculate_rsi` / `calculate_z_score` — known values plus the unguarded NaN/∞ edge cases on flat/monotonic series |
| `tests/test_news_sentiment.py` | `assets/news_sentiment.py` — VADER scoring, financial-lexicon boosts, recency-weighted aggregation, yfinance boundary mocked via `monkeypatch` |
| `tests/test_webapp_models.py` | `webapp/models.py::Subscribers.is_active` / Flask-Login interface — expiry and blacklist logic |
| `tests/test_webapp_forms.py` | `webapp/forms.py::LoginForm` / `RegistrationForm` — validation rules, against an in-memory SQLite DB via the `flask_app` fixture in `tests/conftest.py` |

Run them with:
```bash
pip install -r requirements-dev.txt
pytest
```
`.github/workflows/tests.yml` now runs this suite on every push/PR.

## Findings surfaced while auditing for testability

These aren't fixed here (out of scope for a test-coverage pass) but should
be tracked, since covering them properly will require resolving them first:

1. **Two divergent PayFast signature implementations.** `assets/payments.py:6` builds
   the signature from PayFast's documented fixed field order with `quote_plus`
   URL-encoding; `app.py:57` (`generate_signature`) instead alphabetically
   sorts the dict and does *no* URL-encoding. `app.py`'s version is the one
   actually used to verify inbound IPN signatures (`app.py:326`) and to sign
   outbound subscription requests (`app.py:293`) — `assets/payments.py` looks
   unused dead code, but if it's a leftover from a previous IPN implementation
   it's worth confirming `app.py`'s version actually matches what PayFast
   sends before trusting the IPN signature check in production.
2. **`email_validator` is missing from `requirements.txt`.** WTForms' `Email()`
   validator (`webapp/forms.py:8,21`) imports it lazily on first validation —
   in a clean environment this raises `ImportError` the first time anyone
   submits the login/registration form. Added to `requirements-dev.txt` here
   so the new tests can run; it should also be added to `requirements.txt`.
3. **RSI/Z-score divide-by-zero is unguarded.** A flat price series produces
   `NaN` RSI (`0/0`), and a strictly one-directional series saturates at
   exactly `0` or `100` via `x/0 = inf`. `tests/test_technical_analysis.py`
   pins this as current behavior, but it's worth deciding whether `close_report.py`'s
   own inline RSI implementation (it doesn't import `assets/technical_analysis.py`'s
   functions, despite the docstring suggesting it should) handles this the same way.

## Remaining gaps, in priority order

### 1. `assets/database_queries.py` (850 lines, the only query layer — currently 0% covered)
Every report job and the Flask app funnel through this module
(`fetch_stock_universe_from_db`, `insert_stock_data_history_batch`,
`fetch_active_subscribers`, `upload_adjusted_close`/`upload_close_runs`,
etc.). It's the highest-leverage place to add coverage because a bug here
breaks every downstream consumer at once.
- **Approach**: point `engine`/`webapp_engine` at a throwaway SQLite or
  ephemeral Postgres (e.g. `testcontainers-python`) in a fixture, run the
  real `CREATE TABLE` DDL (or a trimmed subset) against it, and exercise the
  batch-insert/upsert functions (`insert_stock_data_history_batch`,
  `insert_dividends_batch`, `insert_zar_usd_batch`) for "insert new row" and
  "update existing row" (`on_conflict_update=True`) paths — these are exactly
  the cases that are easy to get backwards and hard to notice in production
  until historical data gets silently overwritten or silently skipped.

### 2. `app.py` — Flask routes (0% covered)
Nothing here is tested, including security-sensitive paths:
- **`/payment/ipn`** (`app.py:318-361`): the most important route to cover.
  Test that a tampered signature is rejected (`400`), a missing
  `m_payment_id` is rejected, an unknown user returns `404` without raising,
  and a `COMPLETE` status flips `subscription_paid` while a non-`COMPLETE`
  status takes no action. Use `app.test_client()` plus a monkeypatched
  `Session`/in-memory DB, the same pattern as `tests/conftest.py::flask_app`.
- **`/register`, `/login`**: duplicate-ID-number rejection (`app.py:174-178`),
  the special-case free-subscription email allowlist (`app.py:191`), and the
  "subscription not paid → redirect to pay_subscription" branch (`app.py:236-239`).
- **`/disable-user/<id>`** (`app.py:511-527`): confirm non-admins get `403`
  and admins can't be blacklisted — this is an authorization boundary and
  currently has no regression protection at all.

### 3. ML report pipeline — `close_report.py`, `adjusted_close_report.py`, `training/*.py` (~5,000 lines combined, 0% covered)
These are self-contained monoliths mixing data loading, indicator
calculation, model training/inference, chart rendering, and email/PDF
generation in one `daily_job()` function each — there's no seam to unit-test
without refactoring. Two pragmatic options, roughly in order of effort:
- **Extract pure helpers first.** `assets/news_sentiment.py` already shows
  the pattern this codebase should follow: keep indicator math, prediction
  post-processing, and report-row assembly in small functions that take
  DataFrames/values and return DataFrames/values, with DB/network/model I/O
  pushed to thin call sites. Each extracted helper becomes trivially
  unit-testable.
- **One smoke test per pipeline**, independent of the refactor: run
  `daily_job`-equivalent logic for a single synthetic ticker with a tiny
  hand-built price series and 1-2 training epochs, and assert it produces a
  prediction of the right shape/type without raising — this catches breakage
  from dependency upgrades (TensorFlow/Keras especially) even with zero
  insight into model accuracy.

### 4. Auxiliary data pipeline — `assets/zar_process.py`, `assets/upload_history.py`, `assets/fetch_daily_commodity_data.py`, `assets/dividends.py`, `data_downloader.py` (0% covered)
Same DB-coupled shape as `database_queries.py`. Lowest-hanging fruit here is
any date/period-gap-filling logic (e.g. `update_zar_periods` /
`insert_zar_good_period`/`insert_zar_bad_period` in `database_queries.py`,
used by `zar_process.py`) — gap detection logic is exactly the kind of
off-by-one-prone code that benefits most from unit tests with synthetic
date ranges.

### 5. `assets/get_best_metrics_for_valuation.py`
`process_data` (imputation/outlier handling) and `define_time_windows`
(sliding-window date math) are pure-ish — `process_data` takes/returns a
DataFrame, `define_time_windows` only needs `datetime.now()` mocked via
`freezegun` or dependency injection. Good next target after the database
layer; not included in this PR to keep scope to what's already verified
importable without `scikit-learn`/`seaborn` in the test environment.

### 6. CI coverage gate
`.github/workflows/tests.yml` (added here) runs the suite but does not yet
enforce a coverage floor. Once gaps #1-#2 are partially filled, consider
adding `pytest-cov` with a minimum threshold on the modules that *can*
reasonably reach high coverage (`assets/news_sentiment.py`,
`assets/payments.py`, `assets/technical_analysis.py`, `webapp/`) rather than
a single repo-wide number that the ML pipeline will always drag down.

## Suggested order of work

1. `assets/database_queries.py` batch-insert/upsert functions (#1) — highest
   blast radius if wrong, most testable with a fixture DB.
2. `/payment/ipn` and `/disable-user` in `app.py` (#2) — security-sensitive,
   currently fully unguarded.
3. Extract + test pure helpers out of `close_report.py`/`adjusted_close_report.py`
   following the `news_sentiment.py` pattern (#3).
4. Round out `assets/zar_process.py` / `upload_history.py` /
   `fetch_daily_commodity_data.py` / `dividends.py` (#4) and
   `get_best_metrics_for_valuation.py` (#5) using the same fixture-DB pattern
   established in step 1.
