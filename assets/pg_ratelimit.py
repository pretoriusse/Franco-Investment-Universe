"""PostgreSQL storage backend for Flask-Limiter / ``limits``.

``limits`` ships memory/redis/memcached/mongodb/etcd backends but no SQL one.
This adds a ``postgresql://`` (and ``postgres://``) scheme so rate-limit
counters are shared across all gunicorn workers via the existing Postgres
server — no Redis required.

On construction it will, if missing:
  * create the target database (connecting to the ``postgres`` maintenance DB), and
  * create the ``rate_limits`` table.

Fixed-window strategy only (Flask-Limiter's default) — implements the
``incr``/``get``/``get_expiry`` methods that strategy needs.

Importing this module is enough to register the scheme; do it before the
``Limiter`` is created.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from urllib.parse import urlparse, unquote

import psycopg2
from psycopg2 import sql
from psycopg2.pool import ThreadedConnectionPool

from limits.storage import Storage


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS rate_limits (
    key    TEXT PRIMARY KEY,
    count  INTEGER NOT NULL,
    expiry TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rate_limits_expiry ON rate_limits (expiry);
"""

# now() + N seconds, reused in the upsert below.
_INCR = """
INSERT INTO rate_limits (key, count, expiry)
VALUES (%(key)s, %(amount)s, now() + (%(expiry)s || ' seconds')::interval)
ON CONFLICT (key) DO UPDATE SET
    count = CASE WHEN rate_limits.expiry < now()
                 THEN EXCLUDED.count
                 ELSE rate_limits.count + EXCLUDED.count END,
    expiry = CASE WHEN rate_limits.expiry < now() THEN EXCLUDED.expiry
                  WHEN %(elastic)s THEN EXCLUDED.expiry
                  ELSE rate_limits.expiry END
RETURNING count;
"""


def _parse(uri: str) -> dict:
    p = urlparse(uri)
    return {
        "host": p.hostname or "localhost",
        "port": p.port or 5432,
        "user": unquote(p.username) if p.username else None,
        "password": unquote(p.password) if p.password else None,
        "dbname": (p.path or "/ratelimits").lstrip("/") or "ratelimits",
    }


def _ensure_database(params: dict) -> None:
    """Create the target database if it does not exist (idempotent)."""
    admin = dict(params, dbname="postgres")
    try:
        conn = psycopg2.connect(**admin)
    except psycopg2.Error:
        # Can't reach the maintenance DB — assume the target DB already exists
        # and let the table-creation step surface any real connection problem.
        return
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (params["dbname"],)
            )
            if cur.fetchone() is None:
                try:
                    cur.execute(
                        sql.SQL("CREATE DATABASE {}").format(
                            sql.Identifier(params["dbname"])
                        )
                    )
                except psycopg2.Error:
                    # Lost a race with another worker, or the user lacks CREATEDB.
                    # If the DB genuinely doesn't exist, the connection in
                    # __init__ will fail with a clear error; nothing to do here.
                    pass
    finally:
        conn.close()


class PostgresStorage(Storage):
    STORAGE_SCHEME = ["postgresql", "postgres"]

    def __init__(self, uri: str, **options):
        super().__init__(uri, **options)
        self._params = _parse(uri)
        _ensure_database(self._params)
        self._pool = ThreadedConnectionPool(1, 10, **self._params)
        with self._cursor(commit=True) as cur:
            cur.execute(_CREATE_TABLE)

    # -- connection helper ---------------------------------------------------
    @contextmanager
    def _cursor(self, commit: bool = False):
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                yield cur
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    # -- limits.Storage interface -------------------------------------------
    @property
    def base_exceptions(self):
        return psycopg2.Error

    def incr(self, key, expiry, elastic_expiry=False, amount=1):
        with self._cursor(commit=True) as cur:
            cur.execute(
                _INCR,
                {
                    "key": key,
                    "amount": amount,
                    "expiry": int(expiry),
                    "elastic": bool(elastic_expiry),
                },
            )
            return cur.fetchone()[0]

    def get(self, key):
        with self._cursor() as cur:
            cur.execute(
                "SELECT count FROM rate_limits WHERE key = %s AND expiry >= now()",
                (key,),
            )
            row = cur.fetchone()
            return row[0] if row else 0

    def get_expiry(self, key):
        with self._cursor() as cur:
            cur.execute(
                "SELECT EXTRACT(EPOCH FROM expiry) FROM rate_limits WHERE key = %s AND expiry >= now()",
                (key,),
            )
            row = cur.fetchone()
            return float(row[0]) if row else time.time()

    def check(self):
        try:
            with self._cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except psycopg2.Error:
            return False

    def reset(self):
        with self._cursor(commit=True) as cur:
            cur.execute("DELETE FROM rate_limits")
            return cur.rowcount

    def clear(self, key):
        with self._cursor(commit=True) as cur:
            cur.execute("DELETE FROM rate_limits WHERE key = %s", (key,))


if __name__ == "__main__":
    # Self-check against a real PG (set PG_TEST_URI), else skips.
    import os

    uri = os.getenv("PG_TEST_URI")
    if not uri:
        print(
            "set PG_TEST_URI=postgresql://user:pass@host:5432/ratelimits to run the self-check"
        )
    else:
        s = PostgresStorage(uri)
        s.clear("selfcheck")
        assert s.incr("selfcheck", 60) == 1
        assert s.incr("selfcheck", 60) == 2
        assert s.get("selfcheck") == 2
        assert s.get_expiry("selfcheck") > time.time()
        assert s.check() is True
        s.clear("selfcheck")
        assert s.get("selfcheck") == 0
        print("PostgresStorage self-check passed")
