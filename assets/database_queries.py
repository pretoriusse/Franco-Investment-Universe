"""Canonical query layer shared by the report and data pipelines.

All reusable database access for the ``sharesdata`` DB (market data, runs,
commodities, dividends, sentiment) and the ``webapp`` DB (subscribers) lives
here. Every helper opens its own short-lived session, commits or rolls back,
and always closes the session — callers never manage transactions.

Convention: read helpers return a pandas DataFrame (empty on error) or a
scalar/None; write helpers log and roll back on failure instead of raising.
"""
import pandas as pd
import logging
from sqlalchemy import create_engine, text, func, or_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert

from assets.models import TechnicalAnalysis, NewsSentiment
from assets.models import StockDataHistory, ShowCommodities, AdjRuns, ZARUSD, ZARGood, ZARBad, Stock, Industry, SubIndustry, Dividend, Commodity, CloseRuns
from assets.const import DB_PARAMS
try:
    from ..webapp.models import Subscribers
except ImportError:
    from webapp.models import Subscribers
from sqlalchemy.exc import SQLAlchemyError
import numpy as np
from datetime import date
from .const import DB_PARAMS_WEBAPP


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Engine/session for the sharesdata DB (market data)
engine = create_engine(
    f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
)
Session = sessionmaker(bind=engine)

# Engine/session for the webapp DB (subscribers, subscriptions)
webapp_engine = create_engine(
    f"postgresql://{DB_PARAMS_WEBAPP['user']}:{DB_PARAMS_WEBAPP['password']}@{DB_PARAMS_WEBAPP['host']}:{DB_PARAMS_WEBAPP['port']}/{DB_PARAMS_WEBAPP['dbname']}"
)
WebApp_Session = sessionmaker(bind=webapp_engine)

def fetch_stock_universe_from_db():
    """Return the equity universe (commodities excluded) as a DataFrame.

    Columns: code, share_name, industry, sub_industry, rsi_comparison_market,
    rsi_comparison_sector, commodity. Empty DataFrame on error.
    """
    session = Session()
    try:
        stocks = session.query(
            Stock.code,
            Stock.share_name,
            Industry.name.label("industry"),
            SubIndustry.name.label("sub_industry"),
            Stock.rsi_comparison_market,
            Stock.rsi_comparison_sector,
            Stock.commodity
        ).join(Industry, Stock.industry_id == Industry.id)\
        .join(SubIndustry, Stock.sub_industry_id == SubIndustry.id)\
        .filter(Stock.commodity.is_(False))\
        .order_by(Stock.code)\
        .all()
        stock_universe = pd.DataFrame(stocks, columns=['code', 'share_name', 'industry', 'sub_industry', 'rsi_comparison_market', 'rsi_comparison_sector', 'commodity'])
        return stock_universe

    except Exception as e:
        logger.error(f"Error fetching stock universe from DB: {e}")
        return pd.DataFrame()
    finally:
        session.close()

def fetch_stock_and_commodity_universe_from_db():
    """Return the full universe — equities and commodity futures — as a DataFrame."""
    session = Session()
    try:
        stocks = session.query(
            Stock.code,
            Stock.share_name,
            Industry.name.label("industry"),
            SubIndustry.name.label("sub_industry"),
            Stock.rsi_comparison_market,
            Stock.rsi_comparison_sector,
            Stock.commodity
        ).join(Industry, Stock.industry_id == Industry.id)\
         .join(SubIndustry, Stock.sub_industry_id == SubIndustry.id)\
         .order_by(Stock.code)\
         .all()

        stock_universe = pd.DataFrame(stocks, columns=['code', 'share_name', 'industry', 'sub_industry', 'rsi_comparison_market', 'rsi_comparison_sector', 'commodity'])
        return stock_universe

    except Exception as e:
        logger.error(f"Error fetching stock universe from DB: {e}")
        return pd.DataFrame()
    finally:
        session.close()

def fetch_commodity_universe_from_db():
    """Return only the commodity futures rows of the universe as a DataFrame."""
    session = Session()
    try:
        commodities = session.query(
            Stock.code,
            Stock.share_name,
            Industry.name.label("industry"),
            SubIndustry.name.label("sub_industry"),
            Stock.rsi_comparison_market,
            Stock.rsi_comparison_sector,
            Stock.commodity
        ).join(Industry, Stock.industry_id == Industry.id)\
         .join(SubIndustry, Stock.sub_industry_id == SubIndustry.id)\
         .filter(Stock.commodity.is_(True))\
         .all()

        stock_universe = pd.DataFrame(commodities, columns=['code', 'share_name', 'industry', 'sub_industry', 'rsi_comparison_market', 'rsi_comparison_sector', 'commodity'])
        return stock_universe

    except Exception as e:
        logger.error(f"Error fetching stock universe from DB: {e}")
        return pd.DataFrame()
    finally:
        session.close()

def get_ticker_from_db(ticker: str):
    """Return all OHLCV history for *ticker*, oldest first.

    Note: matching uses ``ILIKE %ticker%``, so a short code can match more
    than one ticker (e.g. 'SOL' also matches 'SOLB'). Pass the full code.
    """
    session = Session()
    try:
        ticker_data = session.query(
            StockDataHistory.date,
            StockDataHistory.ticker.label("code"),
            StockDataHistory.open,
            StockDataHistory.high,
            StockDataHistory.low,
            StockDataHistory.close,
            StockDataHistory.volume,
            StockDataHistory.adj_close.label("Adj Close")
        ).filter(StockDataHistory.ticker.ilike(f"%{ticker.replace('%', '')}%"))\
        .order_by(StockDataHistory.date.asc()).all()

        df = pd.DataFrame(ticker_data)
        if df.empty:
            logger.info(f"No data found for {ticker} after fetching from DB.")
            return pd.DataFrame()
        df['Close'] = df['close']
        return df

    except Exception as e:
        logger.error(f"Error fetching ticker data from DB for {ticker}: {e}")
        return pd.DataFrame()
    finally:
        session.close()

def get_ticker_from_db_with_date_select(ticker: str, start_date: str, end_date: str):
    """Return OHLCV history for *ticker* between two ISO dates, oldest first.

    Same ILIKE matching caveat as ``get_ticker_from_db``. Returns an empty
    DataFrame when nothing matches or the query fails.
    """
    session = Session()
    try:
        ticker_data = session.query(
            StockDataHistory.date,
            StockDataHistory.ticker.label("code"),
            StockDataHistory.open,
            StockDataHistory.high,
            StockDataHistory.low,
            StockDataHistory.close,
            StockDataHistory.volume,
            StockDataHistory.adj_close.label("Adj Close")
        ).filter(StockDataHistory.ticker.ilike(f"%{ticker.replace('%', '')}%"))\
        .filter(StockDataHistory.date.between(start_date, end_date))\
        .order_by(StockDataHistory.date.asc()).all()
        
        df = pd.DataFrame(ticker_data)
        if df.empty:
            logger.info(f"No data found for {ticker} after fetching from DB.")
            return pd.DataFrame()
        df['Close'] = df['close']
        return df

    except Exception as e:
        logger.error(f"Error fetching ticker data from DB for {ticker}: {e}")
        return pd.DataFrame()
    finally:
        session.close()

def get_commodities_from_db(ticker: str):
    """Return ZAR-converted OHLCV history for a commodity ticker, oldest first.

    Rows are ordered ascending to match ``get_ticker_from_db*``: consumers
    compute rolling indicators and read ``iloc[-1]`` as "latest", which broke
    when this query previously returned newest-first.
    """
    session = Session()
    try:
        commodity_data = session.query(
            ShowCommodities.date,
            ShowCommodities.ticker.label("code"),
            ShowCommodities.commodity_zar_open.label("open"),
            ShowCommodities.commodity_zar_high.label("high"),
            ShowCommodities.commodity_zar_low.label("low"),
            ShowCommodities.commodity_zar_close.label("close"),
            ShowCommodities.commodity_zar_adj_close.label("Adj Close"),
            ShowCommodities.volume
        ).filter(ShowCommodities.ticker.ilike(f"%{ticker.replace('%', '')}%"))\
        .order_by(ShowCommodities.date.asc()).all()

        df = pd.DataFrame(commodity_data)
        if df.empty:
            logger.info(f"No data found for {ticker} after fetching from DB.")
            return pd.DataFrame()

        return df

    except Exception as e:
        logger.error(f"Error fetching commodities data from DB for {ticker}: {e}")
        return pd.DataFrame()
    finally:
        session.close()

def fetch_latest_date_for_ticker(ticker: str):
    """Return the most recent stored trading date for *ticker*, or None."""
    session = Session()
    try:
        result = session.query(func.max(StockDataHistory.date)).filter(StockDataHistory.ticker == ticker).scalar()
        return result if result else None
    except Exception as e:
        logger.error(f"Error fetching latest date for {ticker}: {e}")
        return None
    finally:
        session.close()

def insert_stock_data_history_batch(batch, on_conflict_update=False):
    """Bulk-insert OHLCV rows into stock_data_history.

    With ``on_conflict_update=True`` an existing (ticker, date) row is
    updated in place (PostgreSQL upsert); otherwise duplicates raise and the
    whole batch is rolled back.
    """
    session = Session()
    try:
        if on_conflict_update:
            # Use PostgreSQL-specific upsert functionality
            insert_stmt = insert(StockDataHistory).values(batch)
            
            # Define how to resolve conflicts: update the rows with new values if conflict occurs
            upsert_stmt = insert_stmt.on_conflict_do_update(
                index_elements=['ticker', 'date'],  # Specify the columns that define a conflict
                set_={  # Define the columns to update on conflict
                    'open': insert_stmt.excluded.open,
                    'high': insert_stmt.excluded.high,
                    'low': insert_stmt.excluded.low,
                    'close': insert_stmt.excluded.close,
                    'volume': insert_stmt.excluded.volume,
                    'adj_close': insert_stmt.excluded.adj_close,
                    'comparison_market': insert_stmt.excluded.comparison_market,
                    'comparison_sector': insert_stmt.excluded.comparison_sector
                }
            )

            # Execute the upsert statement
            session.execute(upsert_stmt)
        else:
            # Standard bulk insert without conflict handling
            session.bulk_insert_mappings(StockDataHistory, batch)

        session.commit()
    except Exception as e:
        logger.error(f"Error inserting stock data history: {e}")
        session.rollback()
    finally:
        session.close()

def update_zar_periods():
    """Rebuild the ZAR good/bad period tables from the overbought/oversold series.

    Walks the ZAR/USD history in date order and collapses consecutive days of
    the same sign into [start, end] periods: positive readings (weakening
    rand) become 'bad' periods, negative ones 'good'. Days at exactly zero
    extend whichever period is currently open.
    """
    session = Session()
    try:
        # Fetch all overbought/oversold values
        zar_data = session.query(ZARUSD.date, ZARUSD.overbought_oversold)\
                        .filter(ZARUSD.overbought_oversold.isnot(None))\
                        .order_by(ZARUSD.date).all()

        current_period: list = []  # [start_date, end_date, 'good'|'bad']
        current_type = None

        for row_date, overbought_oversold in zar_data:
            try:
                if overbought_oversold > 0:
                    if current_type != 'bad':
                        if current_period:
                            insert_period(session, current_period)
                        current_period = [row_date, row_date, 'bad']
                        current_type = 'bad'
                    else:
                        current_period[1] = row_date
                elif overbought_oversold < 0:
                    if current_type != 'good':
                        if current_period:
                            insert_period(session, current_period)
                        current_period = [row_date, row_date, 'good']
                        current_type = 'good'
                    else:
                        current_period[1] = row_date
                else:
                    # Zero reading: extend the currently open period, if any.
                    if current_type in ('bad', 'good'):
                        current_period[1] = row_date
            except Exception as e:
                logger.error(f"Error processing overbought_oversold for date {row_date}: {e}")

        if current_period:
            insert_period(session, current_period)

        session.commit()

    except Exception as e:
        logger.error(f"Error updating ZAR periods: {e}")
        session.rollback()
    finally:
        session.close()

def insert_period(session, period):
    """Merge a [start_date, end_date, type] period into ZARGood or ZARBad."""
    start_date, end_date, period_type = period
    if period_type == 'good':
        period_entry = ZARGood(start_date=start_date, end_date=end_date)
    else:
        period_entry = ZARBad(start_date=start_date, end_date=end_date)
    session.merge(period_entry)
    session.commit()

def fetch_latest_dividend_date(ticker: str):
    """Return the date of the most recent stored dividend for *ticker*, or None."""
    session = Session()
    try:
        latest_date = session.query(Dividend.date).filter(Dividend.ticker == ticker).order_by(Dividend.date.desc()).first()
        if latest_date:
            return latest_date[0]
        return None
    except Exception as e:
        logger.error(f"Error fetching latest dividend date for {ticker}: {e}")
        return None
    finally:
        session.close()

def insert_dividends_batch(batch):
    """Upsert dividend rows keyed on (date, ticker)."""
    session = Session()
    try:
        stmt = insert(Dividend).values(batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=['date', 'ticker'],
            set_=dict(dividend=stmt.excluded.dividend)
        )
        session.execute(stmt)
        session.commit()
    except Exception as e:
        logger.error(f"Error inserting dividends batch: {e}")
        session.rollback()
    finally:
        session.close()

def fetch_latest_commodity_date(ticker):
    """Return the most recent stored date for a commodity ticker, or None."""
    session = Session()
    try:
        result = session.query(func.max(Commodity.date)).filter(Commodity.ticker == ticker).scalar()
        return result if result else None
    except Exception as e:
        logger.error(f"Error fetching latest commodity date for {ticker}: {e}")
        return None
    finally:
        session.close()

def insert_commodities_batch(data_list, on_conflict_update=False):
    """Upsert commodity OHLCV rows keyed on (date, ticker).

    Note: the ``on_conflict_update`` flag is currently ignored — conflicts
    always update the existing row.
    """
    session = Session()
    try:
        # Prepare the insert statement with ON CONFLICT handling
        stmt = insert(Commodity).values(data_list)

        # Update High/Low/Close/Volume if the (date, ticker) row already exists
        stmt = stmt.on_conflict_do_update(
            index_elements=['date', 'ticker'],
            set_=dict(High=stmt.excluded.High, Low=stmt.excluded.Low, Close=stmt.excluded.Close, Volume=stmt.excluded.Volume)
        )
        
        session.execute(stmt)
        session.commit()
        logger.info("Commodities batch inserted successfully!")

    except SQLAlchemyError as e:
        logger.error(f"Error inserting commodities batch: {e}")
        session.rollback()
    finally:
        session.close()

def fetch_latest_date_for_zar(ticker: str):
    """Return the most recent stored ZAR/USD date, or None.

    The *ticker* argument is unused (the table only holds ZAR/USD) but kept
    for backward compatibility with existing callers.
    """
    session = Session()
    try:
        result = session.query(func.max(ZARUSD.date)).scalar()
        return result if result else None
    except Exception as e:
        logger.error(f"Error fetching latest date: {e}")
        return None
    finally:
        session.close()

def insert_zar_usd_batch(batch):
    """Upsert ZAR/USD daily rows keyed on date, dropping records without a date."""
    session = Session()
    try:
        # Clean the batch: Remove any records with null or invalid date values
        cleaned_batch = [record for record in batch if record['date'] and pd.notnull(record['date'])]
        
        # Check if the cleaned batch is empty after filtering
        if not cleaned_batch:
            logger.error("No valid records to insert into ZAR/USD after cleaning. Batch might contain invalid dates.")
            return

        # Prepare the insert statement with conflict handling on the 'date' column
        stmt = insert(ZARUSD).values(cleaned_batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=['date'],  # Ensure 'date' is the unique constraint
            set_={
                'high': stmt.excluded.high,
                'low': stmt.excluded.low,
                'close': stmt.excluded.close,
                'adj_close': stmt.excluded.adj_close,
                'volume': stmt.excluded.volume,
                'open': stmt.excluded.open,
                'overbought_oversold': stmt.excluded.overbought_oversold
            }
        )

        # Execute the insert statement
        session.execute(stmt)
        session.commit()
        logger.info("ZAR/USD data batch inserted/updated successfully!")

    except SQLAlchemyError as e:
        logger.error(f"Error inserting ZAR/USD data: {e}")
        session.rollback()
    finally:
        session.close()

def fetch_all_zar_usd():
    """Return all (date, overbought_oversold) ZAR/USD rows in date order."""
    session = Session()
    try:
        return session.query(ZARUSD.date, ZARUSD.overbought_oversold).filter(ZARUSD.overbought_oversold.isnot(None)).order_by(ZARUSD.date).all()
    except Exception as e:
        logger.error(f"Error fetching ZAR/USD data: {e}")
        return []
    finally:
        session.close()

def insert_zar_good_period(period):
    """Upsert a (start_date, end_date, _) tuple into the ZAR good-period table."""
    session = Session()
    try:
        start_date, end_date, _ = period
        stmt = insert(ZARGood).values(start_date=start_date, end_date=end_date)
        stmt = stmt.on_conflict_do_update(
            index_elements=['start_date'],
            set_={'end_date': stmt.excluded.end_date}
        )
        session.execute(stmt)
        session.commit()
    except Exception as e:
        logger.error(f"Error inserting ZAR good period: {e}")
        session.rollback()
    finally:
        session.close()

def insert_zar_bad_period(period):
    """Upsert a (start_date, end_date, _) tuple into the ZAR bad-period table."""
    session = Session()
    try:
        start_date, end_date, _ = period
        stmt = insert(ZARBad).values(start_date=start_date, end_date=end_date)
        stmt = stmt.on_conflict_do_update(
            index_elements=['start_date'],
            set_={'end_date': stmt.excluded.end_date}
        )
        session.execute(stmt)
        session.commit()
    except Exception as e:
        logger.error(f"Error inserting ZAR bad period: {e}")
        session.rollback()
    finally:
        session.close()

def insert_prediction(date, code, adj_close=None, close=None):
    """Upsert one prediction row keyed on (date, code).

    Passing only one of *adj_close*/*close* preserves the other column's
    existing value (COALESCE in the ON CONFLICT clause), so the close and
    adjusted-close jobs can write independently.
    """
    session = Session()
    try:
        # Convert numpy.float32 to Python float
        adj_close = float(adj_close) if isinstance(adj_close, np.float32) else adj_close # type: ignore
        close = float(close) if isinstance(close, np.float32) else close # type: ignore

        # SQL Query with ON CONFLICT clause
        query = text("""
        INSERT INTO predictions (date, code, adj_close, close)
        VALUES (:date, :code, :adj_close, :close)
        ON CONFLICT (date, code) DO UPDATE 
        SET adj_close = COALESCE(EXCLUDED.adj_close, predictions.adj_close),
            close = COALESCE(EXCLUDED.close, predictions.close)
        RETURNING predictions.id;
        """)
        
        session.execute(query, {'date': date, 'code': code, 'adj_close': adj_close, 'close': close})
        session.commit()
    except SQLAlchemyError as e:
        logger.error(f"Error inserting prediction: {e}")
        session.rollback()
    finally:
        session.close()

def upload_adjusted_close(file_path):
    """Load a daily adjusted-close run CSV and insert its rows into adj_runs.

    The CSV is the snapshot written by ``adjusted_close_report.daily_job``;
    ``run_date`` is stamped with today's date on every row.
    """
    session = Session()
    try:
        # Load data from CSV into DataFrame
        df = pd.read_csv(file_path)

        # Ensure 'run_date' column exists and is set to today's date
        df['run_date'] = pd.to_datetime('today').date()

        # Convert 'commodity' column to boolean if not already in correct type
        df['commodity'] = df['commodity'].astype(bool)

        # Round numeric fields to 2 decimals
        numeric_columns = [
            'Current Price', 'Current Value', 'Next Week Prediction', 'Next Month Prediction',
            'Z-Score', 'Overbought_Oversold', 'Overbought_Oversold_Value', 'MA24', 'MA55',
            'SECTOR RSI 1M', 'SECTOR RSI 3M', 'SECTOR RSI 6M', 'MARKET RSI 1M', 'MARKET RSI 3M', 'MARKET RSI 6M'
        ]
        df[numeric_columns] = df[numeric_columns].round(2)

        # Convert the DataFrame into a list of dictionaries
        records = df.to_dict(orient='records')

        # Map records to AdjRuns objects
        adj_runs_list = [AdjRuns(
            code=record['code'],
            share_name=record['share_name'],
            industry=record['industry'],
            sub_industry=record.get('sub_industry'),
            rsi_comparison_sector=record.get('rsi_comparison_sector'),
            rsi_comparison_market=record.get('rsi_comparison_market'),
            commodity=record.get('commodity'),
            current_price=record.get('Current Price'),
            current_value=record.get('Current Value'),
            next_week_prediction=record.get('Next Week Prediction'),
            next_month_prediction=record.get('Next Month Prediction'),
            z_score=record.get('Z-Score'),
            overbought_oversold=record.get('Overbought_Oversold'),
            overbought_oversold_value=record.get('Overbought_Oversold_Value'),
            ma24=record.get('MA24'),
            ma55=record.get('MA55'),
            sector_rsi_1m=record.get('SECTOR RSI 1M'),
            sector_rsi_3m=record.get('SECTOR RSI 3M'),
            sector_rsi_6m=record.get('SECTOR RSI 6M'),
            market_rsi_1m=record.get('MARKET RSI 1M'),
            market_rsi_3m=record.get('MARKET RSI 3M'),
            market_rsi_6m=record.get('MARKET RSI 6M'),
            run_date=record['run_date']
        ) for record in records]

        # Insert the records into the database
        session.bulk_save_objects(adj_runs_list)
        session.commit()
        logger.info("Data successfully uploaded to AdjRuns table.")

    except SQLAlchemyError as e:
        logger.error(f"Error uploading adjusted close data: {e}")
        session.rollback()

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        session.rollback()

    finally:
        session.close()

def upload_close_runs(file_path):
    """Load a daily close run CSV and insert its rows into close_runs.

    Counterpart of ``upload_adjusted_close`` for ``close_report.daily_job``.
    """
    session = Session()
    try:
        # Load data from CSV into DataFrame
        df = pd.read_csv(file_path)

        # Ensure the run_date column exists and is set to today's date
        df['run_date'] = pd.to_datetime('today').date()

        # Convert the DataFrame into a list of dictionaries
        records = df.to_dict(orient='records')

        # Map records to CloseRuns objects
        close_runs_list = [CloseRuns(
            code=record['code'],
            share_name=record['share_name'],
            industry=record['industry'],
            sub_industry=record.get('sub_industry'),  # Handle optional fields
            rsi_comparison_market=record.get('rsi_comparison_market'),
            rsi_comparison_sector=record.get('rsi_comparison_sector'),
            commodity=record.get('commodity', False),  # Default to False if not specified
            current_price=record.get('Current Price'),
            current_value=record.get('Current Value'),
            next_week_prediction=record.get('Next Week Prediction'),
            next_month_prediction=record.get('Next Month Prediction'),
            z_score=record.get('Z-Score'),
            overbought_oversold=record.get('Overbought_Oversold'),
            overbought_oversold_value=record.get('Overbought_Oversold_Value'),
            ma24=record.get('MA24'),
            ma55=record.get('MA55'),
            sector_rsi_1m=record.get('SECTOR RSI 1M'),
            sector_rsi_3m=record.get('SECTOR RSI 3M'),
            sector_rsi_6m=record.get('SECTOR RSI 6M'),
            market_rsi_1m=record.get('MARKET RSI 1M'),
            market_rsi_3m=record.get('MARKET RSI 3M'),
            market_rsi_6m=record.get('MARKET RSI 6M'),
            run_date=record['run_date']
        ) for record in records]

        # Insert the records into the database
        session.bulk_save_objects(close_runs_list)
        session.commit()
        logger.info("Data successfully uploaded to CloseRuns table.")

    except SQLAlchemyError as e:
        logger.error(f"Error uploading close runs data: {e}")
        session.rollback()

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        session.rollback()

    finally:
        session.close()

def close_session():
    """Legacy no-op kept for callers that still invoke it.

    Every helper in this module opens and closes its own session, so there is
    no shared session to close; this creates a fresh one and closes it again.
    """
    session = Session()
    session.close()

def fetch_active_subscribers():
    """Return Subscribers whose subscription is paid and not yet expired."""
    session = WebApp_Session()
    try:
        # Query for subscribers where subscription is paid and the expiration date is in the future or today
        active_subscribers = session.query(Subscribers).filter(
            Subscribers.subscription_paid == True,
            Subscribers.subscription_expiration_date >= date.today()
        ).all()

        # Log the number of active subscribers found
        logger.info(f"Found {len(active_subscribers)} active subscribers.")
        
        return active_subscribers

    except SQLAlchemyError as e:
        logger.error(f"Error fetching active subscribers: {e}")
        return []
    finally:
        session.close()

def update_subscriber(subscriber_id, update_data):
    """
    Update a subscriber's details based on their ID.
    
    Parameters:
    subscriber_id (int): The ID of the subscriber to update.
    update_data (dict): A dictionary of fields to update, e.g., {'subscription_paid': False, 'subscription_expiration_date': new_date}.
    
    Returns:
    bool: True if the update was successful, False otherwise.
    """
    session = WebApp_Session()
    try:
        # Find the subscriber by their ID
        subscriber = session.query(Subscribers).filter_by(id=subscriber_id).first()

        if not subscriber:
            logger.error(f"Subscriber with ID {subscriber_id} not found.")
            return False

        # Update the subscriber with the provided data
        for key, value in update_data.items():
            if hasattr(subscriber, key):
                setattr(subscriber, key, value)
            else:
                logger.warning(f"Invalid field {key} provided for update.")

        # Commit the changes to the database
        session.commit()

        logger.info(f"Subscriber with ID {subscriber_id} updated successfully.")
        return True

    except SQLAlchemyError as e:
        logger.error(f"Error updating subscriber with ID {subscriber_id}: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def insert_technical_analysis_batch(batch):
    """Upsert daily technical-analysis rows keyed on (ticker, date)."""
    session = Session()
    try:
        stmt = insert(TechnicalAnalysis).values(batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=['ticker', 'date'],
            set_={
                'signal': stmt.excluded.signal,
                'action': stmt.excluded.action,
                'close': stmt.excluded.close,
                'sma_22': stmt.excluded.sma_22,
                'sma_55': stmt.excluded.sma_55,
                'rsi_1m': stmt.excluded.rsi_1m,
                'rsi_3m': stmt.excluded.rsi_3m,
                'rsi_6m': stmt.excluded.rsi_6m,
                'bollinger_high': stmt.excluded.bollinger_high,
                'bollinger_low': stmt.excluded.bollinger_low,
                'bollinger_mid': stmt.excluded.bollinger_mid,
                'bollinger_perc_b': stmt.excluded.bollinger_perc_b,
                'bollinger_width': stmt.excluded.bollinger_width,
                'overbaughtoversold': stmt.excluded.overbaughtoversold,
                'zscore': stmt.excluded.zscore
            }
        )
        session.execute(stmt)
        session.commit()
        logger.info("Technical analysis data batch inserted/updated successfully!")
    except SQLAlchemyError as e:
        logger.error(f"Error inserting technical analysis batch: {e}")
        session.rollback()
    finally:
        session.close()


def insert_sentiment_batch(batch: list[dict]) -> None:
    """Upsert a list of daily sentiment records into news_sentiment."""
    session = Session()
    try:
        stmt = insert(NewsSentiment).values(batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=['ticker', 'date'],
            set_={
                'sentiment_score': stmt.excluded.sentiment_score,
                'article_count': stmt.excluded.article_count,
                'positive_count': stmt.excluded.positive_count,
                'negative_count': stmt.excluded.negative_count,
                'neutral_count': stmt.excluded.neutral_count,
            }
        )
        session.execute(stmt)
        session.commit()
    except SQLAlchemyError as e:
        logger.error(f"Error inserting sentiment batch: {e}")
        session.rollback()
    finally:
        session.close()


def fetch_sentiment_for_ticker(ticker: str, days: int = 60) -> 'pd.DataFrame':
    """Return the last *days* of sentiment rows for *ticker* ordered ascending.

    Columns: date, sentiment_score, article_count.
    Returns an empty DataFrame if no records exist yet.
    """
    session = Session()
    try:
        from datetime import date as date_cls, timedelta
        cutoff = date_cls.today() - timedelta(days=days)
        rows = (
            session.query(
                NewsSentiment.date,
                NewsSentiment.sentiment_score,
                NewsSentiment.article_count,
            )
            .filter(NewsSentiment.ticker == ticker, NewsSentiment.date >= cutoff)
            .order_by(NewsSentiment.date.asc())
            .all()
        )
        return pd.DataFrame(rows, columns=['date', 'sentiment_score', 'article_count'])
    except Exception as e:
        logger.error(f"Error fetching sentiment for {ticker}: {e}")
        return pd.DataFrame()
    finally:
        session.close()


def get_latest_sentiment_score(ticker: str) -> float:
    """Return the most recent sentiment score for *ticker*, or 0.0 if none."""
    session = Session()
    try:
        row = (
            session.query(NewsSentiment.sentiment_score)
            .filter(NewsSentiment.ticker == ticker)
            .order_by(NewsSentiment.date.desc())
            .first()
        )
        return float(row[0]) if row else 0.0
    except Exception as e:
        logger.error(f"Error fetching latest sentiment for {ticker}: {e}")
        return 0.0
    finally:
        session.close()
