import pandas as pd
import logging
from sqlalchemy import create_engine, text, func, or_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert

from assets.models import TechnicalAnalysis
from assets.models import StockDataHistory, ShowCommodities, AdjRuns, ZARUSD, ZARGood, ZARBad, Stock, Industry, SubIndustry, Dividend, Commodity, CloseRuns
from assets.const import DB_PARAMS
try:
    from ..webapp.models import Subscribers
except ImportError:
    from webapp.models import Subscribers
from sqlalchemy.exc import SQLAlchemyError
import numpy as np
from sqlalchemy.dialects.postgresql import insert as pg_insert
from datetime import date
from .const import DB_PARAMS_WEBAPP


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure the database engine and session
engine = create_engine(
    f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
)
Session = sessionmaker(bind=engine)

webapp_engine = create_engine(
    f"postgresql://{DB_PARAMS_WEBAPP['user']}:{DB_PARAMS_WEBAPP['password']}@{DB_PARAMS_WEBAPP['host']}:{DB_PARAMS_WEBAPP['port']}/{DB_PARAMS_WEBAPP['dbname']}"
)
WebApp_Session = sessionmaker(bind=webapp_engine)

def fetch_stock_universe_from_db():
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
        df['Close'] = df['close']
        if df.empty:
            logger.info(f"No data found for {ticker} after fetching from DB.")
            return pd.DataFrame()

        return df

    except Exception as e:
        logger.error(f"Error fetching ticker data from DB for {ticker}: {e}")
        return pd.DataFrame()
    finally:
        session.close()

def get_ticker_from_db_with_date_select(ticker: str, start_date: str, end_date: str):
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
        df['Close'] = df['close']
        if df.empty:
            logger.info(f"No data found for {ticker} after fetching from DB.")
            return pd.DataFrame()

        return df

    except Exception as e:
        logger.error(f"Error fetching ticker data from DB for {ticker}: {e}")
        return pd.DataFrame()
    finally:
        session.close()

def get_commodities_from_db(ticker: str):
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
        .order_by(ShowCommodities.date.desc()).all()

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
    session = Session()
    try:
        if on_conflict_update:
            # Use PostgreSQL-specific upsert functionality
            insert_stmt = pg_insert(StockDataHistory).values(batch)
            
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
    session = Session()
    try:
        # Fetch all overbought/oversold values
        zar_data = session.query(ZARUSD.date, ZARUSD.overbought_oversold)\
                        .filter(ZARUSD.overbought_oversold.isnot(None))\
                        .order_by(ZARUSD.date).all()

        current_period:list = []
        current_type = None

        for date, overbought_oversold in zar_data:
            try:
                if overbought_oversold > 0:
                    if current_type != 'bad':
                        if current_period:
                            insert_period(session, current_period)
                        current_period = [date, date, 'bad']
                        current_type = 'bad'
                    else:
                        current_period[1] = date
                elif overbought_oversold < 0:
                    if current_type != 'good':
                        if current_period:
                            insert_period(session, current_period)
                        current_period = [date, date, 'good']
                        current_type = 'good'
                    else:
                        current_period[1] = date
                else:
                    if current_type == 'bad':
                        current_period[1] = date
                    elif current_type == 'good':
                        current_period[1] = date
            except Exception as e:
                logger.error(f"Error processing overbought_oversold for date {date}: {e}")

        if current_period:
            insert_period(session, current_period)

        session.commit()

    except Exception as e:
        logger.error(f"Error updating ZAR periods: {e}")
        session.rollback()
    finally:
        session.close()

def insert_period(session, period):
    start_date, end_date, period_type = period
    if period_type == 'good':
        period_entry = ZARGood(start_date=start_date, end_date=end_date)
    else:
        period_entry = ZARBad(start_date=start_date,         end_date=end_date)
    session.merge(period_entry)
    session.commit()

def fetch_latest_dividend_date(ticker: str):
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
    session = Session()
    try:
        result = session.query(func.max(Commodity.date)).scalar()
        return result if result else None
    except Exception as e:
        logger.error(f"Error fetching latest date: {e}")
        return None
    finally:
        session.close()

def insert_commodities_batch(data_list, on_conflict_update=False):
    session = Session()
    try:
        # Prepare the insert statement with ON CONFLICT handling
        stmt = insert(Commodity).values(data_list)
        
        # Handling duplicates by ignoring if the date and ticker already exist
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
    session = Session()
    try:
        return session.query(ZARUSD.date, ZARUSD.overbought_oversold).filter(ZARUSD.overbought_oversold.isnot(None)).order_by(ZARUSD.date).all()
    except Exception as e:
        logger.error(f"Error fetching ZAR/USD data: {e}")
        return []

def insert_zar_good_period(period):
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
    session = Session()
    session.close()

def fetch_active_subscribers():
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
