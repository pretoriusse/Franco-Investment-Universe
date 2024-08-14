import pandas as pd
import logging
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import Column, Date, String, Float, UniqueConstraint
from assets.models import StockDataHistory, ShowCommodities, AdjRuns, ZARUSD, ZARGood, ZARBad, Stock, Industry, SubIndustry, Dividend, Commodity, Prediction
from assets.const import DB_PARAMS
from sqlalchemy.exc import SQLAlchemyError
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure the database engine and session
engine = create_engine(
    f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
)
Session = sessionmaker(bind=engine)
session = Session()

# Fetch commodity data
def get_commodities_from_db(ticker: str):
    query = """
        SELECT date,
            ticker AS code,
            commodity_zar_open AS open,
            commodity_zar_high AS high,
            commodity_zar_low AS low,
            commodity_zar_close AS close,
            commodity_zar_adj_close AS "Adj Close",
            Volume
        FROM show_commodities
        WHERE ticker ILIKE :ticker
        ORDER BY date DESC;
    """
    try:
        df = pd.read_sql_query(text(query), engine, params={"ticker": f'%{ticker.replace("%", "")}%'})
        logger.info(f"Commodity data successfully fetched for {ticker}.")
        return df
    except Exception as e:
        logger.error(f"Error fetching commodity data for {ticker}: {e}")
        return pd.DataFrame()

def fetch_stock_universe_from_db():
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
        print(f"Error fetching stock universe from DB: {e}")
        return pd.DataFrame()

def fetch_stock_and_commodity_universe_from_db():
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
        print(f"Error fetching stock universe from DB: {e}")
        return pd.DataFrame()

def fetch_commodity_universe_from_db():
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
        print(f"Error fetching stock universe from DB: {e}")
        return pd.DataFrame()

def get_ticker_from_db(ticker: str):
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
            print(f"No data found for {ticker} after fetching from DB.")
            return pd.DataFrame()

        return df

    except Exception as e:
        print(f"Error fetching ticker data from DB for {ticker}: {e}")
        return pd.DataFrame()

def get_ticker_from_db_with_date_select(ticker: str, start_date: str, end_date: str):
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
            print(f"No data found for {ticker} after fetching from DB.")
            return pd.DataFrame()

        return df

    except Exception as e:
        print(f"Error fetching ticker data from DB for {ticker}: {e}")
        return pd.DataFrame()

def get_commodities_from_db(ticker: str):
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
            print(f"No data found for {ticker} after fetching from DB.")
            return pd.DataFrame()

        return df

    except Exception as e:
        print(f"Error fetching commodities data from DB for {ticker}: {e}")
        return pd.DataFrame()

def fetch_latest_date_for_ticker(ticker: str):
    try:
        result = session.query(func.max(StockDataHistory.date)).filter(StockDataHistory.ticker == ticker).scalar()
        return result if result else None
    except Exception as e:
        print(f"Error fetching latest date for {ticker}: {e}")
        return None

def insert_stock_data_history_batch(batch):
    try:
        session.bulk_insert_mappings(StockDataHistory, batch)
        session.commit()
    except Exception as e:
        print(f"Error inserting stock data history: {e}")
        session.rollback()

def update_zar_periods():
    try:
        # Fetch all overbought/oversold values
        zar_data = session.query(ZARUSD.date, ZARUSD.overbought_oversold)\
                          .filter(ZARUSD.overbought_oversold.isnot(None))\
                          .order_by(ZARUSD.date).all()

        current_period = None
        current_type = None

        for date, overbought_oversold in zar_data:
            try:
                if overbought_oversold > 0:
                    if current_type != 'bad':
                        if current_period:
                            insert_period(current_period)
                        current_period = [date, date, 'bad']
                        current_type = 'bad'
                    else:
                        current_period[1] = date
                elif overbought_oversold < 0:
                    if current_type != 'good':
                        if current_period:
                            insert_period(current_period)
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
                print(f"Error processing overbought_oversold for date {date}: {e}")

        if current_period:
            insert_period(current_period)

        session.commit()

    except Exception as e:
        print(f"Error updating ZAR periods: {e}")
        session.rollback()

def insert_period(period):
    start_date, end_date, period_type = period
    if period_type == 'good':
        period_entry = ZARGood(start_date=start_date, end_date=end_date)
    else:
        period_entry = ZARBad(start_date=start_date, end_date=end_date)
    session.merge(period_entry)
    session.commit()

def fetch_latest_dividend_date(ticker: str):
    try:
        latest_date = session.query(Dividend.date).filter(Dividend.ticker == ticker).order_by(Dividend.date.desc()).first()
        if latest_date:
            return latest_date[0]
        return None
    except Exception as e:
        logger.error(f"Error fetching latest dividend date for {ticker}: {e}")
        return None

def insert_dividends_batch(batch):
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

def fetch_latest_commodity_date(ticker):
    """Fetch the latest date available in the zar_usd table."""
    try:
        result = session.query(func.max(Commodity.date)).scalar()
        return result if result else None
    except Exception as e:
        logger.error(f"Error fetching latest date: {e}")
        return None

def insert_commodities_batch(data_list):
    session = Session()
    try:
        # Your insert logic here
        session.bulk_insert_mappings(..., data_list)
        session.commit()
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()

def fetch_latest_date_for_zar(ticker: str):
    """Fetch the latest date available in the zar_usd table."""
    try:
        result = session.query(func.max(ZARUSD.date)).scalar()
        return result if result else None
    except Exception as e:
        logger.error(f"Error fetching latest date: {e}")
        return None

def insert_zar_usd_batch(batch):
    """Insert a batch of ZAR/USD data into the zar_usd table."""
    try:
        session.bulk_insert_mappings(ZARUSD, batch)
        session.commit()
    except Exception as e:
        logger.error(f"Error inserting ZAR/USD data: {e}")
        session.rollback()

def fetch_all_zar_usd():
    """Fetch all ZAR/USD data to calculate periods."""
    try:
        return session.query(ZARUSD.date, ZARUSD.overbought_oversold)\
                      .filter(ZARUSD.overbought_oversold.isnot(None))\
                      .order_by(ZARUSD.date).all()
    except Exception as e:
        logger.error(f"Error fetching ZAR/USD data: {e}")
        return []

def insert_zar_good_period(period):
    """Insert a period into the zar_good table."""
    start_date, end_date, _ = period  # Ignore the third value (period type)
    try:
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

def insert_zar_bad_period(period):
    """Insert a period into the zar_bad table."""
    start_date, end_date, _ = period  # Ignore the third value (period type)
    try:
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

def insert_prediction(date, code, adj_close=None, close=None):
    try:
        # Convert numpy.float32 to Python float
        adj_close = float(adj_close) if isinstance(adj_close, np.float32) else adj_close
        close = float(close) if isinstance(close, np.float32) else close

        # SQL Query with ON CONFLICT clause
        query = text("""
        INSERT INTO predictions (date, code, adj_close, close)
        VALUES (:date, :code, :adj_close, :close)
        ON CONFLICT (date, code) DO UPDATE 
        SET adj_close = COALESCE(EXCLUDED.adj_close, predictions.adj_close),
            close = COALESCE(EXCLUDED.close, predictions.close)
        RETURNING predictions.id;
        """)
        
        # Assuming you have a session and engine set up
        session.execute(query, {'date': date, 'code': code, 'adj_close': adj_close, 'close': close})
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Error inserting prediction: {e}")
    finally:
        session.close()

def close_session():
    session.close()