from sqlalchemy import Column, Integer, String, Float, Date, BigInteger, Numeric, ForeignKey, Boolean, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class AdjRuns(Base):
    __tablename__ = 'adj_runs'

    id = Column(Integer, primary_key=True)
    code = Column(String(10), nullable=False)
    share_name = Column(String, nullable=False)
    industry = Column(String, nullable=False)
    shares = Column(Numeric)
    weight = Column(Numeric)
    market_cap = Column(Numeric)
    closing = Column(Numeric)
    rsi_comparison_sector = Column(String)
    rsi_comparison_market = Column(String)
    current_price = Column(Numeric)
    next_week_prediction = Column(Numeric)
    next_month_prediction = Column(Numeric)
    next_week_volume_prediction = Column(Numeric)
    next_month_volume_prediction = Column(Numeric)
    z_score = Column(Numeric)
    overbought_oversold = Column(Numeric)
    sector_rsi_1m = Column(Numeric)
    sector_rsi_3m = Column(Numeric)
    sector_rsi_6m = Column(Numeric)
    market_rsi_1m = Column(Numeric)
    market_rsi_3m = Column(Numeric)
    market_rsi_6m = Column(Numeric)
    run_date = Column(Date, nullable=False)

class CloseRuns(Base):
    __tablename__ = 'close_runs'

    id = Column(Integer, primary_key=True)
    code = Column(String(10), nullable=False)
    share_name = Column(String, nullable=False)
    industry = Column(String, nullable=False)
    shares = Column(Numeric)
    weight = Column(Numeric)
    market_cap = Column(Numeric)
    closing = Column(Numeric)
    rsi_comparison_sector = Column(String)
    rsi_comparison_market = Column(String)
    current_price = Column(Numeric)
    next_week_prediction = Column(Numeric)
    next_month_prediction = Column(Numeric)
    next_week_volume_prediction = Column(Numeric)
    next_month_volume_prediction = Column(Numeric)
    z_score = Column(Numeric)
    overbought_oversold = Column(Numeric)
    sector_rsi_1m = Column(Numeric)
    sector_rsi_3m = Column(Numeric)
    sector_rsi_6m = Column(Numeric)
    market_rsi_1m = Column(Numeric)
    market_rsi_3m = Column(Numeric)
    market_rsi_6m = Column(Numeric)
    run_date = Column(Date, nullable=False)

class Commodity(Base):
    __tablename__ = 'commodities'

    date = Column(Date, primary_key=True, nullable=False)
    ticker = Column(String(10), primary_key=True, nullable=False)
    Open = Column(Float, nullable=True)
    High = Column(Float, nullable=True)
    Low = Column(Float, nullable=True)
    Close = Column(Float, nullable=True)
    AdjClose = Column(Float, nullable=True)

    def __repr__(self):
        return f"<Commodity(date={self.date}, ticker={self.ticker}, Open={self.Open}, High={self.High}, Low={self.Low}, Close={self.Close}, AdjClose={self.AdjClose})>"

class Dividend(Base):
    __tablename__ = 'dividends'

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    ticker = Column(String, nullable=False)
    dividend = Column(Float, nullable=False)

class Industry(Base):
    __tablename__ = 'industries'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

    sub_industries = relationship("SubIndustry", back_populates="industry", cascade="all, delete-orphan")
    stocks = relationship("Stock", back_populates="industry", cascade="all, delete-orphan")

class RSIComparisonSector(Base):
    __tablename__ = 'rsi_comparison_sectors'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

class RSIComparisonMarket(Base):
    __tablename__ = 'rsi_comparison_markets'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

class Prediction(Base):
    __tablename__ = 'predictions'
    __table_args__ = (UniqueConstraint('date', 'code', name='_date_code_uc'),)
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    code = Column(String, nullable=False)
    adj_close = Column(Float, nullable=True)
    close = Column(Float, nullable=True)

    def __init__(self, date, code, adj_close=None, close=None):
        if adj_close is None and close is None:
            raise ValueError("Either 'adj_close' or 'close' must be provided.")
        self.date = date
        self.code = code
        self.adj_close = adj_close
        self.close = close
    
class ShowCommodities(Base):
    __tablename__ = 'show_commodities'

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    commodity_zar_open = Column(Float)
    commodity_zar_high = Column(Float)
    commodity_zar_low = Column(Float)
    commodity_zar_close = Column(Float)
    commodity_zar_adj_close = Column(Float)
    volume = Column(BigInteger)

class Stock(Base):
    __tablename__ = 'stocks'
    id = Column(Integer, primary_key=True)
    code = Column(String, nullable=False)
    share_name = Column(String, nullable=False)
    industry_id = Column(Integer, ForeignKey('industries.id'), nullable=False)
    sub_industry_id = Column(Integer, ForeignKey('sub_industries.id'), nullable=False)
    rsi_comparison_market = Column(String)
    rsi_comparison_sector = Column(String)
    commodity = Column(Boolean, default=False)

    industry = relationship("Industry", back_populates="stocks")
    sub_industry = relationship("SubIndustry", back_populates="stocks")

class StockDataHistory(Base):
    __tablename__ = 'stock_data_history'

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)
    adj_close = Column(Float)
    comparison_market = Column(String)
    comparison_sector = Column(String)

class SubIndustry(Base):
    __tablename__ = 'sub_industries'
    id = Column(Integer, primary_key=True)
    industry_id = Column(Integer, ForeignKey('industries.id'), nullable=False)
    name = Column(String, unique=True, nullable=False)

    industry = relationship("Industry", back_populates="sub_industries")
    stocks = relationship("Stock", back_populates="sub_industry", cascade="all, delete")

class Subscribers(Base):
    __tablename__ = 'subscribers'
    __table_args__ = (UniqueConstraint('email', name='_email_uc'),)
    id = Column(Integer, primary_key=True)
    email = Column(String, nullable=False)  # Use ticker as the primary key
    name = Column(String, nullable=False)  # Use ticker as the primary key
    subscription = Column(Integer, nullable=True)  # Use ticker as the primary key
    email_date = Column(Date, nullable=True)  # Use ticker as the primary key
    subscription_date = Column(Date, nullable=False)  # Use ticker as the primary key
    password = Column(String, nullable=False)

class Subscriptions(Base):
    __tablename__ = 'subscriptions'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    cost = Column(Float, nullable=False)
    detail = Column(String, nullable=False)

class TickerName(Base):
    __tablename__ = 'ticker_name'
    ticker = Column(String, primary_key=True)  # Use ticker as the primary key
    name = Column(String, nullable=False)

class ZARBad(Base):
    __tablename__ = 'zar_bad'

    id = Column(Integer, primary_key=True)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)

class ZARGood(Base):
    __tablename__ = 'zar_good'

    id = Column(Integer, primary_key=True)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)

class ZARUSD(Base):
    __tablename__ = 'zar_usd'

    date = Column(Date, primary_key=True)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(BigInteger)
    open = Column(Float)
    overbought_oversold = Column(Numeric)
