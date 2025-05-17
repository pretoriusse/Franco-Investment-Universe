from sqlalchemy import Column, Integer, Text, String, Float, Date, BigInteger, Numeric, Boolean, ForeignKey, UniqueConstraint, PrimaryKeyConstraint, CHAR, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class AdjRuns(Base):
    __tablename__ = 'adj_runs'
    id = Column(Integer, primary_key=True)
    code = Column(String(10), ForeignKey('stocks.code'), nullable=False)
    share_name = Column(String)
    industry = Column(String)
    sub_industry = Column(String)
    rsi_comparison_market = Column(String)
    rsi_comparison_sector = Column(String)
    commodity = Column(Boolean)
    current_price = Column(Numeric)
    current_value = Column(Numeric)
    next_week_prediction = Column(Numeric)
    next_month_prediction = Column(Numeric)
    closing = Column(Numeric, nullable=True, default=None)
    z_score = Column(Numeric)
    overbought_oversold = Column(Numeric)
    overbought_oversold_value = Column(Numeric)
    ma24 = Column(Numeric)
    ma55 = Column(Numeric)
    sector_rsi_1m = Column(Numeric)
    sector_rsi_3m = Column(Numeric)
    sector_rsi_6m = Column(Numeric)
    market_rsi_1m = Column(Numeric)
    market_rsi_3m = Column(Numeric)
    market_rsi_6m = Column(Numeric)
    run_date = Column(Date)
    next_month_volume_prediction = Column(Numeric, nullable=True, default=None)
    next_week_volume_prediction = Column(Numeric, nullable=True, default=None)
    market_cap = Column(Numeric, nullable=True, default=None)
    weight = Column(Numeric, nullable=True, default=None)
    shares = Column(Numeric, nullable=True, default=None)
    stocks = relationship("Stock", back_populates="adj_runs")

class CloseRuns(Base):
    __tablename__ = 'close_runs'
    id = Column(Integer, primary_key=True)
    code = Column(String(10), ForeignKey('stocks.code'), nullable=False)
    share_name = Column(String, nullable=False)
    industry = Column(String, nullable=False)
    sub_industry = Column(String)
    rsi_comparison_market = Column(String)
    rsi_comparison_sector = Column(String)
    commodity = Column(Boolean)
    current_price = Column(Numeric)
    current_value = Column(Numeric)
    next_week_prediction = Column(Numeric)
    next_month_prediction = Column(Numeric)
    z_score = Column(Numeric)
    overbought_oversold = Column(Numeric)
    overbought_oversold_value = Column(Numeric)
    ma24 = Column(Numeric)
    ma55 = Column(Numeric)
    sector_rsi_1m = Column(Numeric)
    sector_rsi_3m = Column(Numeric)
    sector_rsi_6m = Column(Numeric)
    market_rsi_1m = Column(Numeric)
    market_rsi_3m = Column(Numeric)
    market_rsi_6m = Column(Numeric)
    run_date = Column(Date, nullable=False)
    stocks = relationship("Stock", back_populates="close_runs")

class Commodity(Base):
    __tablename__ = 'commodities'
    date = Column(Date, primary_key=True, nullable=False)
    ticker = Column(String(10), primary_key=True, nullable=False)
    Open = Column(Float, nullable=True)
    High = Column(Float, nullable=True)
    Low = Column(Float, nullable=True)
    Close = Column(Float, nullable=True)
    AdjClose = Column(Float, nullable=True)
    Volume = Column(Float, nullable=True)

class Dividend(Base):
    __tablename__ = 'dividends'
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    ticker = Column(String, ForeignKey('stocks.code'), nullable=False)
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
    code = Column(String, nullable=False, unique=True)
    share_name = Column(String, nullable=False)
    industry_id = Column(Integer, ForeignKey('industries.id'), nullable=False)
    sub_industry_id = Column(Integer, ForeignKey('sub_industries.id'), nullable=False)
    rsi_comparison_market = Column(String)
    rsi_comparison_sector = Column(String)
    commodity = Column(Boolean, default=False)
    # Relationships
    industry = relationship("Industry", back_populates="stocks")
    sub_industry = relationship("SubIndustry", back_populates="stocks")
    stock_data_history = relationship("StockDataHistory", back_populates="stocks")
    ticker_name = relationship("TickerName", back_populates="stocks", uselist=False)
    technical_analysis = relationship("TechnicalAnalysis", back_populates="stocks")
    adj_runs = relationship("AdjRuns", back_populates="stocks")
    close_runs = relationship("CloseRuns", back_populates="stocks")
    portfolios = relationship("Portfolio", secondary="portfolio_stocks", back_populates="stocks")

    # Unique Keys
    __table_args__ = (
        UniqueConstraint('code', name='uq_ticker'),
    )

class StockDataHistory(Base):
    __tablename__ = 'stock_data_history'
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), ForeignKey('stocks.code'),  nullable=False) #
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)
    adj_close = Column(Float)
    comparison_market = Column(String)
    comparison_sector = Column(String)
    __table_args__ = (
        UniqueConstraint('ticker', 'date', name='uq_ticker_date'),
    )
    stocks = relationship("Stock", back_populates="stock_data_history")

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
    email = Column(String, nullable=False)
    name = Column(String, nullable=False)
    subscription = Column(Integer, ForeignKey("subscriptions.id"), nullable=True)
    email_date = Column(Date, nullable=True)
    subscription_date = Column(Date, nullable=False)
    password = Column(String, nullable=False)
    token = Column(String, nullable=True)
    is_admin = Column(Boolean, nullable=False, default=False)
    email_hash = Column(String(64), unique=True)  # Unique hash for tracking
    web_hash = Column(String(64), unique=True)  # Unique hash for tracking
    email_opened_count = Column(Integer, default=0)  # Track email open events
    web_opened_count = Column(Integer, default=0)  # Track email open events
    id_number = Column(String, nullable=False)
    black_listed = Column(Boolean, default=False)

    subscriptions = relationship("Subscriptions", back_populates="Subscribers", uselist=False)

class Subscriptions(Base):
    __tablename__ = 'subscriptions'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    cost = Column(Float, nullable=False)
    detail = Column(String, nullable=False)
    Subscribers = relationship("Subscribers", back_populates="subscriptions")

class TickerName(Base):
    __tablename__ = 'ticker_name'
    ticker = Column(Text, ForeignKey('stocks.code'), primary_key=True)
    name = Column(Text, nullable=False)

    stocks = relationship("Stock", back_populates="ticker_name", uselist=False)

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

class VIData(Base):
    __tablename__ = 'vi_data'
    code = Column(String(20), primary_key=True)
    run_date = Column(Date, primary_key=True)
    eps = Column(Numeric)
    nav = Column(Numeric)
    sales = Column(Numeric)
    eps_growth_f = Column(String(10))
    roe_f = Column(Numeric)
    inst_profit_margin_f = Column(Numeric)
    sales_growth_f = Column(String(10))
    holding = Column(CHAR(1))
    shares = Column(Numeric)
    interest_cover = Column(Numeric)
    comment = Column(String(50))
    tnav = Column(Numeric)
    rote = Column(Numeric)
    actual_roe = Column(Numeric)
    last_update = Column(String(20))
    o_margin = Column(Numeric)
    div = Column(Numeric)
    cash_ps = Column(Numeric)
    act = Column(Numeric)
    heps = Column(Numeric)
    quality_rating = Column(String(10))
    div_decl = Column(Numeric)
    div_ldt = Column(Date)
    div_pay = Column(Date)
    rec = Column(String(20))
    rec_on = Column(String(20))
    ye_release = Column(Date)
    int_release = Column(Date)
    rec_price = Column(Numeric)
    share_price = Column(Numeric)
    peg = Column(Numeric)
    peg_pe = Column(Numeric)
    peg_pe_value = Column(Numeric)
    peg_nav = Column(String(10))
    peg_pe_nav_value = Column(Numeric)
    __table_args__ = (
        PrimaryKeyConstraint('code', 'run_date'),
    )


# Model for Actions
class Action(Base):
    __tablename__ = 'actions'
    
    ticker = Column(String(10), primary_key=True)
    date = Column(Date, primary_key=True)
    dividends = Column(Float)
    stock_splits = Column(Float)

# Model for Balance Sheet
class BalanceSheet(Base):
    __tablename__ = 'balance_sheet'
    
    ticker = Column(String(10), primary_key=True)
    date = Column(Date, primary_key=True)
    column_name = Column(String, primary_key=True)
    value = Column(Numeric)

# Model for Cash Flow
class CashFlow(Base):
    __tablename__ = 'cash_flow'
    
    ticker = Column(String(10), primary_key=True)
    date = Column(Date, primary_key=True)
    column_name = Column(String, primary_key=True)
    value = Column(Numeric)

# Model for Earnings Dates
class EarningsDate(Base):
    __tablename__ = 'earnings_dates'
    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False)
    earnings_date = Column(Date, nullable=False)
    eps_estimate = Column(Float, nullable=True)
    reported_eps = Column(Float, nullable=True)
    surprise_percentage = Column(Float, nullable=True)

# Model for Financials
class Financial(Base):
    __tablename__ = 'financials'
    
    ticker = Column(String(10), primary_key=True)
    date = Column(Date, primary_key=True)
    column_name = Column(String, primary_key=True)
    value = Column(Numeric)

# Model for Major Holders
class MajorHolder(Base):
    __tablename__ = 'major_holders'
    
    ticker = Column(String(10), primary_key=True)
    holder_name = Column(String, primary_key=True)
    shares_held = Column(BigInteger)
    percentage_held = Column(Float)

# Model for Mutual Fund Holders
class MutualFundHolder(Base):
    __tablename__ = 'mutualfund_holders'
    
    ticker = Column(String(10), primary_key=True)
    holder_name = Column(String, primary_key=True)
    shares_held = Column(BigInteger)
    percentage_held = Column(Float)

# Model for Recommendations
class Recommendation(Base):
    __tablename__ = 'recommendations'
    
    ticker = Column(String(10), primary_key=True)
    date = Column(Date, primary_key=True)
    firm = Column(String)
    to_grade = Column(String)
    from_grade = Column(String)
    action = Column(String)

class TechnicalAnalysis(Base):
    __tablename__ = 'technical_analysis'
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), ForeignKey("stocks.code"), nullable=False)
    date = Column(Date, nullable=False)
    signal = Column(String, nullable=True)
    action = Column(String, nullable=True)
    close = Column(Float, nullable=True)
    sma_22 = Column(Float, nullable=False)
    sma_55 = Column(Float, nullable=False)
    rsi_1m = Column(Float, nullable=False)
    rsi_3m = Column(Float, nullable=False)
    rsi_6m = Column(Float, nullable=False)
    bollinger_high = Column(Float, nullable=False)
    bollinger_low = Column(Float, nullable=False)
    bollinger_mid = Column(Float, nullable=True)
    bollinger_perc_b = Column(Float, nullable=True)
    bollinger_width = Column(Float, nullable=True)
    overbaughtoversold = Column(Float, nullable=False)
    zscore = Column(Float, nullable=False)
    __table_args__ = (
        UniqueConstraint('ticker', 'date', name='uq_ticker_date'),
    )
    
    stocks = relationship("Stock", back_populates="technical_analysis")

# Association table for many-to-many relationship between portfolios and stocks
class PortfolioStock(Base):
    __tablename__ = 'portfolio_stocks'
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'), primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), primary_key=True)
    shares = Column(Float, default=1, nullable=False)

class Portfolio(Base):
    __tablename__ = 'portfolios'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    stocks = relationship("Stock", secondary="portfolio_stocks", back_populates="portfolios")
