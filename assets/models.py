"""SQLAlchemy ORM models for the market-data (``sharesdata``) database.

Declared with the SQLAlchemy 2.0 typed ORM (``Mapped`` / ``mapped_column``)
so column and relationship types are statically checked. Nullability mirrors
the database schema exactly: a non-optional ``Mapped[T]`` is ``NOT NULL`` (or a
primary key), while ``Mapped[Optional[T]]`` is a nullable column.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    CHAR,
    BigInteger,
    Boolean,
    Date,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ``sharesdata`` ORM models."""


class NewsSentiment(Base):
    """Daily aggregated news-sentiment score per ticker.

    sentiment_score: VADER compound in [-1, +1]; 0.0 = neutral/no news.
    article_count  : number of news articles scored that day.
    positive_count : articles with compound > 0.05.
    negative_count : articles with compound < -0.05.
    neutral_count  : articles within [-0.05, +0.05].
    """

    __tablename__ = "news_sentiment"
    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_sentiment_ticker_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(
        String(10), ForeignKey("stocks.code"), nullable=False
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    article_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    positive_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    negative_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    neutral_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class MacroSentiment(Base):
    """Daily aggregated macro/thematic news-sentiment score per theme.

    theme          : one of ``assets.macro_sentiment.ALL_THEMES`` (e.g. MARKET,
                     OIL, GOLD, RAND, FINANCIALS, GLOBAL_RISK).
    sentiment_score: recency-weighted VADER compound in [-1, +1]; 0.0 = neutral.
    article_count  : number of articles classified into the theme that day.
    """

    __tablename__ = "macro_sentiment"
    __table_args__ = (UniqueConstraint("date", "theme", name="uq_macro_date_theme"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    theme: Mapped[str] = mapped_column(String(32), nullable=False)
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    article_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    positive_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    negative_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    neutral_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class AdjRuns(Base):
    __tablename__ = "adj_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(
        String(10), ForeignKey("stocks.code"), nullable=False
    )
    share_name: Mapped[Optional[str]] = mapped_column(String)
    industry: Mapped[Optional[str]] = mapped_column(String)
    sub_industry: Mapped[Optional[str]] = mapped_column(String)
    rsi_comparison_market: Mapped[Optional[str]] = mapped_column(String)
    rsi_comparison_sector: Mapped[Optional[str]] = mapped_column(String)
    commodity: Mapped[Optional[bool]] = mapped_column(Boolean)
    current_price: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    current_value: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    next_week_prediction: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    next_month_prediction: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    closing: Mapped[Optional[Decimal]] = mapped_column(
        Numeric, nullable=True, default=None
    )
    z_score: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    overbought_oversold: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    overbought_oversold_value: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    ma24: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    ma55: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    sector_rsi_1m: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    sector_rsi_3m: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    sector_rsi_6m: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    market_rsi_1m: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    market_rsi_3m: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    market_rsi_6m: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    daily_tr_return: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    daily_close_return: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    true_range: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    true_range_pct: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    atr_14_pct: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    vol_24d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    vol_55d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    vol_ratio_24_55: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    return_24d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    return_55d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    risk_adj_mom_24d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    drawdown_55d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    avg_volume_24d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    volume_ratio_24d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    ma_24d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    ma_55d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    price_vs_ma_24d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    price_vs_ma_55d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    ma_trend_24_55: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    volume: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    run_date: Mapped[Optional[date]] = mapped_column(Date)
    next_month_volume_prediction: Mapped[Optional[Decimal]] = mapped_column(
        Numeric, nullable=True, default=None
    )
    next_week_volume_prediction: Mapped[Optional[Decimal]] = mapped_column(
        Numeric, nullable=True, default=None
    )
    market_cap: Mapped[Optional[Decimal]] = mapped_column(
        Numeric, nullable=True, default=None
    )
    weight: Mapped[Optional[Decimal]] = mapped_column(
        Numeric, nullable=True, default=None
    )
    shares: Mapped[Optional[Decimal]] = mapped_column(
        Numeric, nullable=True, default=None
    )
    stocks: Mapped["Stock"] = relationship(back_populates="adj_runs")


class CloseRuns(Base):
    __tablename__ = "close_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(
        String(10), ForeignKey("stocks.code"), nullable=False
    )
    share_name: Mapped[str] = mapped_column(String, nullable=False)
    industry: Mapped[str] = mapped_column(String, nullable=False)
    sub_industry: Mapped[Optional[str]] = mapped_column(String)
    rsi_comparison_market: Mapped[Optional[str]] = mapped_column(String)
    rsi_comparison_sector: Mapped[Optional[str]] = mapped_column(String)
    commodity: Mapped[Optional[bool]] = mapped_column(Boolean)
    current_price: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    current_value: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    next_week_prediction: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    next_month_prediction: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    z_score: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    overbought_oversold: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    overbought_oversold_value: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    ma24: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    ma55: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    sector_rsi_1m: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    sector_rsi_3m: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    sector_rsi_6m: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    market_rsi_1m: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    market_rsi_3m: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    market_rsi_6m: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    daily_tr_return: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    daily_close_return: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    true_range: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    true_range_pct: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    atr_14_pct: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    vol_24d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    vol_55d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    vol_ratio_24_55: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    return_24d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    return_55d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    risk_adj_mom_24d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    drawdown_55d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    avg_volume_24d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    volume_ratio_24d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    ma_24d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    ma_55d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    price_vs_ma_24d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    price_vs_ma_55d: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    ma_trend_24_55: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    volume: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    run_date: Mapped[date] = mapped_column(Date, nullable=False)
    stocks: Mapped["Stock"] = relationship(back_populates="close_runs")


class Commodity(Base):
    __tablename__ = "commodities"
    date: Mapped[date] = mapped_column(Date, primary_key=True, nullable=False)
    ticker: Mapped[str] = mapped_column(String(10), primary_key=True, nullable=False)
    Open: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    High: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    Low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    Close: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AdjClose: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    Volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class Dividend(Base):
    __tablename__ = "dividends"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    ticker: Mapped[str] = mapped_column(
        String, ForeignKey("stocks.code"), nullable=False
    )
    dividend: Mapped[float] = mapped_column(Float, nullable=False)


class Industry(Base):
    __tablename__ = "industries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    sub_industries: Mapped[list["SubIndustry"]] = relationship(
        back_populates="industry", cascade="all, delete-orphan"
    )
    stocks: Mapped[list["Stock"]] = relationship(
        back_populates="industry", cascade="all, delete-orphan"
    )


class RSIComparisonSector(Base):
    __tablename__ = "rsi_comparison_sectors"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)


class RSIComparisonMarket(Base):
    __tablename__ = "rsi_comparison_markets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (UniqueConstraint("date", "code", name="_date_code_uc"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    code: Mapped[str] = mapped_column(String, nullable=False)
    adj_close: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    close: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class ShowCommodities(Base):
    __tablename__ = "show_commodities"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    commodity_zar_open: Mapped[Optional[float]] = mapped_column(Float)
    commodity_zar_high: Mapped[Optional[float]] = mapped_column(Float)
    commodity_zar_low: Mapped[Optional[float]] = mapped_column(Float)
    commodity_zar_close: Mapped[Optional[float]] = mapped_column(Float)
    commodity_zar_adj_close: Mapped[Optional[float]] = mapped_column(Float)
    volume: Mapped[Optional[int]] = mapped_column(BigInteger)


class Stock(Base):
    __tablename__ = "stocks"
    __table_args__ = (UniqueConstraint("code", name="uq_ticker"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    share_name: Mapped[str] = mapped_column(String, nullable=False)
    industry_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("industries.id"), nullable=False
    )
    sub_industry_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sub_industries.id"), nullable=False
    )
    rsi_comparison_market: Mapped[Optional[str]] = mapped_column(String)
    rsi_comparison_sector: Mapped[Optional[str]] = mapped_column(String)
    commodity: Mapped[Optional[bool]] = mapped_column(Boolean, default=False)
    # Relationships
    industry: Mapped["Industry"] = relationship(back_populates="stocks")
    sub_industry: Mapped["SubIndustry"] = relationship(back_populates="stocks")
    stock_data_history: Mapped[list["StockDataHistory"]] = relationship(
        back_populates="stocks"
    )
    ticker_name: Mapped[Optional["TickerName"]] = relationship(
        back_populates="stocks", uselist=False
    )
    technical_analysis: Mapped[list["TechnicalAnalysis"]] = relationship(
        back_populates="stocks"
    )
    adj_runs: Mapped[list["AdjRuns"]] = relationship(back_populates="stocks")
    close_runs: Mapped[list["CloseRuns"]] = relationship(back_populates="stocks")
    portfolios: Mapped[list["Portfolio"]] = relationship(
        secondary="portfolio_stocks", back_populates="stocks"
    )


class StockDataHistory(Base):
    __tablename__ = "stock_data_history"
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_ticker_date"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(
        String(10), ForeignKey("stocks.code"), nullable=False
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[Optional[float]] = mapped_column(Float)
    high: Mapped[Optional[float]] = mapped_column(Float)
    low: Mapped[Optional[float]] = mapped_column(Float)
    close: Mapped[Optional[float]] = mapped_column(Float)
    volume: Mapped[Optional[int]] = mapped_column(BigInteger)
    adj_close: Mapped[Optional[float]] = mapped_column(Float)
    comparison_market: Mapped[Optional[str]] = mapped_column(String)
    comparison_sector: Mapped[Optional[str]] = mapped_column(String)
    stocks: Mapped["Stock"] = relationship(back_populates="stock_data_history")


class SubIndustry(Base):
    __tablename__ = "sub_industries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    industry_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("industries.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    industry: Mapped["Industry"] = relationship(back_populates="sub_industries")
    stocks: Mapped[list["Stock"]] = relationship(
        back_populates="sub_industry", cascade="all, delete"
    )


class Subscribers(Base):
    __tablename__ = "subscribers"
    __table_args__ = (UniqueConstraint("email", name="_email_uc"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    subscription: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("subscriptions.id"), nullable=True
    )
    email_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    subscription_date: Mapped[date] = mapped_column(Date, nullable=False)
    password: Mapped[str] = mapped_column(String, nullable=False)
    token: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    email_hash: Mapped[Optional[str]] = mapped_column(
        String(64), unique=True
    )  # Unique hash for tracking
    web_hash: Mapped[Optional[str]] = mapped_column(
        String(64), unique=True
    )  # Unique hash for tracking
    email_opened_count: Mapped[Optional[int]] = mapped_column(
        Integer, default=0
    )  # Track email open events
    web_opened_count: Mapped[Optional[int]] = mapped_column(
        Integer, default=0
    )  # Track email open events
    id_number: Mapped[str] = mapped_column(String, nullable=False)
    black_listed: Mapped[Optional[bool]] = mapped_column(Boolean, default=False)

    subscriptions: Mapped[Optional["Subscriptions"]] = relationship(
        back_populates="Subscribers", uselist=False
    )


class Subscriptions(Base):
    __tablename__ = "subscriptions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    cost: Mapped[float] = mapped_column(Float, nullable=False)
    detail: Mapped[str] = mapped_column(String, nullable=False)
    Subscribers: Mapped[list["Subscribers"]] = relationship(
        back_populates="subscriptions"
    )


class TickerName(Base):
    __tablename__ = "ticker_name"
    ticker: Mapped[str] = mapped_column(
        Text, ForeignKey("stocks.code"), primary_key=True
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)

    stocks: Mapped["Stock"] = relationship(back_populates="ticker_name", uselist=False)


class ZARBad(Base):
    __tablename__ = "zar_bad"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)


class ZARGood(Base):
    __tablename__ = "zar_good"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)


class ZARUSD(Base):
    __tablename__ = "zar_usd"
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    high: Mapped[Optional[float]] = mapped_column(Float)
    low: Mapped[Optional[float]] = mapped_column(Float)
    close: Mapped[Optional[float]] = mapped_column(Float)
    adj_close: Mapped[Optional[float]] = mapped_column(Float)
    volume: Mapped[Optional[int]] = mapped_column(BigInteger)
    open: Mapped[Optional[float]] = mapped_column(Float)
    overbought_oversold: Mapped[Optional[Decimal]] = mapped_column(Numeric)


class VIData(Base):
    __tablename__ = "vi_data"
    __table_args__ = (PrimaryKeyConstraint("code", "run_date"),)
    code: Mapped[str] = mapped_column(String(20), primary_key=True)
    run_date: Mapped[date] = mapped_column(Date, primary_key=True)
    eps: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    nav: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    sales: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    eps_growth_f: Mapped[Optional[str]] = mapped_column(String(10))
    roe_f: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    inst_profit_margin_f: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    sales_growth_f: Mapped[Optional[str]] = mapped_column(String(10))
    holding: Mapped[Optional[str]] = mapped_column(CHAR(1))
    shares: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    interest_cover: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    comment: Mapped[Optional[str]] = mapped_column(String(50))
    tnav: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    rote: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    actual_roe: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    last_update: Mapped[Optional[str]] = mapped_column(String(20))
    o_margin: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    div: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    cash_ps: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    act: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    heps: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    quality_rating: Mapped[Optional[str]] = mapped_column(String(10))
    div_decl: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    div_ldt: Mapped[Optional[date]] = mapped_column(Date)
    div_pay: Mapped[Optional[date]] = mapped_column(Date)
    rec: Mapped[Optional[str]] = mapped_column(String(20))
    rec_on: Mapped[Optional[str]] = mapped_column(String(20))
    ye_release: Mapped[Optional[date]] = mapped_column(Date)
    int_release: Mapped[Optional[date]] = mapped_column(Date)
    rec_price: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    share_price: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    peg: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    peg_pe: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    peg_pe_value: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    peg_nav: Mapped[Optional[str]] = mapped_column(String(10))
    peg_pe_nav_value: Mapped[Optional[Decimal]] = mapped_column(Numeric)


# Model for Actions
class Action(Base):
    __tablename__ = "actions"

    ticker: Mapped[str] = mapped_column(String(10), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    dividends: Mapped[Optional[float]] = mapped_column(Float)
    stock_splits: Mapped[Optional[float]] = mapped_column(Float)


# Model for Balance Sheet
class BalanceSheet(Base):
    __tablename__ = "balance_sheet"

    ticker: Mapped[str] = mapped_column(String(10), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    column_name: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[Optional[Decimal]] = mapped_column(Numeric)


# Model for Cash Flow
class CashFlow(Base):
    __tablename__ = "cash_flow"

    ticker: Mapped[str] = mapped_column(String(10), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    column_name: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[Optional[Decimal]] = mapped_column(Numeric)


# Model for Earnings Dates
class EarningsDate(Base):
    __tablename__ = "earnings_dates"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String, nullable=False)
    earnings_date: Mapped[date] = mapped_column(Date, nullable=False)
    eps_estimate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reported_eps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    surprise_percentage: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


# Model for Financials
class Financial(Base):
    __tablename__ = "financials"

    ticker: Mapped[str] = mapped_column(String(10), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    column_name: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[Optional[Decimal]] = mapped_column(Numeric)


# Model for Major Holders
class MajorHolder(Base):
    __tablename__ = "major_holders"

    ticker: Mapped[str] = mapped_column(String(10), primary_key=True)
    holder_name: Mapped[str] = mapped_column(String, primary_key=True)
    shares_held: Mapped[Optional[int]] = mapped_column(BigInteger)
    percentage_held: Mapped[Optional[float]] = mapped_column(Float)


# Model for Mutual Fund Holders
class MutualFundHolder(Base):
    __tablename__ = "mutualfund_holders"

    ticker: Mapped[str] = mapped_column(String(10), primary_key=True)
    holder_name: Mapped[str] = mapped_column(String, primary_key=True)
    shares_held: Mapped[Optional[int]] = mapped_column(BigInteger)
    percentage_held: Mapped[Optional[float]] = mapped_column(Float)


# Model for Recommendations
class Recommendation(Base):
    __tablename__ = "recommendations"

    ticker: Mapped[str] = mapped_column(String(10), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    firm: Mapped[Optional[str]] = mapped_column(String)
    to_grade: Mapped[Optional[str]] = mapped_column(String)
    from_grade: Mapped[Optional[str]] = mapped_column(String)
    action: Mapped[Optional[str]] = mapped_column(String)


class TechnicalAnalysis(Base):
    __tablename__ = "technical_analysis"
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_ticker_date"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(
        String(10), ForeignKey("stocks.code"), nullable=False
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    signal: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    action: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    close: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sma_22: Mapped[float] = mapped_column(Float, nullable=False)
    sma_55: Mapped[float] = mapped_column(Float, nullable=False)
    rsi_1m: Mapped[float] = mapped_column(Float, nullable=False)
    rsi_3m: Mapped[float] = mapped_column(Float, nullable=False)
    rsi_6m: Mapped[float] = mapped_column(Float, nullable=False)
    bollinger_high: Mapped[float] = mapped_column(Float, nullable=False)
    bollinger_low: Mapped[float] = mapped_column(Float, nullable=False)
    bollinger_mid: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bollinger_perc_b: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bollinger_width: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    overbaughtoversold: Mapped[float] = mapped_column(Float, nullable=False)
    zscore: Mapped[float] = mapped_column(Float, nullable=False)

    stocks: Mapped["Stock"] = relationship(back_populates="technical_analysis")


# Association table for many-to-many relationship between portfolios and stocks
class PortfolioStock(Base):
    __tablename__ = "portfolio_stocks"
    portfolio_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("portfolios.id"), primary_key=True
    )
    stock_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("stocks.id"), primary_key=True
    )
    shares: Mapped[float] = mapped_column(Float, default=1, nullable=False)


class Portfolio(Base):
    __tablename__ = "portfolios"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    stocks: Mapped[list["Stock"]] = relationship(
        secondary="portfolio_stocks", back_populates="portfolios"
    )
