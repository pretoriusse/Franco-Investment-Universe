"""Flask-SQLAlchemy ORM models for the web-application (``webapp``) database.

Uses a typed ``DeclarativeBase`` passed to ``SQLAlchemy(model_class=...)`` so
the models are statically checkable under ``mypy --strict`` (subclassing the
dynamic ``db.Model`` attribute is not).
"""

from __future__ import annotations

import datetime
from typing import Optional

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import (
    Boolean,
    Date,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ``webapp`` ORM models."""


db = SQLAlchemy(model_class=Base)


class Subscribers(Base):
    __tablename__ = "subscribers"
    __table_args__ = (UniqueConstraint("email", name="_email_uc"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    subscription_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("subscriptions.id"), nullable=True
    )
    email_date: Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)
    subscription_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    subscription_expiration_date: Mapped[datetime.date] = mapped_column(
        Date, nullable=False
    )
    subscription_paid: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
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
    api_key_hash: Mapped[Optional[str]] = mapped_column(
        String(64), unique=True
    )  # sha256 of the API key

    # Relationships
    subscription: Mapped[Optional["Subscriptions"]] = relationship(
        "Subscriptions", back_populates="subscribers"
    )
    referals: Mapped[list["Referals"]] = relationship(
        "Referals", back_populates="subscriber", cascade="all, delete-orphan"
    )
    id_numbers: Mapped[list["SubscriberIDNumbers"]] = relationship(
        "SubscriberIDNumbers", back_populates="subscriber"
    )
    html_web_views: Mapped[list["HTMLWebView"]] = relationship(
        "HTMLWebView", back_populates="subscriber", cascade="all, delete-orphan"
    )

    @property
    def is_active(self) -> bool:
        # This could check if the user is active based on expiration date or other criteria
        return (
            self.subscription_expiration_date > datetime.datetime.now().date()
            and not self.black_listed
        )

    @property
    def is_authenticated(self) -> bool:
        # Since this is managed by Flask-Login, you can return True for authenticated users
        return True

    @property
    def is_anonymous(self) -> bool:
        # Return False as our users are not anonymous
        return False

    def get_id(self) -> str:
        # Return the user's unique identifier
        return str(self.id)


class Subscriptions(Base):
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    cost: Mapped[float] = mapped_column(Float, nullable=False)
    detail: Mapped[str] = mapped_column(String, nullable=False)

    # Relationships
    subscribers: Mapped[list["Subscribers"]] = relationship(
        "Subscribers", back_populates="subscription", cascade="all, delete-orphan"
    )
    functions: Mapped[list["SubscriptionFunctions"]] = relationship(
        "SubscriptionFunctions",
        back_populates="subscription",
        cascade="all, delete-orphan",
    )


class SubscriberIDNumbers(Base):
    __tablename__ = "subscription_id_numbers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subscriber_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("subscribers.id"), nullable=False
    )
    id_number: Mapped[str] = mapped_column(String, nullable=False)

    # Relationships
    subscriber: Mapped["Subscribers"] = relationship(
        "Subscribers", back_populates="id_numbers"
    )


class SubscriptionFunctions(Base):
    __tablename__ = "subscription_functions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subscription_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("subscriptions.id"), nullable=False
    )
    company_research: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    portfolio: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    api_access: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Relationships
    subscription: Mapped["Subscriptions"] = relationship(
        "Subscriptions", back_populates="functions"
    )


class Referals(Base):
    __tablename__ = "referals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subscriber_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("subscribers.id"), nullable=False
    )
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    refferal_code: Mapped[str] = mapped_column(String, nullable=False)
    count: Mapped[int] = mapped_column(Integer, nullable=False)
    expiry: Mapped[datetime.date] = mapped_column(Date, nullable=False)

    # Relationships
    subscriber: Mapped["Subscribers"] = relationship(
        "Subscribers", back_populates="referals"
    )


class HTMLWebView(Base):
    __tablename__ = "htmlwebview"
    __table_args__ = (
        UniqueConstraint("html_summary_path", name="_html_summary_path_uc"),
        UniqueConstraint("html_detailed_path", name="_html_detailed_path_uc"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    display_date: Mapped[str] = mapped_column(String, nullable=False)
    report_type: Mapped[str] = mapped_column(String, nullable=False)
    html_summary_path: Mapped[str] = mapped_column(String, nullable=False)
    html_detailed_path: Mapped[str] = mapped_column(String, nullable=False)
    pdf_summary_path: Mapped[str] = mapped_column(String, nullable=False)
    pdf_detailed_path: Mapped[str] = mapped_column(String, nullable=False)
    actual_run_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    subscriber_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("subscribers.id"), nullable=False
    )

    # Relationship to link back to subscriber
    subscriber: Mapped["Subscribers"] = relationship(
        "Subscribers", back_populates="html_web_views"
    )


class PortfolioTransactionHistory(Base):
    __tablename__ = "portfolio_transaction_history"
    __table_args__ = (
        UniqueConstraint("subscriber_id", "date", "share", "action", name="_txn_uc"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subscriber_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("subscribers.id"), nullable=True
    )
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    share: Mapped[str] = mapped_column(String, nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[str] = mapped_column(String, nullable=False)


class PortfolioTracker(Base):
    __tablename__ = "portfolio_tracker"
    __table_args__ = (
        UniqueConstraint("ticker", "subscriber_id", name="_ticker_subscriber_uc"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subscriber_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("subscribers.id"), nullable=True
    )
    ticker: Mapped[str] = mapped_column(String, nullable=False)
    weight: Mapped[str] = mapped_column(String, nullable=False)
    comment: Mapped[str] = mapped_column(String, nullable=False)
