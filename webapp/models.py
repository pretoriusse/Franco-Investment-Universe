from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, Float, Date, Boolean, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
import datetime

db = SQLAlchemy()

class Subscribers(db.Model):
    __tablename__ = 'subscribers'
    __table_args__ = (UniqueConstraint('email', name='_email_uc'),)
    
    id = Column(Integer, primary_key=True)
    email = Column(String, nullable=False)
    name = Column(String, nullable=False)
    subscription_id = Column(Integer, ForeignKey('subscriptions.id'), nullable=True)
    email_date = Column(Date, nullable=True)
    subscription_date = Column(Date, nullable=False)
    subscription_expiration_date = Column(Date, nullable=False)
    subscription_paid = Column(Boolean, nullable=False, default=False)
    password = Column(String, nullable=False)
    token = Column(String, nullable=True)
    is_admin = Column(Boolean, nullable=False, default=False)
    email_hash = db.Column(db.String(64), unique=True)  # Unique hash for tracking
    web_hash = db.Column(db.String(64), unique=True)  # Unique hash for tracking
    email_opened_count = db.Column(db.Integer, default=0)  # Track email open events
    web_opened_count = db.Column(db.Integer, default=0)  # Track email open events
    id_number = Column(String, nullable=False)
    black_listed = Column(Boolean, default=False)
    
    # Relationships
    subscription = relationship('Subscriptions', back_populates='subscribers')
    referals = relationship('Referals', back_populates='subscriber', cascade="all, delete-orphan")
    id_numbers = relationship('SubscriberIDNumbers', back_populates='subscriber')
    html_web_views = relationship('HTMLWebView', back_populates='subscriber', cascade="all, delete-orphan")

    @property
    def is_active(self):
        # This could check if the user is active based on expiration date or other criteria
        return self.subscription_expiration_date > datetime.datetime.now().date() and not self.black_listed

    @property
    def is_authenticated(self):
        # Since this is managed by Flask-Login, you can return True for authenticated users
        return True

    @property
    def is_anonymous(self):
        # Return False as our users are not anonymous
        return False

    def get_id(self):
        # Return the user's unique identifier
        return str(self.id)

class Subscriptions(db.Model):
    __tablename__ = 'subscriptions'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    cost = Column(Float, nullable=False)
    detail = Column(String, nullable=False)
    
    # Relationships
    subscribers = relationship('Subscribers', back_populates='subscription', cascade="all, delete-orphan")
    functions = relationship('SubscriptionFunctions', back_populates='subscription', cascade="all, delete-orphan")

class SubscriberIDNumbers(db.Model):
    __tablename__ = 'subscription_id_numbers'
    
    id = Column(Integer, primary_key=True)
    subscriber_id = Column(Integer, ForeignKey('subscribers.id'), nullable=False)
    id_number = Column(String, nullable=False)
    
    # Relationships
    subscriber = relationship('Subscribers', back_populates='id_numbers')

class SubscriptionFunctions(db.Model):
    __tablename__ = 'subscription_functions'
    
    id = Column(Integer, primary_key=True)
    subscription_id = Column(Integer, ForeignKey('subscriptions.id'), nullable=False)
    company_research = Column(Boolean, nullable=False, default=True)
    portfolio = Column(Boolean, nullable=False, default=False)
    
    # Relationships
    subscription = relationship('Subscriptions', back_populates='functions')

class Referals(db.Model):
    __tablename__ = 'referals'
    
    id = Column(Integer, primary_key=True)
    subscriber_id = Column(Integer, ForeignKey('subscribers.id'), nullable=False)
    active = Column(Boolean, nullable=False, default=True)
    refferal_code = Column(String, nullable=False)
    count = Column(Integer, nullable=False)
    expiry = Column(Date, nullable=False)
    
    # Relationships
    subscriber = relationship('Subscribers', back_populates='referals')

class HTMLWebView(db.Model):
    __tablename__ = 'htmlwebview'
    __table_args__ = (UniqueConstraint('html_summary_path', name='_html_summary_path_uc'), UniqueConstraint('html_detailed_path', name='_html_detailed_path_uc'))
    
    id = Column(Integer, primary_key=True)
    display_date = Column(String, nullable=False)
    report_type = Column(String, nullable=False)
    html_summary_path = Column(String, nullable=False)
    html_detailed_path = Column(String, nullable=False)
    pdf_summary_path = Column(String, nullable=False)
    pdf_detailed_path = Column(String, nullable=False)
    actual_run_date = Column(Date, nullable=False)
    subscriber_id = Column(Integer, ForeignKey('subscribers.id'), nullable=False)
    
    # Relationship to link back to subscriber
    subscriber = relationship('Subscribers', back_populates='html_web_views')

class PortfolioTransactionHistory(db.Model):
    __tablename__ = 'portfolio_transaction_history'
    
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    share = Column(String, nullable=False)
    action = Column(String, nullable=False)
    value = Column(String, nullable=False)

class PortfolioTracker(db.Model):
    __tablename__ = 'portfolio_tracker'
    __table_args__ = (UniqueConstraint('ticker', name='_ticker_uc'),)
    
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False)
    weight = Column(String, nullable=False)
    comment = Column(String, nullable=False)
