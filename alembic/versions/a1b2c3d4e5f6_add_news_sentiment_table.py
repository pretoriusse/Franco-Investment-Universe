"""Add news_sentiment table

Revision ID: a1b2c3d4e5f6
Revises: 08252727159b
Create Date: 2026-06-06 00:00:00.000000

"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "08252727159b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "news_sentiment",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("ticker", sa.String(10), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("sentiment_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("article_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("positive_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("negative_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("neutral_count", sa.Integer(), nullable=False, server_default="0"),
        sa.ForeignKeyConstraint(["ticker"], ["stocks.code"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ticker", "date", name="uq_sentiment_ticker_date"),
    )
    op.create_index(
        "ix_news_sentiment_ticker_date", "news_sentiment", ["ticker", "date"]
    )


def downgrade() -> None:
    op.drop_index("ix_news_sentiment_ticker_date", table_name="news_sentiment")
    op.drop_table("news_sentiment")
