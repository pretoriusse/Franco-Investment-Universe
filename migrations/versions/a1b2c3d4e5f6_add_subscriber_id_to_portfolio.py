"""Add subscriber_id to portfolio tables and fix constraints

Revision ID: a1b2c3d4e5f6
Revises: 99f6b5224502
Create Date: 2026-06-07 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "99f6b5224502"
branch_labels = None
depends_on = None


def upgrade():
    # Add subscriber_id to portfolio_transaction_history
    with op.batch_alter_table("portfolio_transaction_history", schema=None) as batch_op:
        batch_op.add_column(sa.Column("subscriber_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_txn_subscriber", "subscribers", ["subscriber_id"], ["id"]
        )
        batch_op.create_unique_constraint(
            "_txn_uc", ["subscriber_id", "date", "share", "action"]
        )

    # Make subscriber_id NOT NULL after backfill (set to a default or leave nullable in prod)
    # Note: existing rows will have subscriber_id = NULL until manually backfilled.

    # Fix portfolio_tracker: drop old single-column unique constraint, add per-subscriber one
    with op.batch_alter_table("portfolio_tracker", schema=None) as batch_op:
        batch_op.add_column(sa.Column("subscriber_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_tracker_subscriber", "subscribers", ["subscriber_id"], ["id"]
        )
        try:
            batch_op.drop_constraint("_ticker_uc", type_="unique")
        except Exception:
            pass  # constraint may not exist with that exact name
        batch_op.create_unique_constraint(
            "_ticker_subscriber_uc", ["ticker", "subscriber_id"]
        )


def downgrade():
    with op.batch_alter_table("portfolio_tracker", schema=None) as batch_op:
        batch_op.drop_constraint("_ticker_subscriber_uc", type_="unique")
        batch_op.drop_constraint("fk_tracker_subscriber", type_="foreignkey")
        batch_op.drop_column("subscriber_id")
        batch_op.create_unique_constraint("_ticker_uc", ["ticker"])

    with op.batch_alter_table("portfolio_transaction_history", schema=None) as batch_op:
        batch_op.drop_constraint("_txn_uc", type_="unique")
        batch_op.drop_constraint("fk_txn_subscriber", type_="foreignkey")
        batch_op.drop_column("subscriber_id")
