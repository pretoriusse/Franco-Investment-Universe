"""Add API key hash to subscribers and api_access flag to subscription_functions

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-07-02 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "b2c3d4e5f6a7"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("subscribers", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("api_key_hash", sa.String(length=64), nullable=True)
        )
        batch_op.create_unique_constraint("_api_key_hash_uc", ["api_key_hash"])

    with op.batch_alter_table("subscription_functions", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "api_access", sa.Boolean(), nullable=False, server_default=sa.false()
            )
        )


def downgrade():
    with op.batch_alter_table("subscription_functions", schema=None) as batch_op:
        batch_op.drop_column("api_access")

    with op.batch_alter_table("subscribers", schema=None) as batch_op:
        batch_op.drop_constraint("_api_key_hash_uc", type_="unique")
        batch_op.drop_column("api_key_hash")
