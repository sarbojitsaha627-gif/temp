"""Add settlement_items table

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-08-31 09:56:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'settlement_items',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('settlement_id', sa.UUID(), nullable=False),
        sa.Column(
            'order_id', sa.String(length=64), nullable=False,
            comment='Claimed order_id from gateway report (references orders.order_id)',
        ),
        sa.Column(
            'gross_paise', sa.BigInteger(), nullable=False,
            comment="This line item's claimed gross amount in paise",
        ),
        sa.Column(
            'fee_paise', sa.BigInteger(), nullable=False,
            comment="This line item's claimed platform/MDR fee in paise",
        ),
        sa.Column(
            'tax_paise', sa.BigInteger(), nullable=False,
            comment="This line item's claimed GST on fee in paise",
        ),
        sa.Column(
            'net_paise', sa.BigInteger(), nullable=False,
            comment="This line item's claimed net payout in paise (gross - fee - tax)",
        ),
        sa.Column(
            'created_at', sa.DateTime(timezone=True),
            server_default=sa.text('now()'), nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['settlement_id'], ['settlements.id'], ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_settlement_items_settlement_id'),
        'settlement_items', ['settlement_id'], unique=False,
    )
    op.create_index(
        op.f('ix_settlement_items_order_id'),
        'settlement_items', ['order_id'], unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_settlement_items_order_id'), table_name='settlement_items')
    op.drop_index(op.f('ix_settlement_items_settlement_id'), table_name='settlement_items')
    op.drop_table('settlement_items')
