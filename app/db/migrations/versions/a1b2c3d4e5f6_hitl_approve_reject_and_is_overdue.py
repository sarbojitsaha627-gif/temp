"""Split HitlStatus RESOLVED into APPROVED+REJECTED; add is_overdue to exception_staging

Revision ID: a1b2c3d4e5f6
Revises: 3de6cf028dad
Create Date: 2026-08-31 08:34:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '3de6cf028dad'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ── 1. HitlStatus enum: ensure APPROVED + REJECTED exist ──
    # The initial schema may already include them; ADD VALUE IF NOT EXISTS
    # is safe and idempotent.
    op.execute("ALTER TYPE hitl_status_enum ADD VALUE IF NOT EXISTS 'approved'")
    op.execute("ALTER TYPE hitl_status_enum ADD VALUE IF NOT EXISTS 'rejected'")

    # ── 2. Add is_overdue boolean to exception_staging ──
    op.add_column(
        'exception_staging',
        sa.Column(
            'is_overdue',
            sa.Boolean(),
            nullable=True,
            comment=(
                'For TIMING_SETTLEMENT_FLOAT only: '
                'False = PENDING_FLOAT (<=T+2), True = ANOMALOUS_FLOAT_OVERDUE (>T+2). '
                'NULL for all other categories.'
            ),
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    # ── 2. Drop is_overdue ──
    op.drop_column('exception_staging', 'is_overdue')

    # ── 1. Restore RESOLVED, drop APPROVED + REJECTED ──
    op.execute("ALTER TYPE hitl_status_enum ADD VALUE IF NOT EXISTS 'resolved'")
    op.execute("COMMIT")

    op.execute(
        "UPDATE match_groups SET hitl_status = 'resolved' "
        "WHERE hitl_status IN ('approved', 'rejected')"
    )

    op.execute("ALTER TYPE hitl_status_enum RENAME TO hitl_status_enum_old")
    op.execute(
        "CREATE TYPE hitl_status_enum AS ENUM "
        "('not_required', 'pending', 'resolved', 'eval_bypassed')"
    )
    op.execute(
        "ALTER TABLE match_groups "
        "ALTER COLUMN hitl_status TYPE hitl_status_enum "
        "USING hitl_status::text::hitl_status_enum"
    )
    op.execute("DROP TYPE hitl_status_enum_old")
