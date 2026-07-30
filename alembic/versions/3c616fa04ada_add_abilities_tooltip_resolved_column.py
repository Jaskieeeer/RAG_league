"""add abilities tooltip resolved column

Revision ID: 3c616fa04ada
Revises: b3eb5b0a6c6e
Create Date: 2026-07-30 21:18:51.713536

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "3c616fa04ada"
down_revision: str | Sequence[str] | None = "b3eb5b0a6c6e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable abilities.tooltip_resolved column."""
    op.add_column("abilities", sa.Column("tooltip_resolved", sa.Text(), nullable=True))


def downgrade() -> None:
    """Drop the abilities.tooltip_resolved column."""
    op.drop_column("abilities", "tooltip_resolved")
