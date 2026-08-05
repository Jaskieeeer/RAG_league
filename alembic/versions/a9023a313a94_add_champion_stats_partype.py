"""add champion stats partype

Revision ID: a9023a313a94
Revises: c7a1d4b90e52
Create Date: 2026-08-05 15:58:43.391724

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a9023a313a94"
down_revision: str | Sequence[str] | None = "c7a1d4b90e52"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the champion_stats partype column naming each champion's primary resource."""
    op.add_column(
        "champion_stats",
        sa.Column("partype", sa.String(length=32), nullable=False, server_default=""),
    )
    op.alter_column("champion_stats", "partype", server_default=None)


def downgrade() -> None:
    """Drop the partype column."""
    op.drop_column("champion_stats", "partype")
