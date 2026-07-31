"""add ability value stat formula

Revision ID: 41050e898cbc
Revises: 3c616fa04ada
Create Date: 2026-07-31 00:29:08.770874

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "41050e898cbc"
down_revision: str | Sequence[str] | None = "3c616fa04ada"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable ability_values.stat_formula column and its check."""
    op.add_column("ability_values", sa.Column("stat_formula", sa.String(length=16), nullable=True))
    op.create_check_constraint(
        "ck_ability_values_stat_formula",
        "ability_values",
        "stat_formula IN ('total','bonus')",
    )


def downgrade() -> None:
    """Drop the stat_formula check and the column it guards."""
    op.drop_constraint("ck_ability_values_stat_formula", "ability_values", type_="check")
    op.drop_column("ability_values", "stat_formula")
