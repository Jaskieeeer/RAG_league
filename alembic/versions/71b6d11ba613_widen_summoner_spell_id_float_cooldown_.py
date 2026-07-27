"""widen summoner spell id, float cooldown, item component quantity

Revision ID: 71b6d11ba613
Revises: 5425a6258efb
Create Date: 2026-07-27 22:12:59.775323

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "71b6d11ba613"
down_revision: str | Sequence[str] | None = "5425a6258efb"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Widen the summoner spell id, float the cooldown, add item component quantity."""
    op.alter_column(
        "documents",
        "summoner_spell_id",
        existing_type=sa.VARCHAR(length=16),
        type_=sa.String(length=64),
        existing_nullable=True,
    )
    op.add_column(
        "item_components",
        sa.Column("quantity", sa.Integer(), server_default="1", nullable=False),
    )
    op.alter_column(
        "summoner_spells",
        "id",
        existing_type=sa.VARCHAR(length=16),
        type_=sa.String(length=64),
        existing_nullable=False,
    )
    op.alter_column(
        "summoner_spells",
        "cooldown",
        existing_type=sa.INTEGER(),
        type_=sa.Float(),
        existing_nullable=True,
    )


def downgrade() -> None:
    """Reverse the item component quantity, the cooldown float and the id widening."""
    op.alter_column(
        "summoner_spells",
        "cooldown",
        existing_type=sa.Float(),
        type_=sa.INTEGER(),
        existing_nullable=True,
    )
    op.alter_column(
        "summoner_spells",
        "id",
        existing_type=sa.String(length=64),
        type_=sa.VARCHAR(length=16),
        existing_nullable=False,
    )
    op.drop_column("item_components", "quantity")
    op.alter_column(
        "documents",
        "summoner_spell_id",
        existing_type=sa.String(length=64),
        type_=sa.VARCHAR(length=16),
        existing_nullable=True,
    )
