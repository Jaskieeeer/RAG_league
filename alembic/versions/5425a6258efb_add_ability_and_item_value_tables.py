"""add ability and item value tables

Revision ID: 5425a6258efb
Revises: 8587595d6912
Create Date: 2026-07-27 16:22:29.424507

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "5425a6258efb"
down_revision: str | Sequence[str] | None = "8587595d6912"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add per-value numeric tables, ability max_rank, and rename story section_count."""
    op.create_table(
        "ability_values",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("ability_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("values", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column("damage_type", sa.String(length=16), nullable=True),
        sa.Column("display_as_percent", sa.Boolean(), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.CheckConstraint(
            "damage_type IN ('magic','physical','true')", name="ck_ability_values_damage_type"
        ),
        sa.CheckConstraint(
            "kind IN ('per_rank','by_level','scalar','ratio')", name="ck_ability_values_kind"
        ),
        sa.CheckConstraint("source IN ('ddragon','cdragon')", name="ck_ability_values_source"),
        sa.ForeignKeyConstraint(["ability_id"], ["abilities.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ability_id", "name", name="uq_ability_values_ability_id_name"),
    )
    op.create_table(
        "item_values",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("item_id", sa.String(length=16), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("values", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column("display_as_percent", sa.Boolean(), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.CheckConstraint(
            "kind IN ('per_rank','by_level','scalar','ratio')", name="ck_item_values_kind"
        ),
        sa.CheckConstraint("source IN ('ddragon','cdragon')", name="ck_item_values_source"),
        sa.ForeignKeyConstraint(["item_id"], ["items.ddragon_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("item_id", "name", name="uq_item_values_item_id_name"),
    )
    op.add_column("abilities", sa.Column("max_rank", sa.Integer(), nullable=True))
    op.alter_column(
        "stories",
        "section_count",
        new_column_name="subsection_count",
        existing_type=sa.Integer(),
        existing_nullable=False,
    )


def downgrade() -> None:
    """Reverse the story column rename, ability max_rank, and per-value numeric tables."""
    op.alter_column(
        "stories",
        "subsection_count",
        new_column_name="section_count",
        existing_type=sa.Integer(),
        existing_nullable=False,
    )
    op.drop_column("abilities", "max_rank")
    op.drop_table("item_values")
    op.drop_table("ability_values")
