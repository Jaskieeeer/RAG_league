"""add champion stats item variant columns and summoner spell modes

Revision ID: c7a1d4b90e52
Revises: fe8fe5748030
Create Date: 2026-08-04 10:12:44.187204

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "c7a1d4b90e52"
down_revision: str | Sequence[str] | None = "fe8fe5748030"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_COLLECTIONS = "collection IN ('abilities','equipment','lore')"
NEW_COLLECTIONS = "collection IN ('abilities','champion_stats','equipment','lore')"


def upgrade() -> None:
    """Add champion_stats, the item variant columns, spell modes and the new collection."""
    op.create_table(
        "champion_stats",
        sa.Column("champion_slug", sa.String(length=64), nullable=False),
        sa.Column("hp", sa.Float(), nullable=False),
        sa.Column("hp_per_level", sa.Float(), nullable=False),
        sa.Column("mp", sa.Float(), nullable=False),
        sa.Column("mp_per_level", sa.Float(), nullable=False),
        sa.Column("move_speed", sa.Float(), nullable=False),
        sa.Column("armor", sa.Float(), nullable=False),
        sa.Column("armor_per_level", sa.Float(), nullable=False),
        sa.Column("spell_block", sa.Float(), nullable=False),
        sa.Column("spell_block_per_level", sa.Float(), nullable=False),
        sa.Column("attack_range", sa.Float(), nullable=False),
        sa.Column("hp_regen", sa.Float(), nullable=False),
        sa.Column("hp_regen_per_level", sa.Float(), nullable=False),
        sa.Column("mp_regen", sa.Float(), nullable=False),
        sa.Column("mp_regen_per_level", sa.Float(), nullable=False),
        sa.Column("crit", sa.Float(), nullable=False),
        sa.Column("crit_per_level", sa.Float(), nullable=False),
        sa.Column("attack_damage", sa.Float(), nullable=False),
        sa.Column("attack_damage_per_level", sa.Float(), nullable=False),
        sa.Column("attack_speed", sa.Float(), nullable=False),
        sa.Column("attack_speed_per_level", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["champion_slug"], ["champions.slug"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("champion_slug"),
    )
    op.add_column("items", sa.Column("variant_of_id", sa.String(length=16), nullable=True))
    op.add_column("items", sa.Column("display_name_id", sa.String(length=16), nullable=True))
    op.add_column(
        "summoner_spells",
        sa.Column(
            "modes",
            postgresql.ARRAY(sa.String(length=64)),
            nullable=False,
            server_default="{}",
        ),
    )
    op.alter_column("summoner_spells", "modes", server_default=None)
    op.drop_constraint("ck_documents_collection", "documents", type_="check")
    op.create_check_constraint("ck_documents_collection", "documents", NEW_COLLECTIONS)


def downgrade() -> None:
    """Restore the old collection set and drop everything this revision added."""
    op.drop_constraint("ck_documents_collection", "documents", type_="check")
    op.create_check_constraint("ck_documents_collection", "documents", OLD_COLLECTIONS)
    op.drop_column("summoner_spells", "modes")
    op.drop_column("items", "display_name_id")
    op.drop_column("items", "variant_of_id")
    op.drop_table("champion_stats")
