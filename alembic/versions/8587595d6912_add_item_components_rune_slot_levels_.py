"""add item components, rune slot levels, cleaned text columns

Revision ID: 8587595d6912
Revises: 396e70d3d253
Create Date: 2026-07-26 11:45:52.606429

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "8587595d6912"
down_revision: str | Sequence[str] | None = "396e70d3d253"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add item build components, split rune slot_index, add cleaned text columns."""
    op.create_table(
        "item_components",
        sa.Column("item_id", sa.String(length=16), nullable=False),
        sa.Column("component_id", sa.String(length=16), nullable=False),
        sa.ForeignKeyConstraint(["component_id"], ["items.ddragon_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["item_id"], ["items.ddragon_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("item_id", "component_id"),
    )
    op.add_column("abilities", sa.Column("tooltip_text", sa.Text(), nullable=True))
    op.add_column("champions", sa.Column("bio_full_text", sa.Text(), nullable=False))
    op.add_column("champions", sa.Column("bio_short_text", sa.Text(), nullable=True))
    op.add_column("champions", sa.Column("playable", sa.Boolean(), nullable=False))
    op.alter_column("champions", "ddragon_key", existing_type=sa.VARCHAR(length=64), nullable=True)
    op.add_column("factions", sa.Column("overview_text", sa.Text(), nullable=True))
    op.add_column("items", sa.Column("description_text", sa.Text(), nullable=False))
    op.add_column("items", sa.Column("depth", sa.Integer(), nullable=True))
    op.add_column("runes", sa.Column("short_desc_text", sa.Text(), nullable=False))
    op.add_column("runes", sa.Column("long_desc_text", sa.Text(), nullable=False))
    op.add_column("runes", sa.Column("row_index", sa.Integer(), nullable=False))
    op.add_column("runes", sa.Column("position_index", sa.Integer(), nullable=False))
    op.create_unique_constraint(
        "uq_runes_path_id_row_index_position_index",
        "runes",
        ["path_id", "row_index", "position_index"],
    )
    op.drop_column("runes", "slot_index")
    op.add_column("stories", sa.Column("content_text", sa.Text(), nullable=False))
    op.add_column("summoner_spells", sa.Column("description_text", sa.Text(), nullable=False))


def downgrade() -> None:
    """Reverse item build components, rune slot_index split, and cleaned text columns."""
    op.drop_column("summoner_spells", "description_text")
    op.drop_column("stories", "content_text")
    op.add_column(
        "runes", sa.Column("slot_index", sa.Integer(), autoincrement=False, nullable=False)
    )
    op.drop_constraint("uq_runes_path_id_row_index_position_index", "runes", type_="unique")
    op.drop_column("runes", "position_index")
    op.drop_column("runes", "row_index")
    op.drop_column("runes", "long_desc_text")
    op.drop_column("runes", "short_desc_text")
    op.drop_column("items", "depth")
    op.drop_column("items", "description_text")
    op.drop_column("factions", "overview_text")
    op.alter_column("champions", "ddragon_key", existing_type=sa.VARCHAR(length=64), nullable=False)
    op.drop_column("champions", "playable")
    op.drop_column("champions", "bio_short_text")
    op.drop_column("champions", "bio_full_text")
    op.drop_column("abilities", "tooltip_text")
    op.drop_table("item_components")
