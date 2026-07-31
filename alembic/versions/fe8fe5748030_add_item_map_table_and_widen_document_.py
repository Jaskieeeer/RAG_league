"""add item map table and widen document doc key

Revision ID: fe8fe5748030
Revises: 41050e898cbc
Create Date: 2026-07-31 15:05:50.300988

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "fe8fe5748030"
down_revision: str | Sequence[str] | None = "41050e898cbc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the item shop flags and item_map, and widen documents.doc_key."""
    op.add_column(
        "items",
        sa.Column("purchasable", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.add_column(
        "items",
        sa.Column("in_store", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.alter_column("items", "purchasable", server_default=None)
    op.alter_column("items", "in_store", server_default=None)
    op.create_table(
        "item_map",
        sa.Column("item_id", sa.String(length=16), nullable=False),
        sa.Column("map_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["item_id"], ["items.ddragon_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("item_id", "map_id"),
    )
    op.alter_column(
        "documents",
        "doc_key",
        existing_type=sa.String(length=64),
        type_=sa.String(length=160),
        existing_nullable=False,
    )


def downgrade() -> None:
    """Narrow documents.doc_key back, drop item_map and the item shop flags."""
    op.alter_column(
        "documents",
        "doc_key",
        existing_type=sa.String(length=160),
        type_=sa.String(length=64),
        existing_nullable=False,
    )
    op.drop_table("item_map")
    op.drop_column("items", "in_store")
    op.drop_column("items", "purchasable")
