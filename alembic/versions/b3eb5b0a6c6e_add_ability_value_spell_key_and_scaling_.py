"""add ability value spell key and scaling stat

Revision ID: b3eb5b0a6c6e
Revises: 71b6d11ba613
Create Date: 2026-07-29 00:05:05.625411

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b3eb5b0a6c6e"
down_revision: str | Sequence[str] | None = "71b6d11ba613"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add ability value spell_key and scaling_stat, widening name and the unique key."""
    op.add_column("ability_values", sa.Column("spell_key", sa.String(length=64), nullable=False))
    op.add_column("ability_values", sa.Column("scaling_stat", sa.String(length=16), nullable=True))
    op.alter_column(
        "ability_values",
        "name",
        existing_type=sa.VARCHAR(length=64),
        type_=sa.String(length=128),
        existing_nullable=False,
    )
    op.drop_constraint("uq_ability_values_ability_id_name", "ability_values", type_="unique")
    op.create_unique_constraint(
        "uq_ability_values_ability_id_spell_key_name",
        "ability_values",
        ["ability_id", "spell_key", "name"],
    )
    op.create_check_constraint(
        "ck_ability_values_scaling_stat",
        "ability_values",
        "scaling_stat IN ('ap','ad','armor','magic_resist','attack_speed','crit','health')",
    )


def downgrade() -> None:
    """Reverse the scaling_stat check, the unique key, the widened name and both columns."""
    op.drop_constraint("ck_ability_values_scaling_stat", "ability_values", type_="check")
    op.drop_constraint(
        "uq_ability_values_ability_id_spell_key_name", "ability_values", type_="unique"
    )
    op.create_unique_constraint(
        "uq_ability_values_ability_id_name", "ability_values", ["ability_id", "name"]
    )
    op.alter_column(
        "ability_values",
        "name",
        existing_type=sa.String(length=128),
        type_=sa.VARCHAR(length=64),
        existing_nullable=False,
    )
    op.drop_column("ability_values", "scaling_stat")
    op.drop_column("ability_values", "spell_key")
