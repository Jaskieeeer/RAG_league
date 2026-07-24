"""initial schema

Revision ID: 396e70d3d253
Revises:
Create Date: 2026-07-24 05:24:58.603875

"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "396e70d3d253"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the pgvector extension and all lolrag tables and indexes."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "factions",
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("overview", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("slug"),
    )
    op.create_table(
        "items",
        sa.Column("ddragon_id", sa.String(length=16), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("plaintext", sa.Text(), nullable=True),
        sa.Column("gold_total", sa.Integer(), nullable=False),
        sa.Column("gold_base", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("ddragon_id"),
    )
    op.create_table(
        "roles",
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.PrimaryKeyConstraint("slug"),
    )
    op.create_table(
        "rune_paths",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "stories",
        sa.Column("slug", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=256), nullable=False),
        sa.Column("author", sa.String(length=128), nullable=True),
        sa.Column("word_count", sa.Integer(), nullable=False),
        sa.Column("section_count", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("release_date", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("slug"),
    )
    op.create_table(
        "summoner_spells",
        sa.Column("id", sa.String(length=16), nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("cooldown", sa.Integer(), nullable=True),
        sa.Column("summoner_level", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "champions",
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("ddragon_key", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=256), nullable=False),
        sa.Column("faction_slug", sa.String(length=64), nullable=False),
        sa.Column("bio_full", sa.Text(), nullable=False),
        sa.Column("bio_short", sa.Text(), nullable=True),
        sa.Column("release_date", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["faction_slug"],
            ["factions.slug"],
        ),
        sa.PrimaryKeyConstraint("slug"),
        sa.UniqueConstraint("ddragon_key"),
    )
    op.create_table(
        "item_tag",
        sa.Column("item_id", sa.String(length=16), nullable=False),
        sa.Column("tag", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["item_id"], ["items.ddragon_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("item_id", "tag"),
    )
    op.create_table(
        "runes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("path_id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("short_desc", sa.Text(), nullable=False),
        sa.Column("long_desc", sa.Text(), nullable=False),
        sa.Column("slot_index", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["path_id"], ["rune_paths.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "abilities",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("champion_slug", sa.String(length=64), nullable=False),
        sa.Column("slot", sa.String(length=1), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("tooltip", sa.Text(), nullable=True),
        sa.CheckConstraint("slot IN ('P','Q','W','E','R')", name="ck_abilities_slot"),
        sa.ForeignKeyConstraint(["champion_slug"], ["champions.slug"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("champion_slug", "slot"),
    )
    op.create_table(
        "champion_related",
        sa.Column("champion_slug", sa.String(length=64), nullable=False),
        sa.Column("related_slug", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["champion_slug"], ["champions.slug"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["related_slug"], ["champions.slug"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("champion_slug", "related_slug"),
    )
    op.create_table(
        "champion_role",
        sa.Column("champion_slug", sa.String(length=64), nullable=False),
        sa.Column("role_slug", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["champion_slug"], ["champions.slug"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["role_slug"], ["roles.slug"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("champion_slug", "role_slug"),
    )
    op.create_table(
        "story_champion",
        sa.Column("story_slug", sa.String(length=128), nullable=False),
        sa.Column("champion_slug", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["champion_slug"], ["champions.slug"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["story_slug"], ["stories.slug"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("story_slug", "champion_slug"),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("doc_key", sa.String(length=64), nullable=False),
        sa.Column("collection", sa.String(length=32), nullable=False),
        sa.Column("champion_slug", sa.String(length=64), nullable=True),
        sa.Column("story_slug", sa.String(length=128), nullable=True),
        sa.Column("faction_slug", sa.String(length=64), nullable=True),
        sa.Column("ability_id", sa.Integer(), nullable=True),
        sa.Column("item_id", sa.String(length=16), nullable=True),
        sa.Column("rune_id", sa.Integer(), nullable=True),
        sa.Column("summoner_spell_id", sa.String(length=16), nullable=True),
        sa.Column("title", sa.String(length=256), nullable=False),
        sa.Column("source", sa.String(length=512), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("indexed_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "collection IN ('abilities','equipment','lore')", name="ck_documents_collection"
        ),
        sa.CheckConstraint(
            "(champion_slug IS NOT NULL)::int"
            " + (story_slug IS NOT NULL)::int"
            " + (faction_slug IS NOT NULL)::int"
            " + (ability_id IS NOT NULL)::int"
            " + (item_id IS NOT NULL)::int"
            " + (rune_id IS NOT NULL)::int"
            " + (summoner_spell_id IS NOT NULL)::int = 1",
            name="ck_documents_exactly_one_entity",
        ),
        sa.ForeignKeyConstraint(["ability_id"], ["abilities.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["champion_slug"], ["champions.slug"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["faction_slug"], ["factions.slug"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["item_id"], ["items.ddragon_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["rune_id"], ["runes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["story_slug"], ["stories.slug"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["summoner_spell_id"], ["summoner_spells.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("doc_key"),
    )
    op.create_index(
        "ix_documents_ability_id",
        "documents",
        ["ability_id"],
        unique=False,
        postgresql_where=sa.text("ability_id IS NOT NULL"),
    )
    op.create_index(
        "ix_documents_champion_slug",
        "documents",
        ["champion_slug"],
        unique=False,
        postgresql_where=sa.text("champion_slug IS NOT NULL"),
    )
    op.create_index(
        "ix_documents_faction_slug",
        "documents",
        ["faction_slug"],
        unique=False,
        postgresql_where=sa.text("faction_slug IS NOT NULL"),
    )
    op.create_index(
        "ix_documents_item_id",
        "documents",
        ["item_id"],
        unique=False,
        postgresql_where=sa.text("item_id IS NOT NULL"),
    )
    op.create_index(
        "ix_documents_rune_id",
        "documents",
        ["rune_id"],
        unique=False,
        postgresql_where=sa.text("rune_id IS NOT NULL"),
    )
    op.create_index(
        "ix_documents_story_slug",
        "documents",
        ["story_slug"],
        unique=False,
        postgresql_where=sa.text("story_slug IS NOT NULL"),
    )
    op.create_index(
        "ix_documents_summoner_spell_id",
        "documents",
        ["summoner_spell_id"],
        unique=False,
        postgresql_where=sa.text("summoner_spell_id IS NOT NULL"),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "chunk_index"),
    )
    op.execute(
        "CREATE INDEX ix_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    """Drop all lolrag tables and indexes, leaving the pgvector extension installed."""
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding")
    op.drop_table("chunks")
    op.drop_index(
        "ix_documents_summoner_spell_id",
        table_name="documents",
        postgresql_where=sa.text("summoner_spell_id IS NOT NULL"),
    )
    op.drop_index(
        "ix_documents_story_slug",
        table_name="documents",
        postgresql_where=sa.text("story_slug IS NOT NULL"),
    )
    op.drop_index(
        "ix_documents_rune_id",
        table_name="documents",
        postgresql_where=sa.text("rune_id IS NOT NULL"),
    )
    op.drop_index(
        "ix_documents_item_id",
        table_name="documents",
        postgresql_where=sa.text("item_id IS NOT NULL"),
    )
    op.drop_index(
        "ix_documents_faction_slug",
        table_name="documents",
        postgresql_where=sa.text("faction_slug IS NOT NULL"),
    )
    op.drop_index(
        "ix_documents_champion_slug",
        table_name="documents",
        postgresql_where=sa.text("champion_slug IS NOT NULL"),
    )
    op.drop_index(
        "ix_documents_ability_id",
        table_name="documents",
        postgresql_where=sa.text("ability_id IS NOT NULL"),
    )
    op.drop_table("documents")
    op.drop_table("story_champion")
    op.drop_table("champion_role")
    op.drop_table("champion_related")
    op.drop_table("abilities")
    op.drop_table("runes")
    op.drop_table("item_tag")
    op.drop_table("champions")
    op.drop_table("summoner_spells")
    op.drop_table("stories")
    op.drop_table("rune_paths")
    op.drop_table("roles")
    op.drop_table("items")
    op.drop_table("factions")
    # ### end Alembic commands ###
