from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    String,
    Table,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all lolrag ORM models."""


# ---------- association tables ----------

champion_role = Table(
    "champion_role",
    Base.metadata,
    Column("champion_slug", ForeignKey("champions.slug", ondelete="CASCADE"), primary_key=True),
    Column("role_slug", ForeignKey("roles.slug", ondelete="CASCADE"), primary_key=True),
)

champion_related = Table(
    "champion_related",
    Base.metadata,
    Column("champion_slug", ForeignKey("champions.slug", ondelete="CASCADE"), primary_key=True),
    Column("related_slug", ForeignKey("champions.slug", ondelete="CASCADE"), primary_key=True),
)

story_champion = Table(
    "story_champion",
    Base.metadata,
    Column("story_slug", ForeignKey("stories.slug", ondelete="CASCADE"), primary_key=True),
    Column("champion_slug", ForeignKey("champions.slug", ondelete="CASCADE"), primary_key=True),
)

item_tag = Table(
    "item_tag",
    Base.metadata,
    Column("item_id", ForeignKey("items.ddragon_id", ondelete="CASCADE"), primary_key=True),
    Column("tag", String(64), primary_key=True),
)


# ---------- entities ----------


class Faction(Base):
    """A lore faction that champions belong to.

    Args:
        slug: Unique faction identifier, primary key.
        name: Human-readable faction name.
        overview: Long-form faction description, nullable.
    """

    __tablename__ = "factions"

    slug: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    overview: Mapped[str | None] = mapped_column(Text, nullable=True)

    champions: Mapped[list["Champion"]] = relationship(back_populates="faction")


class Role(Base):
    """A champion role/class tag, e.g. Fighter or Mage.

    Args:
        slug: Unique role identifier, primary key.
        name: Human-readable role name.
    """

    __tablename__ = "roles"

    slug: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))


class Champion(Base):
    """A playable League of Legends champion.

    Args:
        slug: Unique champion identifier, primary key.
        ddragon_key: Data Dragon key for this champion, unique.
        name: Champion display name.
        title: Champion title, e.g. "the Darkin Blade".
        faction_slug: Foreign key to the champion's lore faction, not null.
        bio_full: Full champion biography text.
        bio_short: Short champion biography text, nullable.
        release_date: Champion release date, nullable.
    """

    __tablename__ = "champions"

    slug: Mapped[str] = mapped_column(String(64), primary_key=True)
    ddragon_key: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(128))
    title: Mapped[str] = mapped_column(String(256))
    faction_slug: Mapped[str] = mapped_column(ForeignKey("factions.slug"), nullable=False)
    bio_full: Mapped[str] = mapped_column(Text)
    bio_short: Mapped[str | None] = mapped_column(Text, nullable=True)
    release_date: Mapped[datetime | None] = mapped_column(nullable=True)

    faction: Mapped["Faction"] = relationship(back_populates="champions")
    roles: Mapped[list["Role"]] = relationship(secondary=champion_role)
    abilities: Mapped[list["Ability"]] = relationship(
        back_populates="champion", cascade="all, delete-orphan"
    )
    stories: Mapped[list["Story"]] = relationship(
        secondary=story_champion, back_populates="champions"
    )
    related: Mapped[list["Champion"]] = relationship(
        "Champion",
        secondary=champion_related,
        primaryjoin="Champion.slug == champion_related.c.champion_slug",
        secondaryjoin="Champion.slug == champion_related.c.related_slug",
    )


class Ability(Base):
    """A champion ability, either a passive or a Q/W/E/R spell.

    Args:
        id: Surrogate primary key.
        champion_slug: Foreign key to the owning champion, cascades on delete.
        slot: Ability slot, one of P, Q, W, E, R.
        name: Ability name.
        description: Ability description text.
        tooltip: Ability tooltip text, nullable.
    """

    __tablename__ = "abilities"
    __table_args__ = (
        UniqueConstraint("champion_slug", "slot"),
        CheckConstraint("slot IN ('P','Q','W','E','R')", name="ck_abilities_slot"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    champion_slug: Mapped[str] = mapped_column(ForeignKey("champions.slug", ondelete="CASCADE"))
    slot: Mapped[str] = mapped_column(String(1))
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text)
    tooltip: Mapped[str | None] = mapped_column(Text, nullable=True)

    champion: Mapped["Champion"] = relationship(back_populates="abilities")


class Story(Base):
    """A long-form champion lore story.

    Args:
        slug: Unique story identifier, primary key.
        title: Story title.
        author: Story author, nullable.
        word_count: Word count of the story content.
        section_count: Number of sections in the story.
        content: Full story text.
        release_date: Story release date, nullable.
    """

    __tablename__ = "stories"

    slug: Mapped[str] = mapped_column(String(128), primary_key=True)
    title: Mapped[str] = mapped_column(String(256))
    author: Mapped[str | None] = mapped_column(String(128), nullable=True)
    word_count: Mapped[int] = mapped_column()
    section_count: Mapped[int] = mapped_column()
    content: Mapped[str] = mapped_column(Text)
    release_date: Mapped[datetime | None] = mapped_column(nullable=True)

    champions: Mapped[list["Champion"]] = relationship(
        secondary=story_champion, back_populates="stories"
    )


class Item(Base):
    """A purchasable in-game item.

    Args:
        ddragon_id: Data Dragon item id, primary key.
        name: Item name.
        description: Full item description text.
        plaintext: Short plaintext item description, nullable.
        gold_total: Total purchase gold cost.
        gold_base: Base gold cost, excluding component value.
    """

    __tablename__ = "items"

    ddragon_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text)
    plaintext: Mapped[str | None] = mapped_column(Text, nullable=True)
    gold_total: Mapped[int] = mapped_column()
    gold_base: Mapped[int] = mapped_column()


class RunePath(Base):
    """A rune tree/path, e.g. Domination or Precision.

    Args:
        id: Surrogate primary key.
        key: Rune path key from Data Dragon.
        name: Rune path display name.
    """

    __tablename__ = "rune_paths"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(128))

    runes: Mapped[list["Rune"]] = relationship(back_populates="path")


class Rune(Base):
    """A single rune within a rune path.

    Args:
        id: Surrogate primary key.
        path_id: Foreign key to the owning rune path, cascades on delete.
        key: Rune key from Data Dragon.
        name: Rune display name.
        short_desc: Short rune description.
        long_desc: Long rune description.
        slot_index: Position of the rune within its slot/row.
    """

    __tablename__ = "runes"

    id: Mapped[int] = mapped_column(primary_key=True)
    path_id: Mapped[int] = mapped_column(ForeignKey("rune_paths.id", ondelete="CASCADE"))
    key: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(128))
    short_desc: Mapped[str] = mapped_column(Text)
    long_desc: Mapped[str] = mapped_column(Text)
    slot_index: Mapped[int] = mapped_column()

    path: Mapped["RunePath"] = relationship(back_populates="runes")


class SummonerSpell(Base):
    """A summoner spell, e.g. Flash or Ignite.

    Args:
        id: Data Dragon summoner spell id, primary key.
        key: Summoner spell key from Data Dragon.
        name: Summoner spell display name.
        description: Summoner spell description.
        cooldown: Cooldown in seconds, nullable.
        summoner_level: Minimum summoner level required, nullable.
    """

    __tablename__ = "summoner_spells"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    key: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text)
    cooldown: Mapped[int | None] = mapped_column(nullable=True)
    summoner_level: Mapped[int | None] = mapped_column(nullable=True)


# ---------- retrieval ----------


class Document(Base):
    """A retrievable unit of source content, indexed for RAG.

    Exactly one of the seven entity foreign keys is non-null; it identifies
    which entity this document was generated from.

    Args:
        id: Surrogate primary key.
        doc_key: Unique deterministic document identifier.
        collection: Logical collection this document belongs to, one of
            'abilities', 'equipment', 'lore'.
        champion_slug: Foreign key to the source champion, nullable.
        story_slug: Foreign key to the source story, nullable.
        faction_slug: Foreign key to the source faction, nullable.
        ability_id: Foreign key to the source ability, nullable.
        item_id: Foreign key to the source item, nullable.
        rune_id: Foreign key to the source rune, nullable.
        summoner_spell_id: Foreign key to the source summoner spell, nullable.
        title: Document title.
        source: Human-readable provenance string for the document.
        content: Full document text.
        indexed_at: Timestamp the document was last indexed.
    """

    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "(champion_slug IS NOT NULL)::int"
            " + (story_slug IS NOT NULL)::int"
            " + (faction_slug IS NOT NULL)::int"
            " + (ability_id IS NOT NULL)::int"
            " + (item_id IS NOT NULL)::int"
            " + (rune_id IS NOT NULL)::int"
            " + (summoner_spell_id IS NOT NULL)::int = 1",
            name="ck_documents_exactly_one_entity",
        ),
        CheckConstraint(
            "collection IN ('abilities','equipment','lore')", name="ck_documents_collection"
        ),
        Index(
            "ix_documents_champion_slug",
            "champion_slug",
            postgresql_where=text("champion_slug IS NOT NULL"),
        ),
        Index(
            "ix_documents_story_slug",
            "story_slug",
            postgresql_where=text("story_slug IS NOT NULL"),
        ),
        Index(
            "ix_documents_faction_slug",
            "faction_slug",
            postgresql_where=text("faction_slug IS NOT NULL"),
        ),
        Index(
            "ix_documents_ability_id",
            "ability_id",
            postgresql_where=text("ability_id IS NOT NULL"),
        ),
        Index(
            "ix_documents_item_id",
            "item_id",
            postgresql_where=text("item_id IS NOT NULL"),
        ),
        Index(
            "ix_documents_rune_id",
            "rune_id",
            postgresql_where=text("rune_id IS NOT NULL"),
        ),
        Index(
            "ix_documents_summoner_spell_id",
            "summoner_spell_id",
            postgresql_where=text("summoner_spell_id IS NOT NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    doc_key: Mapped[str] = mapped_column(String(64), unique=True)
    collection: Mapped[str] = mapped_column(String(32))

    champion_slug: Mapped[str | None] = mapped_column(
        ForeignKey("champions.slug", ondelete="CASCADE"), nullable=True
    )
    story_slug: Mapped[str | None] = mapped_column(
        ForeignKey("stories.slug", ondelete="CASCADE"), nullable=True
    )
    faction_slug: Mapped[str | None] = mapped_column(
        ForeignKey("factions.slug", ondelete="CASCADE"), nullable=True
    )
    ability_id: Mapped[int | None] = mapped_column(
        ForeignKey("abilities.id", ondelete="CASCADE"), nullable=True
    )
    item_id: Mapped[str | None] = mapped_column(
        ForeignKey("items.ddragon_id", ondelete="CASCADE"), nullable=True
    )
    rune_id: Mapped[int | None] = mapped_column(
        ForeignKey("runes.id", ondelete="CASCADE"), nullable=True
    )
    summoner_spell_id: Mapped[str | None] = mapped_column(
        ForeignKey("summoner_spells.id", ondelete="CASCADE"), nullable=True
    )

    title: Mapped[str] = mapped_column(String(256))
    source: Mapped[str] = mapped_column(String(512))
    content: Mapped[str] = mapped_column(Text)
    indexed_at: Mapped[datetime] = mapped_column()

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """A single embedded text chunk belonging to a document.

    Args:
        id: Surrogate primary key.
        document_id: Foreign key to the owning document, cascades on delete.
        chunk_index: Position of the chunk within its document.
        content: Chunk text content.
        embedding: 384-dimensional dense embedding vector for the chunk.
    """

    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column()
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))

    document: Mapped["Document"] = relationship(back_populates="chunks")
