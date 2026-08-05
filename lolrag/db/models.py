from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY
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

item_map = Table(
    "item_map",
    Base.metadata,
    Column("item_id", ForeignKey("items.ddragon_id", ondelete="CASCADE"), primary_key=True),
    Column("map_id", Integer, primary_key=True),
)

item_components = Table(
    "item_components",
    Base.metadata,
    Column("item_id", ForeignKey("items.ddragon_id", ondelete="CASCADE"), primary_key=True),
    Column("component_id", ForeignKey("items.ddragon_id", ondelete="CASCADE"), primary_key=True),
    Column("quantity", Integer, nullable=False, server_default="1"),
)


# ---------- entities ----------


class Faction(Base):
    """A lore faction that champions belong to.

    Args:
        slug: Unique faction identifier, primary key.
        name: Human-readable faction name.
        overview: Long-form faction description, nullable.
        overview_text: Markup-stripped form of overview, used for embedding,
            nullable.
    """

    __tablename__ = "factions"

    slug: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    overview: Mapped[str | None] = mapped_column(Text, nullable=True)
    overview_text: Mapped[str | None] = mapped_column(Text, nullable=True)

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
        ddragon_key: Data Dragon key for this champion, unique, nullable for
            lore-only characters with no Data Dragon entry.
        name: Champion display name.
        title: Champion title, e.g. "the Darkin Blade".
        faction_slug: Foreign key to the champion's lore faction, not null.
        bio_full: Full champion biography text.
        bio_full_text: Markup-stripped form of bio_full, used for embedding.
        bio_short: Short champion biography text, nullable.
        bio_short_text: Markup-stripped form of bio_short, used for
            embedding, nullable.
        playable: Whether the character is a playable champion; False for
            lore-only characters.
        release_date: Champion release date, nullable.
    """

    __tablename__ = "champions"

    slug: Mapped[str] = mapped_column(String(64), primary_key=True)
    ddragon_key: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)
    name: Mapped[str] = mapped_column(String(128))
    title: Mapped[str] = mapped_column(String(256))
    faction_slug: Mapped[str] = mapped_column(ForeignKey("factions.slug"), nullable=False)
    bio_full: Mapped[str] = mapped_column(Text)
    bio_full_text: Mapped[str] = mapped_column(Text)
    bio_short: Mapped[str | None] = mapped_column(Text, nullable=True)
    bio_short_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    playable: Mapped[bool] = mapped_column()
    release_date: Mapped[datetime | None] = mapped_column(nullable=True)

    faction: Mapped["Faction"] = relationship(back_populates="champions")
    roles: Mapped[list["Role"]] = relationship(secondary=champion_role, viewonly=True)
    abilities: Mapped[list["Ability"]] = relationship(
        back_populates="champion", cascade="all, delete-orphan"
    )
    stories: Mapped[list["Story"]] = relationship(
        secondary=story_champion, back_populates="champions", viewonly=True
    )
    related: Mapped[list["Champion"]] = relationship(
        "Champion",
        secondary=champion_related,
        primaryjoin="Champion.slug == champion_related.c.champion_slug",
        secondaryjoin="Champion.slug == champion_related.c.related_slug",
        viewonly=True,
    )


class ChampionStats(Base):
    """The base statistics Data Dragon publishes for one playable champion.

    Held apart from Champion rather than folded into it because the champion
    roster comes from Riot Universe and includes lore-only characters that have
    no Data Dragon entry and therefore no statistics at all; twenty nullable
    columns on Champion would state that those characters have unknown values
    when in truth they have none.

    Every per-level column is the increment the source publishes, not the value
    at any particular level: Riot's own growth curve is not a plain multiple of
    it, so nothing here may be summed into a level-18 total.

    Args:
        champion_slug: Foreign key to the champion, primary key, cascades on
            delete.
        partype: Name Data Dragon publishes for the champion's primary resource,
            e.g. "Mana", "Fury" or "Blood Well"; the literal "None" or an empty
            string for the champions that pay no resource at all. This column is
            the only authority on whether a resource exists, because mp carries
            a sentinel for several champions it says have none: 10000 for Viego
            and 60 for Belveth.
        hp: Base health.
        hp_per_level: Published health growth per level.
        mp: Base value of the champion's primary resource, meaningful only when
            partype names one; a champion whose partype names a resource with no
            published maximum carries 0 here.
        mp_per_level: Published primary-resource growth per level.
        move_speed: Base movement speed.
        armor: Base armor.
        armor_per_level: Published armor growth per level.
        spell_block: Base magic resistance.
        spell_block_per_level: Published magic resistance growth per level.
        attack_range: Base attack range.
        hp_regen: Base health regeneration.
        hp_regen_per_level: Published health regeneration growth per level.
        mp_regen: Base primary-resource regeneration.
        mp_regen_per_level: Published primary-resource regeneration growth per
            level.
        crit: Base critical strike chance.
        crit_per_level: Published critical strike chance growth per level.
        attack_damage: Base attack damage.
        attack_damage_per_level: Published attack damage growth per level, which
            Data Dragon reports as 0 for every champion in 16.14.1 although
            champions demonstrably do gain attack damage per level. It is stored
            as published and deliberately not rendered into any document.
        attack_speed: Base attack speed, a ratio.
        attack_speed_per_level: Published attack speed growth per level, a
            percentage of the base ratio rather than a second ratio: 2.5 means
            2.5%, so it must never be added to attack_speed as it stands.
    """

    __tablename__ = "champion_stats"

    champion_slug: Mapped[str] = mapped_column(
        ForeignKey("champions.slug", ondelete="CASCADE"), primary_key=True
    )
    partype: Mapped[str] = mapped_column(String(32))
    hp: Mapped[float] = mapped_column(Float)
    hp_per_level: Mapped[float] = mapped_column(Float)
    mp: Mapped[float] = mapped_column(Float)
    mp_per_level: Mapped[float] = mapped_column(Float)
    move_speed: Mapped[float] = mapped_column(Float)
    armor: Mapped[float] = mapped_column(Float)
    armor_per_level: Mapped[float] = mapped_column(Float)
    spell_block: Mapped[float] = mapped_column(Float)
    spell_block_per_level: Mapped[float] = mapped_column(Float)
    attack_range: Mapped[float] = mapped_column(Float)
    hp_regen: Mapped[float] = mapped_column(Float)
    hp_regen_per_level: Mapped[float] = mapped_column(Float)
    mp_regen: Mapped[float] = mapped_column(Float)
    mp_regen_per_level: Mapped[float] = mapped_column(Float)
    crit: Mapped[float] = mapped_column(Float)
    crit_per_level: Mapped[float] = mapped_column(Float)
    attack_damage: Mapped[float] = mapped_column(Float)
    attack_damage_per_level: Mapped[float] = mapped_column(Float)
    attack_speed: Mapped[float] = mapped_column(Float)
    attack_speed_per_level: Mapped[float] = mapped_column(Float)

    champion: Mapped["Champion"] = relationship()


class Ability(Base):
    """A champion ability, either a passive or a Q/W/E/R spell.

    Args:
        id: Surrogate primary key.
        champion_slug: Foreign key to the owning champion, cascades on delete.
        slot: Ability slot, one of P, Q, W, E, R.
        name: Ability name.
        description: Ability description text.
        tooltip: Ability tooltip text, nullable.
        tooltip_text: Markup-stripped form of tooltip, used for embedding,
            nullable.
        tooltip_resolved: Community Dragon tooltip with every "@token@"
            substituted for the number it names, markup-stripped; NULL when the
            substitution was blocked by a token that cannot be answered from the
            spell in hand, or when the ability publishes no dynamic description
            at all, as every passive does.
        max_rank: Number of rank-up levels for this ability; NULL for passives.
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
    tooltip_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    tooltip_resolved: Mapped[str | None] = mapped_column(Text, nullable=True)
    max_rank: Mapped[int | None] = mapped_column(nullable=True)

    champion: Mapped["Champion"] = relationship(back_populates="abilities")
    values: Mapped[list["AbilityValue"]] = relationship(
        back_populates="ability", cascade="all, delete-orphan"
    )


class Story(Base):
    """A long-form champion lore story.

    Args:
        slug: Unique story identifier, primary key.
        title: Story title.
        author: Story author, nullable.
        word_count: Word count of the story content.
        subsection_count: Total number of content subsections across all story
            sections.
        content: Full story text.
        content_text: Markup-stripped form of content, used for embedding.
        release_date: Story release date, nullable.
    """

    __tablename__ = "stories"

    slug: Mapped[str] = mapped_column(String(128), primary_key=True)
    title: Mapped[str] = mapped_column(String(256))
    author: Mapped[str | None] = mapped_column(String(128), nullable=True)
    word_count: Mapped[int] = mapped_column()
    subsection_count: Mapped[int] = mapped_column()
    content: Mapped[str] = mapped_column(Text)
    content_text: Mapped[str] = mapped_column(Text)
    release_date: Mapped[datetime | None] = mapped_column(nullable=True)

    champions: Mapped[list["Champion"]] = relationship(
        secondary=story_champion, back_populates="stories", viewonly=True
    )


class Item(Base):
    """A purchasable in-game item.

    Args:
        ddragon_id: Data Dragon item id, primary key.
        name: Item name.
        description: Full item description text.
        description_text: Markup-stripped form of description, used for
            embedding.
        plaintext: Short plaintext item description, nullable.
        gold_total: Total purchase gold cost.
        gold_base: Base gold cost, excluding component value.
        depth: Number of build steps from base items, nullable; only a
            subset of items carry this in the source.
        purchasable: Source gold.purchasable flag, False for the engine-side
            entries a player can never buy.
        in_store: Source inStore flag, False for entries the shop never lists.
            The source omits it far more often than it publishes it, and an
            absent flag means True.
        variant_of_id: Item id this record declares itself a mode variant of,
            as published by Community Dragon; NULL when it declares none. Not a
            foreign key: it records what the source asserts, including an
            assertion about an id Data Dragon does not publish.
        display_name_id: Item id whose name string this record is published
            under, as named by the Community Dragon display-name locale key;
            NULL when the record is published under its own id. Not a foreign
            key, for the same reason.
    """

    __tablename__ = "items"

    ddragon_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text)
    description_text: Mapped[str] = mapped_column(Text)
    plaintext: Mapped[str | None] = mapped_column(Text, nullable=True)
    gold_total: Mapped[int] = mapped_column()
    gold_base: Mapped[int] = mapped_column()
    depth: Mapped[int | None] = mapped_column(nullable=True)
    purchasable: Mapped[bool] = mapped_column()
    in_store: Mapped[bool] = mapped_column()
    variant_of_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    display_name_id: Mapped[str | None] = mapped_column(String(16), nullable=True)

    components: Mapped[list["Item"]] = relationship(
        "Item",
        secondary=item_components,
        primaryjoin="Item.ddragon_id == item_components.c.item_id",
        secondaryjoin="Item.ddragon_id == item_components.c.component_id",
        back_populates="builds_into",
        viewonly=True,
    )
    builds_into: Mapped[list["Item"]] = relationship(
        "Item",
        secondary=item_components,
        primaryjoin="Item.ddragon_id == item_components.c.component_id",
        secondaryjoin="Item.ddragon_id == item_components.c.item_id",
        back_populates="components",
        viewonly=True,
    )
    values: Mapped[list["ItemValue"]] = relationship(
        back_populates="item", cascade="all, delete-orphan"
    )


class AbilityValue(Base):
    """A named numeric value published by the source for one ability.

    The kind field records the shape of the values array: 'per_rank' holds one
    entry per learnable rank, 'by_level' holds exactly two entries that are the
    level-1 and level-18 endpoints of a linear interpolation, 'scalar' holds a
    single value, and 'ratio' holds a single scaling coefficient.

    Args:
        id: Surrogate primary key.
        ability_id: Foreign key to the owning ability, cascades on delete.
        spell_key: Short source name of the spell that publishes the value, e.g.
            AatroxQ or AurelionSolR2. One ability spans a root spell and its
            child spells, which publish values of the same name independently.
        name: Source value name, e.g. BaseDamage, ChampionHeal, CooldownTime.
        kind: Shape of the values array, one of per_rank, by_level, scalar,
            ratio.
        values: Numeric values in source order.
        scaling_stat: Champion stat this value scales with, one of ap, ad,
            armor, magic_resist, attack_speed, crit, health; nullable when the
            value does not scale or the source enum is undecoded.
        stat_formula: Which amount of the scaling stat this value applies to,
            one of total, bonus; nullable when the value does not scale or the
            source enum is undecoded.
        damage_type: Damage type this value applies to, one of magic, physical,
            true; nullable when the source declares no damage type.
        display_as_percent: Source hint that the value is displayed as a
            percentage.
        source: Origin API for this value, one of ddragon, cdragon.
    """

    __tablename__ = "ability_values"
    __table_args__ = (
        UniqueConstraint(
            "ability_id",
            "spell_key",
            "name",
            name="uq_ability_values_ability_id_spell_key_name",
        ),
        CheckConstraint(
            "kind IN ('per_rank','by_level','scalar','ratio')", name="ck_ability_values_kind"
        ),
        CheckConstraint(
            "scaling_stat IN ('ap','ad','armor','magic_resist','attack_speed','crit','health')",
            name="ck_ability_values_scaling_stat",
        ),
        CheckConstraint("stat_formula IN ('total','bonus')", name="ck_ability_values_stat_formula"),
        CheckConstraint(
            "damage_type IN ('magic','physical','true')", name="ck_ability_values_damage_type"
        ),
        CheckConstraint("source IN ('ddragon','cdragon')", name="ck_ability_values_source"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    ability_id: Mapped[int] = mapped_column(
        ForeignKey("abilities.id", ondelete="CASCADE"), nullable=False
    )
    spell_key: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(16))
    values: Mapped[list[float]] = mapped_column(ARRAY(Float))
    scaling_stat: Mapped[str | None] = mapped_column(String(16), nullable=True)
    stat_formula: Mapped[str | None] = mapped_column(String(16), nullable=True)
    damage_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    display_as_percent: Mapped[bool] = mapped_column(default=False)
    source: Mapped[str] = mapped_column(String(16))

    ability: Mapped["Ability"] = relationship(back_populates="values")


class ItemValue(Base):
    """A named numeric value published by the source for one item.

    The kind field records the shape of the values array: 'per_rank' holds one
    entry per learnable rank, 'by_level' holds exactly two entries that are the
    level-1 and level-18 endpoints of a linear interpolation, 'scalar' holds a
    single value, and 'ratio' holds a single scaling coefficient.

    Args:
        id: Surrogate primary key.
        item_id: Foreign key to the owning item, cascades on delete.
        name: Source value name, e.g. Armor, HealthRegen, Cooldown.
        kind: Shape of the values array, one of per_rank, by_level, scalar,
            ratio.
        values: Numeric values in source order.
        display_as_percent: Source hint that the value is displayed as a
            percentage.
        source: Origin API for this value, one of ddragon, cdragon.
    """

    __tablename__ = "item_values"
    __table_args__ = (
        UniqueConstraint("item_id", "name", name="uq_item_values_item_id_name"),
        CheckConstraint(
            "kind IN ('per_rank','by_level','scalar','ratio')", name="ck_item_values_kind"
        ),
        CheckConstraint("source IN ('ddragon','cdragon')", name="ck_item_values_source"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    item_id: Mapped[str] = mapped_column(
        ForeignKey("items.ddragon_id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(16))
    values: Mapped[list[float]] = mapped_column(ARRAY(Float))
    display_as_percent: Mapped[bool] = mapped_column(default=False)
    source: Mapped[str] = mapped_column(String(16))

    item: Mapped["Item"] = relationship(back_populates="values")


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
        short_desc_text: Markup-stripped form of short_desc, used for
            embedding.
        long_desc: Long rune description.
        long_desc_text: Markup-stripped form of long_desc, used for
            embedding.
        row_index: Row within the path, 0-3; row 0 holds the keystones.
        position_index: Order of the rune within its row.
    """

    __tablename__ = "runes"
    __table_args__ = (UniqueConstraint("path_id", "row_index", "position_index"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    path_id: Mapped[int] = mapped_column(ForeignKey("rune_paths.id", ondelete="CASCADE"))
    key: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(128))
    short_desc: Mapped[str] = mapped_column(Text)
    short_desc_text: Mapped[str] = mapped_column(Text)
    long_desc: Mapped[str] = mapped_column(Text)
    long_desc_text: Mapped[str] = mapped_column(Text)
    row_index: Mapped[int] = mapped_column()
    position_index: Mapped[int] = mapped_column()

    path: Mapped["RunePath"] = relationship(back_populates="runes")


class SummonerSpell(Base):
    """A summoner spell, e.g. Flash or Ignite.

    Args:
        id: Data Dragon summoner spell id, primary key.
        key: Summoner spell key from Data Dragon.
        name: Summoner spell display name.
        description: Summoner spell description.
        description_text: Markup-stripped form of description, used for
            embedding.
        cooldown: Cooldown in seconds as a float, nullable.
        summoner_level: Minimum summoner level required, nullable.
        modes: Raw game-mode enum strings the source lists the spell under, in
            ascending order. Stored as an array rather than an association
            table because a mode is a bare source enum with no entity of its
            own and nothing else in the schema refers to one.
    """

    __tablename__ = "summoner_spells"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    key: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text)
    description_text: Mapped[str] = mapped_column(Text)
    cooldown: Mapped[float | None] = mapped_column(Float, nullable=True)
    summoner_level: Mapped[int | None] = mapped_column(nullable=True)
    modes: Mapped[list[str]] = mapped_column(ARRAY(String(64)), default=list)


# ---------- retrieval ----------


class Document(Base):
    """A retrievable unit of source content, indexed for RAG.

    Exactly one of the seven entity foreign keys is non-null; it identifies
    which entity this document was generated from.

    Args:
        id: Surrogate primary key.
        doc_key: Unique deterministic document identifier.
        collection: Logical collection this document belongs to, one of
            'abilities', 'champion_stats', 'equipment', 'lore'.
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
            "collection IN ('abilities','champion_stats','equipment','lore')",
            name="ck_documents_collection",
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
    doc_key: Mapped[str] = mapped_column(String(160), unique=True)
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
        String(64), ForeignKey("summoner_spells.id", ondelete="CASCADE"), nullable=True
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
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index"),
        Index(
            "ix_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column()
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))

    document: Mapped["Document"] = relationship(back_populates="chunks")
