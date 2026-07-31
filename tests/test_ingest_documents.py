from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from lolrag.config import get_settings
from lolrag.db.models import (
    Ability,
    AbilityValue,
    Champion,
    Chunk,
    Document,
    Faction,
    Item,
    ItemValue,
    Rune,
    RunePath,
    Story,
    SummonerSpell,
)
from lolrag.ingest.documents import (
    CHUNK_SIZE,
    MAP_NAMES,
    build_ability_document,
    build_champion_document,
    build_faction_document,
    build_item_document,
    build_rune_document,
    build_splitter,
    build_story_document,
    build_summoner_spell_document,
    chunk_document,
    load_documents,
    render_value,
)

DOC_KEY_WIDTH = 160


# ---------- value rendering tests ----------


def test_render_value_joins_per_rank_values_and_names_the_scaling_stat() -> None:
    """A per-rank coefficient renders every rank and says which stat it applies to."""
    line = render_value(
        "QTotalADRatio",
        "per_rank",
        [0.6, 0.675, 0.75, 0.825, 0.9],
        scaling_stat="ad",
        stat_formula="total",
        damage_type="physical",
    )

    assert line == "QTotalADRatio: 0.6/0.675/0.75/0.825/0.9 of total AD, physical damage"


def test_render_value_shows_a_percent_value_as_a_percentage() -> None:
    """A stored 0.3 flagged as a percentage reads as 30%, never as 0.3."""
    assert render_value("SlowAmount", "scalar", [0.3], display_as_percent=True) == "SlowAmount: 30%"


def test_render_value_rounds_the_float32_noise_out_of_a_percentage() -> None:
    """The stored 0.010999999940395355 is the source's float32 noise, not a fact about the game."""
    line = render_value(
        "EVampHPRatio", "per_rank", [0.010999999940395355] * 2, display_as_percent=True
    )

    assert line == "EVampHPRatio: 1.1%/1.1%"


def test_render_value_states_a_by_level_row_as_a_level_range() -> None:
    """The two entries are interpolation endpoints, so they are rendered as the range they are."""
    line = render_value("MonsterDamageCap", "by_level", [100.0, 320.0])

    assert line == "MonsterDamageCap: 100 at level 1 to 320 at level 18"


def test_render_value_renders_a_scalar_as_one_number() -> None:
    """A single-entry value carries no separator."""
    assert render_value("Cooldown", "scalar", [12.0]) == "Cooldown: 12"


def test_render_value_names_a_stat_with_no_declared_formula() -> None:
    """A scaling stat whose formula the source left undeclared is still named."""
    line = render_value("Ratio", "scalar", [0.5], scaling_stat="ap")

    assert line == "Ratio: 0.5 of AP"


def test_render_value_emits_nothing_for_a_stat_formula_with_no_scaling_stat() -> None:
    """The eleven rows carrying a formula but no stat must not render a dangling "bonus"."""
    line = render_value("Orphan", "scalar", [0.5], stat_formula="bonus")

    assert line == "Orphan: 0.5"
    assert "bonus" not in line


def test_render_value_omits_the_damage_type_when_the_source_declares_none() -> None:
    """A value with no damage type gets no trailing clause."""
    assert render_value("CooldownTime", "per_rank", [14.0, 12.0]) == "CooldownTime: 14/12"


# ---------- document builder tests ----------


def champion(slug: str = "aatrox") -> Champion:
    """Build a transient champion carrying the fields the builders read.

    Args:
        slug: Universe slug the champion is keyed on.

    Returns:
        A Champion with a faction attached and no roles.
    """
    return Champion(
        slug=slug,
        name="Aatrox",
        title="the Darkin Blade",
        faction_slug="noxus",
        faction=Faction(slug="noxus", name="Noxus"),
        bio_full="Full bio.",
        bio_full_text="Full bio.",
        bio_short="Short bio.",
        bio_short_text="Short bio.",
        playable=True,
    )


def ability() -> Ability:
    """Build a transient ability with one value row and a resolved tooltip.

    Returns:
        An Ability whose champion and values are attached in memory.
    """
    return Ability(
        id=7,
        champion_slug="aatrox",
        champion=champion(),
        slot="Q",
        name="The Darkin Blade",
        description="Aatrox slams his greatsword down.",
        tooltip_resolved="Deals 10/25/40/55/70 physical damage.",
        max_rank=5,
        values=[
            AbilityValue(
                spell_key="AatroxQ",
                name="QBaseDamage",
                kind="per_rank",
                values=[10.0, 25.0, 40.0, 55.0, 70.0],
                damage_type="physical",
                display_as_percent=False,
                source="cdragon",
            )
        ],
    )


def item() -> Item:
    """Build a transient item with one value row.

    Returns:
        An Item whose values are attached in memory.
    """
    return Item(
        ddragon_id="3153",
        name="Blade of the Ruined King",
        description="<mainText>Bork</mainText>",
        description_text="Deals damage on hit.",
        gold_total=3200,
        gold_base=725,
        purchasable=True,
        in_store=True,
        values=[
            ItemValue(
                name="AttackDamage",
                kind="scalar",
                values=[40.0],
                display_as_percent=False,
                source="ddragon",
            )
        ],
    )


def test_build_ability_document_opens_with_its_title_line() -> None:
    """The first line of the content is the title, which chunking reuses as its header."""
    document = build_ability_document(ability())

    assert document.content.startswith("Aatrox Q: The Darkin Blade\n")
    assert document.title == "Aatrox Q: The Darkin Blade"
    assert document.doc_key == "ability:aatrox:Q"
    assert document.collection == "abilities"
    assert document.entity_column == "ability_id"


def test_build_ability_document_cleans_the_markup_out_of_the_description() -> None:
    """Abilities publish no stripped description column, so the builder strips it."""
    row = ability()
    row.description = "Heals him.<br>Then again."

    assert "Heals him.\nThen again." in build_ability_document(row).content
    assert "<br>" not in build_ability_document(row).content


def test_build_ability_document_omits_the_tooltip_block_when_it_is_null() -> None:
    """A blocked or absent tooltip leaves no empty block behind."""
    row = ability()
    row.tooltip_resolved = None
    content = build_ability_document(row).content

    assert "\n\n\n" not in content
    assert "physical damage." not in content.split("Values:")[0]


def test_build_ability_document_renders_its_values_block() -> None:
    """The values block is headed once and carries one line per stored row."""
    content = build_ability_document(ability()).content

    assert content.endswith("Values:\nQBaseDamage: 10/25/40/55/70, physical damage")


def test_build_item_document_names_every_mode_it_is_sold_on() -> None:
    """The title lists the game modes by name, in map id order."""
    document = build_item_document(item(), ["Damage"], [11, 12])

    assert document.title == "Blade of the Ruined King (Summoner's Rift, Howling Abyss)"
    assert document.doc_key == "item:3153"
    assert document.collection == "equipment"
    assert "Cost: 3200 gold (725 to combine)" in document.content
    assert "Tags: Damage" in document.content


def test_build_item_document_rejects_a_map_id_the_constant_does_not_name() -> None:
    """A new game mode must fail loudly rather than vanish from the title."""
    with pytest.raises(KeyError):
        build_item_document(item(), [], [99])


def test_build_rune_document_calls_a_row_zero_rune_a_keystone() -> None:
    """Row zero of every path holds the keystones, which the title must say."""
    path = RunePath(id=8100, key="Domination", name="Domination")
    rune = Rune(
        id=8112,
        path=path,
        key="Electrocute",
        name="Electrocute",
        short_desc="Short.",
        short_desc_text="Short.",
        long_desc="Long.",
        long_desc_text="Long.",
        row_index=0,
        position_index=0,
    )

    document = build_rune_document(rune)

    assert document.title == "Electrocute (Domination keystone)"
    assert document.doc_key == "rune:8112"


def test_build_rune_document_drops_a_long_description_that_repeats_the_short_one() -> None:
    """An identical long description adds no fact and would only crowd the chunk."""
    path = RunePath(id=8100, key="Domination", name="Domination")
    rune = Rune(
        id=8126,
        path=path,
        key="CheapShot",
        name="Cheap Shot",
        short_desc="Same text.",
        short_desc_text="Same text.",
        long_desc="Same text.",
        long_desc_text="Same text.",
        row_index=1,
        position_index=0,
    )

    assert build_rune_document(rune).content.count("Same text.") == 1


def test_build_summoner_spell_document_states_cooldown_and_unlock_level() -> None:
    """Both facts the source publishes reach the document."""
    spell = SummonerSpell(
        id="SummonerFlash",
        key="4",
        name="Flash",
        description="<br>Teleports.",
        description_text="Teleports.",
        cooldown=300.0,
        summoner_level=7,
    )

    document = build_summoner_spell_document(spell)

    assert document.title == "Flash (summoner spell)"
    assert document.doc_key == "summoner_spell:SummonerFlash"
    assert "Cooldown: 300 seconds" in document.content
    assert "Unlocked at summoner level 7" in document.content


def test_build_champion_document_leads_with_name_and_title() -> None:
    """The champion document opens with the line that names it."""
    document = build_champion_document(champion())

    assert document.title == "Aatrox, the Darkin Blade"
    assert document.doc_key == "champion:aatrox"
    assert document.collection == "lore"
    assert "Faction: Noxus" in document.content
    assert document.content.endswith("Full bio.")


def test_build_story_document_keeps_the_scene_boundaries() -> None:
    """The triple newlines the story loader wrote survive into the document."""
    story = Story(
        slug="the-darkin-blade",
        title="The Darkin Blade",
        author=None,
        word_count=4,
        subsection_count=2,
        content="Scene one.\n\nScene two.",
        content_text="Scene one.\n\n\nScene two.",
    )

    document = build_story_document(story)

    assert document.doc_key == "story:the-darkin-blade"
    assert "Scene one.\n\n\nScene two." in document.content


def test_build_faction_document_survives_a_faction_with_no_overview() -> None:
    """The synthetic unaffiliated faction publishes nothing and still gets a document."""
    document = build_faction_document(Faction(slug="unaffiliated", name="Unaffiliated"))

    assert document.doc_key == "faction:unaffiliated"
    assert document.content == "Faction: Unaffiliated"


def test_every_builder_produces_a_doc_key_inside_the_column_width() -> None:
    """A doc_key that overflows its column would fail the insert at ingest time."""
    documents = [
        build_ability_document(ability()),
        build_item_document(item(), [], [11]),
        build_champion_document(champion()),
        build_faction_document(Faction(slug="unaffiliated", name="Unaffiliated")),
    ]

    assert all(len(document.doc_key) <= DOC_KEY_WIDTH for document in documents)
    assert Document.__table__.columns["doc_key"].type.length == DOC_KEY_WIDTH


def test_map_names_covers_every_mode_the_title_can_need() -> None:
    """The constant is the only source of mode names, so it must be complete and correct."""
    assert MAP_NAMES[12] == "Howling Abyss"
    assert MAP_NAMES[33] == "Swarm"
    assert MAP_NAMES[35] == "The Bandlewood"


# ---------- chunking tests ----------


def test_chunk_document_leaves_a_short_document_whole_and_unprefixed() -> None:
    """A document that fits in one chunk already opens with its title."""
    content = "Aatrox Q: The Darkin Blade\n\nShort body."

    chunks = chunk_document(content, build_splitter())

    assert chunks == [content]


def test_chunk_document_prefixes_the_title_onto_every_later_chunk() -> None:
    """A tail chunk of bare numbers must still name what it describes."""
    title = "Aatrox Q: The Darkin Blade"
    body = "\n\n".join(f"Paragraph {index} " + "word " * 60 for index in range(6))
    content = f"{title}\n\n{body}"

    chunks = chunk_document(content, build_splitter())

    assert len(chunks) > 1
    assert all(chunk.startswith(title) for chunk in chunks)
    assert chunks[0].count(title) == 1


def test_chunk_document_splits_a_story_on_its_scene_boundary() -> None:
    """The triple newline outranks the paragraph break, so scenes stay whole."""
    scene_one = "One " * 150
    scene_two = "Two " * 150
    content = f"A Tale\n\n{scene_one.strip()}\n\n\n{scene_two.strip()}"

    chunks = chunk_document(content, build_splitter())

    assert len(chunks) > 1
    assert all(len(chunk) <= CHUNK_SIZE + len("A Tale") + 1 for chunk in chunks)
    assert not any("One" in chunk and "Two" in chunk for chunk in chunks)


# ---------- orchestrator tests ----------


@pytest.fixture
def db_session() -> Iterator[Session]:
    """Yield a Session bound to a transaction that is rolled back after the test.

    Returns:
        A Session against the configured database; every change made through it
        is discarded when the test finishes, so the database stays empty.
    """
    engine = create_engine(get_settings().database_url)
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    try:
        yield session
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()
        engine.dispose()


def seed_entities(session: Session) -> None:
    """Insert the two entity rows the orchestrator tests build documents from.

    Args:
        session: Open Session the rows are added to.

    Returns:
        None. Both entities are free of foreign keys, so no other table is
        needed to make the documents insertable.
    """
    session.add(
        Faction(
            slug="noxus", name="Noxus", overview="<p>An empire.</p>", overview_text="An empire."
        )
    )
    session.add(
        SummonerSpell(
            id="SummonerFlash",
            key="4",
            name="Flash",
            description="Teleports.",
            description_text="Teleports.",
            cooldown=300.0,
            summoner_level=7,
        )
    )
    session.flush()


def test_load_documents_writes_a_document_and_its_chunks(db_session: Session) -> None:
    """One run lands every seeded entity as a document with at least one chunk."""
    seed_entities(db_session)

    stats = load_documents(db_session, get_settings())

    assert stats.documents_built == 2
    assert stats.documents_changed == 2
    assert stats.chunks_written == 2
    assert stats.chunks_skipped == 0
    assert db_session.execute(select(func.count()).select_from(Document)).scalar_one() == 2
    assert db_session.execute(select(func.count()).select_from(Chunk)).scalar_one() == 2


def test_load_documents_run_twice_changes_nothing_and_embeds_nothing(db_session: Session) -> None:
    """The conditional upsert is the whole point: an unchanged corpus costs zero embeddings."""
    seed_entities(db_session)
    load_documents(db_session, get_settings())

    second = load_documents(db_session, get_settings())

    assert second.documents_built == 2
    assert second.documents_changed == 0
    assert second.chunks_written == 0
    assert second.chunks_skipped == 2


def test_load_documents_re_embeds_only_the_document_whose_content_changed(
    db_session: Session,
) -> None:
    """A single edited entity must not drag the rest of the corpus through the model again."""
    seed_entities(db_session)
    load_documents(db_session, get_settings())
    faction = db_session.get(Faction, "noxus")
    assert faction is not None
    faction.overview_text = "An empire that rewrote itself."
    db_session.flush()

    stats = load_documents(db_session, get_settings())

    assert stats.documents_changed == 1
    assert stats.chunks_written == 1
    assert stats.chunks_skipped == 1
