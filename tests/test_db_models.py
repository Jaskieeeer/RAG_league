from collections.abc import Iterator

import pytest
from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, UniqueConstraint, create_engine, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from lolrag.config import get_settings
from lolrag.db.models import (
    Ability,
    AbilityValue,
    Base,
    Champion,
    Faction,
    Item,
    ItemValue,
    Rune,
    RunePath,
    Story,
    SummonerSpell,
    champion_related,
    item_components,
)

_EXPECTED_TABLES = {
    "factions",
    "roles",
    "champions",
    "abilities",
    "stories",
    "items",
    "rune_paths",
    "runes",
    "summoner_spells",
    "documents",
    "chunks",
    "champion_role",
    "champion_related",
    "story_champion",
    "item_tag",
    "item_components",
    "ability_values",
    "item_values",
}

_DOCUMENT_ENTITY_FK_COLUMNS = (
    "champion_slug",
    "story_slug",
    "faction_slug",
    "ability_id",
    "item_id",
    "rune_id",
    "summoner_spell_id",
)


def test_metadata_contains_all_expected_tables() -> None:
    """Base.metadata registers every model and association table."""
    assert _EXPECTED_TABLES <= set(Base.metadata.tables)


def test_documents_entity_foreign_keys_are_nullable() -> None:
    """Every entity foreign key column on documents is nullable."""
    documents = Base.metadata.tables["documents"]
    for column_name in _DOCUMENT_ENTITY_FK_COLUMNS:
        assert documents.columns[column_name].nullable is True


def test_documents_exactly_one_entity_check_constraint_exists() -> None:
    """The ck_documents_exactly_one_entity CheckConstraint is registered on documents."""
    documents = Base.metadata.tables["documents"]
    check_names = {
        constraint.name
        for constraint in documents.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert "ck_documents_exactly_one_entity" in check_names


def test_documents_collection_check_constraint_exists() -> None:
    """The ck_documents_collection CheckConstraint is registered on documents."""
    documents = Base.metadata.tables["documents"]
    check_names = {
        constraint.name
        for constraint in documents.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert "ck_documents_collection" in check_names


def test_abilities_slot_check_constraint_exists() -> None:
    """The ck_abilities_slot CheckConstraint is registered on abilities."""
    abilities = Base.metadata.tables["abilities"]
    check_names = {
        constraint.name
        for constraint in abilities.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert "ck_abilities_slot" in check_names


def test_abilities_champion_slot_unique_constraint_exists() -> None:
    """The champion_slug+slot UniqueConstraint is registered on abilities."""
    abilities = Base.metadata.tables["abilities"]
    unique_column_sets = {
        frozenset(column.name for column in constraint.columns)
        for constraint in abilities.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    assert frozenset({"champion_slug", "slot"}) in unique_column_sets


def test_chunks_embedding_is_384_dimensional_vector() -> None:
    """chunks.embedding is a pgvector Vector column with dimension 384."""
    chunks = Base.metadata.tables["chunks"]
    embedding_type = chunks.columns["embedding"].type
    assert isinstance(embedding_type, Vector)
    assert embedding_type.dim == 384


def test_champion_related_self_referential_relationship_is_configured() -> None:
    """Champion.related is a self-referential relationship through champion_related."""
    mapper = inspect(Champion)
    relationship_property = mapper.relationships["related"]
    assert relationship_property.mapper.class_ is Champion
    assert relationship_property.secondary is champion_related


@pytest.fixture
def db_session() -> Iterator[Session]:
    """Yield a Session bound to a transaction that is rolled back after the test.

    Returns:
        A Session against the configured database; every change made through
        it is discarded when the test finishes, so the database stays empty.
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


def test_item_components_and_builds_into_are_populated_in_both_directions(
    db_session: Session,
) -> None:
    """Component rows inserted through Core are visible from both ends of the relationship."""
    parent = Item(
        ddragon_id="9001",
        name="Test Parent Item",
        description="Parent description",
        description_text="Parent description",
        gold_total=3000,
        gold_base=1000,
    )
    component_one = Item(
        ddragon_id="9002",
        name="Test Component One",
        description="Component one description",
        description_text="Component one description",
        gold_total=1000,
        gold_base=1000,
    )
    component_two = Item(
        ddragon_id="9003",
        name="Test Component Two",
        description="Component two description",
        description_text="Component two description",
        gold_total=1000,
        gold_base=1000,
    )
    db_session.add_all([parent, component_one, component_two])
    db_session.flush()
    db_session.execute(
        item_components.insert(),
        [
            {"item_id": "9001", "component_id": "9002"},
            {"item_id": "9001", "component_id": "9003"},
        ],
    )
    db_session.flush()
    db_session.expire_all()

    parent = db_session.get(Item, "9001")
    component_one = db_session.get(Item, "9002")
    component_two = db_session.get(Item, "9003")
    assert parent is not None
    assert component_one is not None
    assert component_two is not None
    assert set(parent.components) == {component_one, component_two}
    assert parent in component_one.builds_into
    assert parent in component_two.builds_into


def test_item_component_quantity_defaults_to_one(db_session: Session) -> None:
    """A link inserted without naming quantity gets 1 from the server default."""
    _make_item(db_session, "9201")
    _make_item(db_session, "9202")
    db_session.execute(item_components.insert().values(item_id="9201", component_id="9202"))
    db_session.flush()

    quantity = db_session.execute(
        select(item_components.c.quantity).where(item_components.c.item_id == "9201")
    ).scalar_one()
    assert quantity == 1


def test_item_component_quantity_records_a_doubled_component(db_session: Session) -> None:
    """One association row can record a recipe consuming two copies of a component."""
    _make_item(db_session, "9203")
    _make_item(db_session, "9204")
    db_session.execute(
        item_components.insert().values(item_id="9203", component_id="9204", quantity=2)
    )
    db_session.flush()
    db_session.expire_all()

    parent = db_session.get(Item, "9203")
    assert parent is not None
    assert [component.ddragon_id for component in parent.components] == ["9204"]
    quantity = db_session.execute(
        select(item_components.c.quantity).where(item_components.c.item_id == "9203")
    ).scalar_one()
    assert quantity == 2


def test_rune_row_index_and_position_index_persist(db_session: Session) -> None:
    """A rune stores its row_index and position_index within the path."""
    path = RunePath(key="Domination", name="Domination")
    rune = Rune(
        path=path,
        key="Electrocute",
        name="Electrocute",
        short_desc="short",
        short_desc_text="short",
        long_desc="long",
        long_desc_text="long",
        row_index=0,
        position_index=0,
    )
    db_session.add_all([path, rune])
    db_session.flush()

    assert rune.row_index == 0
    assert rune.position_index == 0


def test_rune_duplicate_path_row_position_violates_unique_constraint(db_session: Session) -> None:
    """Two runes sharing (path_id, row_index, position_index) raise IntegrityError."""
    path = RunePath(key="Domination", name="Domination")
    first = Rune(
        path=path,
        key="Electrocute",
        name="Electrocute",
        short_desc="short",
        short_desc_text="short",
        long_desc="long",
        long_desc_text="long",
        row_index=0,
        position_index=0,
    )
    second = Rune(
        path=path,
        key="DarkHarvest",
        name="Dark Harvest",
        short_desc="short",
        short_desc_text="short",
        long_desc="long",
        long_desc_text="long",
        row_index=0,
        position_index=0,
    )
    db_session.add_all([path, first, second])

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_champion_lore_only_persists_with_no_ddragon_key(db_session: Session) -> None:
    """A non-playable champion with ddragon_key=None persists successfully."""
    faction = Faction(slug="test-faction-lore-only", name="Test Faction")
    champion = Champion(
        slug="test-lore-only-champion",
        ddragon_key=None,
        name="Norra",
        title="Test Title",
        faction=faction,
        bio_full="Full bio",
        bio_full_text="Full bio",
        playable=False,
    )
    db_session.add_all([faction, champion])
    db_session.flush()

    assert champion.ddragon_key is None
    assert champion.playable is False


def test_champion_playable_with_ddragon_key_persists(db_session: Session) -> None:
    """A playable champion with a ddragon_key still persists successfully."""
    faction = Faction(slug="test-faction-playable", name="Test Faction")
    champion = Champion(
        slug="test-playable-champion",
        ddragon_key="TestPlayableChampion",
        name="Test Champion",
        title="Test Title",
        faction=faction,
        bio_full="Full bio",
        bio_full_text="Full bio",
        playable=True,
    )
    db_session.add_all([faction, champion])
    db_session.flush()

    assert champion.ddragon_key == "TestPlayableChampion"
    assert champion.playable is True


def _make_ability(db_session: Session, slug: str, max_rank: int | None = 5) -> Ability:
    """Persist a champion with one ability and return that ability.

    Args:
        db_session: Session the new rows are added to.
        slug: Suffix making the faction, champion and ability rows unique.
        max_rank: Value stored on the ability's max_rank column.

    Returns:
        The flushed Ability, with its owning champion and faction persisted.
    """
    faction = Faction(slug=f"test-faction-{slug}", name="Test Faction")
    champion = Champion(
        slug=f"test-champion-{slug}",
        ddragon_key=f"TestChampion{slug}",
        name="Test Champion",
        title="Test Title",
        faction=faction,
        bio_full="Full bio",
        bio_full_text="Full bio",
        playable=True,
    )
    ability = Ability(
        champion=champion,
        slot="Q",
        name="Test Ability",
        description="Test description",
        max_rank=max_rank,
    )
    db_session.add_all([faction, champion, ability])
    db_session.flush()
    return ability


def _make_item(db_session: Session, ddragon_id: str) -> Item:
    """Persist a single item and return it.

    Args:
        db_session: Session the new row is added to.
        ddragon_id: Primary key for the new item.

    Returns:
        The flushed Item.
    """
    item = Item(
        ddragon_id=ddragon_id,
        name="Test Item",
        description="Test description",
        description_text="Test description",
        gold_total=3000,
        gold_base=1000,
    )
    db_session.add(item)
    db_session.flush()
    return item


def test_ability_value_per_rank_round_trips(db_session: Session) -> None:
    """A per_rank AbilityValue returns its five-element float array unchanged."""
    ability = _make_ability(db_session, "per-rank")
    value = AbilityValue(
        ability=ability,
        spell_key="TestQ",
        name="BaseDamage",
        kind="per_rank",
        values=[70.0, 105.0, 140.0, 175.0, 210.0],
        source="cdragon",
    )
    db_session.add(value)
    db_session.flush()
    db_session.expire_all()

    stored = db_session.execute(
        select(AbilityValue).where(AbilityValue.ability_id == ability.id)
    ).scalar_one()
    assert stored.values == [70.0, 105.0, 140.0, 175.0, 210.0]
    assert stored.kind == "per_rank"
    assert stored.display_as_percent is False


def test_ability_value_by_level_round_trips(db_session: Session) -> None:
    """A by_level AbilityValue returns its two interpolation endpoints unchanged."""
    ability = _make_ability(db_session, "by-level", max_rank=None)
    value = AbilityValue(
        ability=ability,
        spell_key="TestPassive",
        name="ChampionHeal",
        kind="by_level",
        values=[20.0, 240.0],
        display_as_percent=True,
        source="cdragon",
    )
    db_session.add(value)
    db_session.flush()
    db_session.expire_all()

    stored = db_session.execute(
        select(AbilityValue).where(AbilityValue.ability_id == ability.id)
    ).scalar_one()
    assert stored.values == [20.0, 240.0]
    assert stored.display_as_percent is True


def test_ability_value_accepts_null_and_valid_damage_type(db_session: Session) -> None:
    """AbilityValue accepts both damage_type=None and damage_type='magic'."""
    ability = _make_ability(db_session, "damage-type")
    untyped = AbilityValue(
        ability=ability,
        spell_key="TestQ",
        name="Cooldown",
        kind="per_rank",
        values=[12.0, 11.0, 10.0, 9.0, 8.0],
        damage_type=None,
        source="ddragon",
    )
    typed = AbilityValue(
        ability=ability,
        spell_key="TestQ",
        name="BaseDamage",
        kind="per_rank",
        values=[70.0, 105.0, 140.0, 175.0, 210.0],
        damage_type="magic",
        source="cdragon",
    )
    db_session.add_all([untyped, typed])
    db_session.flush()

    assert untyped.damage_type is None
    assert typed.damage_type == "magic"


def test_ability_value_invalid_kind_violates_check_constraint(db_session: Session) -> None:
    """An AbilityValue with an unknown kind raises IntegrityError."""
    ability = _make_ability(db_session, "bad-kind")
    db_session.add(
        AbilityValue(
            ability=ability,
            spell_key="TestQ",
            name="BaseDamage",
            kind="nonsense",
            values=[10.0],
            source="cdragon",
        )
    )

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_ability_value_invalid_damage_type_violates_check_constraint(db_session: Session) -> None:
    """An AbilityValue with an unknown damage_type raises IntegrityError."""
    ability = _make_ability(db_session, "bad-damage-type")
    db_session.add(
        AbilityValue(
            ability=ability,
            spell_key="TestQ",
            name="BaseDamage",
            kind="per_rank",
            values=[10.0],
            damage_type="ice",
            source="cdragon",
        )
    )

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_ability_value_duplicate_name_violates_unique_constraint(db_session: Session) -> None:
    """Two AbilityValue rows sharing (ability_id, spell_key, name) raise IntegrityError."""
    ability = _make_ability(db_session, "duplicate-name")
    db_session.add_all(
        [
            AbilityValue(
                ability=ability,
                spell_key="TestQ",
                name="BaseDamage",
                kind="per_rank",
                values=[10.0],
                source="cdragon",
            ),
            AbilityValue(
                ability=ability,
                spell_key="TestQ",
                name="BaseDamage",
                kind="scalar",
                values=[20.0],
                source="ddragon",
            ),
        ]
    )

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_deleting_ability_cascades_to_ability_values(db_session: Session) -> None:
    """Deleting an ability removes the ability_values rows that belong to it."""
    ability = _make_ability(db_session, "cascade")
    db_session.add(
        AbilityValue(
            ability=ability,
            spell_key="TestQ",
            name="BaseDamage",
            kind="per_rank",
            values=[10.0, 20.0, 30.0, 40.0, 50.0],
            source="cdragon",
        )
    )
    db_session.flush()
    ability_id = ability.id

    db_session.delete(ability)
    db_session.flush()

    remaining = (
        db_session.execute(select(AbilityValue).where(AbilityValue.ability_id == ability_id))
        .scalars()
        .all()
    )
    assert remaining == []


def test_item_value_round_trips(db_session: Session) -> None:
    """A scalar ItemValue returns its single-element float array unchanged."""
    item = _make_item(db_session, "9101")
    value = ItemValue(
        item=item,
        name="Armor",
        kind="scalar",
        values=[45.0],
        source="ddragon",
    )
    db_session.add(value)
    db_session.flush()
    db_session.expire_all()

    stored = db_session.execute(select(ItemValue).where(ItemValue.item_id == "9101")).scalar_one()
    assert stored.values == [45.0]
    assert stored.kind == "scalar"
    assert stored.display_as_percent is False


def test_item_value_duplicate_name_violates_unique_constraint(db_session: Session) -> None:
    """Two ItemValue rows sharing (item_id, name) raise IntegrityError."""
    item = _make_item(db_session, "9102")
    db_session.add_all(
        [
            ItemValue(item=item, name="Armor", kind="scalar", values=[45.0], source="ddragon"),
            ItemValue(item=item, name="Armor", kind="ratio", values=[0.35], source="cdragon"),
        ]
    )

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_item_value_invalid_kind_violates_check_constraint(db_session: Session) -> None:
    """An ItemValue with an unknown kind raises IntegrityError."""
    item = _make_item(db_session, "9103")
    db_session.add(
        ItemValue(item=item, name="Armor", kind="nonsense", values=[45.0], source="ddragon")
    )

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_deleting_item_cascades_to_item_values(db_session: Session) -> None:
    """Deleting an item removes the item_values rows that belong to it."""
    item = _make_item(db_session, "9104")
    db_session.add(
        ItemValue(item=item, name="Armor", kind="scalar", values=[45.0], source="ddragon")
    )
    db_session.flush()

    db_session.delete(item)
    db_session.flush()

    remaining = (
        db_session.execute(select(ItemValue).where(ItemValue.item_id == "9104")).scalars().all()
    )
    assert remaining == []


def test_ability_max_rank_accepts_null_and_integer(db_session: Session) -> None:
    """Ability.max_rank stores NULL for a passive and an integer for a spell."""
    passive = _make_ability(db_session, "passive", max_rank=None)
    spell = _make_ability(db_session, "spell", max_rank=5)

    assert passive.max_rank is None
    assert spell.max_rank == 5


def test_story_subsection_count_persists(db_session: Session) -> None:
    """A story stores the total number of content subsections."""
    story = Story(
        slug="test-story-subsection-count",
        title="Test Story",
        word_count=1200,
        subsection_count=7,
        content="Story content",
        content_text="Story content",
    )
    db_session.add(story)
    db_session.flush()
    db_session.expire_all()

    stored = db_session.get(Story, "test-story-subsection-count")
    assert stored is not None
    assert stored.subsection_count == 7


def test_summoner_spell_long_id_and_fractional_cooldown_round_trip(db_session: Session) -> None:
    """The longest Data Dragon spell id and a sub-second cooldown survive a round trip."""
    spell_id = "Summoner_UltBookSmitePlaceholder"
    spell = SummonerSpell(
        id=spell_id,
        key="55",
        name="Test Spell",
        description="Test description",
        description_text="Test description",
        cooldown=0.25,
        summoner_level=1,
    )
    db_session.add(spell)
    db_session.flush()
    db_session.expire_all()

    stored = db_session.get(SummonerSpell, spell_id)
    assert stored is not None
    assert len(spell_id) == 32
    assert stored.cooldown == 0.25
