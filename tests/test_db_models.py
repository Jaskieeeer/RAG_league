from collections.abc import Iterator

import pytest
from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, UniqueConstraint, create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from lolrag.config import get_settings
from lolrag.db.models import Base, Champion, Faction, Item, Rune, RunePath, champion_related

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
    """Linking an item to its components populates builds_into on each component."""
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
    parent.components = [component_one, component_two]
    db_session.add_all([parent, component_one, component_two])
    db_session.flush()

    assert set(parent.components) == {component_one, component_two}
    assert parent in component_one.builds_into
    assert parent in component_two.builds_into


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
