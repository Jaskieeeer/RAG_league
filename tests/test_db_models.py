from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, UniqueConstraint, inspect

from lolrag.db.models import Base, Champion, champion_related

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
