import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from lolrag.config import get_settings
from lolrag.db.models import Ability, AbilityValue, Base, story_champion
from lolrag.fetch.client import FetchClient
from lolrag.ingest.identifiers import PASSIVE_SLOT
from lolrag.ingest.run import run_ingest

CACHE_DIR = Path(get_settings().cache_dir)

pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(not CACHE_DIR.is_dir(), reason=f"no warm corpus cache at {CACHE_DIR}"),
]

EXPECTED_COUNTS = {
    "factions": 14,
    "roles": 6,
    "champions": 174,
    "abilities": 865,
    "stories": 199,
    "items": 706,
    "rune_paths": 5,
    "runes": 62,
    "summoner_spells": 16,
    "ability_values": 7252,
    "item_values": 3667,
    "champion_role": 304,
    "champion_related": 556,
    "story_champion": 260,
    "item_tag": 2402,
    "item_components": 508,
    "documents": 0,
    "chunks": 0,
}

AATROX_Q_SPELL_KEY = "AatroxQ"

EXPECTED_SPELLS_RESOLVED = 560
EXPECTED_SPELLS_UNRESOLVED = 132
EXPECTED_PASSIVES = 173

EXPECTED_STAT_FORMULAS = {"total": 459, "bonus": 165, "none": 6628}


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


def row_counts(session: Session) -> dict[str, int]:
    """Count the rows of every mapped table.

    Args:
        session: Session the counts are read through.

    Returns:
        Mapping of table name to row count, covering every table in the schema
        rather than only the ones the loaders write, so a table left empty by
        mistake is as visible as one filled wrongly.
    """
    return {
        name: session.execute(select(func.count()).select_from(table)).scalar_one()
        for name, table in sorted(Base.metadata.tables.items())
    }


def corpus_hash(session: Session) -> str:
    """Hash the full ordered row projection of every mapped table.

    Args:
        session: Session the rows are read through.

    Returns:
        Hex digest over every column of every row of every table, tables in name
        order and rows in primary key order. Row counts alone cannot tell a true
        upsert from a delete-then-insert or from a merge that rewrites a column
        on every run; an equal digest across two runs can.
    """
    digest = hashlib.sha256()
    for name, table in sorted(Base.metadata.tables.items()):
        digest.update(name.encode())
        for row in session.execute(select(table).order_by(*table.primary_key.columns)):
            digest.update(repr(tuple(row)).encode())
    return digest.hexdigest()


def ability_values_by_name(
    session: Session, champion_slug: str, slot: str, spell_key: str
) -> dict[str, AbilityValue]:
    """Read one spell's stored values, keyed by value name.

    Args:
        session: Session the rows are read through.
        champion_slug: Universe slug of the owning champion.
        slot: Ability slot, one of P, Q, W, E, R.
        spell_key: Short source name of the spell that publishes the values.

    Returns:
        Mapping of value name to the stored AbilityValue row.
    """
    rows = session.execute(
        select(AbilityValue)
        .join(Ability, Ability.id == AbilityValue.ability_id)
        .where(
            Ability.champion_slug == champion_slug,
            Ability.slot == slot,
            AbilityValue.spell_key == spell_key,
        )
    ).scalars()
    return {row.name: row for row in rows}


def tooltip_counts(session: Session) -> dict[str, int]:
    """Count the stored abilities by slot kind and tooltip resolution.

    Args:
        session: Session the counts are read through.

    Returns:
        Mapping with the number of spell rows carrying a resolved tooltip, the
        number of spell rows whose tooltip is NULL, the number of passive rows
        and the number of passive rows carrying a resolved tooltip.
    """

    def count(*conditions: Any) -> int:
        return session.execute(
            select(func.count()).select_from(Ability).where(*conditions)
        ).scalar_one()

    return {
        "spells_resolved": count(
            Ability.slot != PASSIVE_SLOT, Ability.tooltip_resolved.is_not(None)
        ),
        "spells_unresolved": count(
            Ability.slot != PASSIVE_SLOT, Ability.tooltip_resolved.is_(None)
        ),
        "passives": count(Ability.slot == PASSIVE_SLOT),
        "passives_resolved": count(
            Ability.slot == PASSIVE_SLOT, Ability.tooltip_resolved.is_not(None)
        ),
    }


def stat_formula_counts(session: Session) -> dict[str, int]:
    """Count the stored ability values by the stat formula they resolved to.

    Args:
        session: Session the counts are read through.

    Returns:
        Mapping with the number of rows applying to the total amount of their
        scaling stat, the number applying to the bonus amount, and the number
        whose formula is NULL, which are every value no stat-scaling part
        encloses plus the few whose calculations disagree or whose source enum
        has no proven meaning.
    """

    def count(*conditions: Any) -> int:
        return session.execute(
            select(func.count()).select_from(AbilityValue).where(*conditions)
        ).scalar_one()

    return {
        "total": count(AbilityValue.stat_formula == "total"),
        "bonus": count(AbilityValue.stat_formula == "bonus"),
        "none": count(AbilityValue.stat_formula.is_(None)),
    }


async def test_run_ingest_resolves_the_measured_share_of_tooltips(db_session: Session) -> None:
    """The corpus splits into resolved spells, blocked spells and passives that publish none."""
    settings = get_settings()
    async with FetchClient(settings) as client:
        await run_ingest(db_session, client, settings)

    assert tooltip_counts(db_session) == {
        "spells_resolved": EXPECTED_SPELLS_RESOLVED,
        "spells_unresolved": EXPECTED_SPELLS_UNRESOLVED,
        "passives": EXPECTED_PASSIVES,
        "passives_resolved": 0,
    }


async def test_run_ingest_loads_the_full_corpus_and_repeats_it_identically(
    db_session: Session,
) -> None:
    """Two runs over the warm cache land the same rows, with the same values, twice."""
    settings = get_settings()
    async with FetchClient(settings) as client:
        await run_ingest(db_session, client, settings)
        first_counts = row_counts(db_session)
        first_hash = corpus_hash(db_session)

        await run_ingest(db_session, client, settings)
        second_counts = row_counts(db_session)
        second_hash = corpus_hash(db_session)

    assert first_counts == EXPECTED_COUNTS
    assert second_counts == EXPECTED_COUNTS
    assert second_hash == first_hash
    assert stat_formula_counts(db_session) == EXPECTED_STAT_FORMULAS

    values = ability_values_by_name(db_session, "aatrox", "Q", AATROX_Q_SPELL_KEY)

    base_damage = values["QBaseDamage"]
    assert base_damage.kind == "per_rank"
    assert base_damage.values == [10, 25, 40, 55, 70]
    assert base_damage.damage_type == "physical"

    ad_ratio = values["QTotalADRatio"]
    assert ad_ratio.kind == "per_rank"
    assert ad_ratio.values == pytest.approx([0.6, 0.675, 0.75, 0.825, 0.9])
    assert ad_ratio.scaling_stat == "ad"
    assert ad_ratio.damage_type == "physical"

    cooldown = values["CooldownTime"]
    assert cooldown.kind == "per_rank"
    assert cooldown.values == [14, 12, 10, 8, 6]

    assert (
        db_session.execute(
            select(func.count())
            .select_from(story_champion)
            .where(
                story_champion.c.story_slug == "hollowspun",
                story_champion.c.champion_slug == "taliyah",
            )
        ).scalar_one()
        == 1
    )
