from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from sqlalchemy import Table, create_engine, func, select
from sqlalchemy.orm import Session

from lolrag.config import get_settings
from lolrag.db.models import (
    Champion,
    Faction,
    Item,
    Role,
    Story,
    champion_related,
    champion_role,
    item_components,
    item_tag,
    story_champion,
)
from lolrag.ingest.associations import (
    AssociationStats,
    build_champion_related,
    build_champion_roles,
    build_item_components,
    build_item_tags,
    build_story_champions,
    load_associations,
)

ROSTER_SLUGS = ("aatrox", "renataglasc", "norra")
STORY_SLUGS = ("darkin-blade", "hollowspun", "chemtech")
ROLE_SLUGS = ("fighter", "tank", "support", "mage")
ITEM_IDS = ("1036", "3035", "1029")


# ---------- payload fixtures ----------


def ddragon_champion_list() -> dict[str, Any]:
    """Build a Data Dragon champion.json body carrying two tagged champions.

    Returns:
        Payload whose "data" key maps champion id to a summary with a "tags"
        list. Renata is present so the Universe slug override is exercised.
    """
    return {
        "data": {
            "Aatrox": {"id": "Aatrox", "key": "266", "tags": ["Fighter", "Tank"]},
            "Renata": {"id": "Renata", "key": "888", "tags": ["Support", "Mage"]},
        }
    }


def universe_champion(
    slug: str,
    *,
    related: Sequence[str] = (),
    stories: Sequence[tuple[str, Sequence[str]]] = (),
) -> dict[str, Any]:
    """Build a Universe champion payload carrying related champions and stories.

    Args:
        slug: Universe champion slug the payload describes.
        related: Slugs published under the top-level "related-champions" array.
        stories: (story slug, featured champion slugs) pairs, one per
            story-preview module the page carries.

    Returns:
        Payload shaped like a real Universe champion response. It also carries a
        featured-video module with its own "related-champions" array and no
        story slug, reproducing the decoy the builders must ignore.
    """
    modules: list[dict[str, Any]] = [
        {
            "type": "featured-video",
            "related-champions": [{"slug": "decoy"}],
        }
    ]
    modules.extend(
        {
            "type": "story-preview",
            "story-slug": story_slug,
            "featured-champions": [{"slug": featured} for featured in featured_slugs],
        }
        for story_slug, featured_slugs in stories
    )
    return {
        "champion": {"slug": slug, "name": slug.capitalize()},
        "related-champions": [{"slug": entry} for entry in related],
        "explore-champions": [{"slug": "decoy"}],
        "modules": modules,
    }


def universe_champions() -> dict[str, dict]:
    """Build the Universe roster of three champions.

    Returns:
        Mapping of slug to payload. Aatrox names Renata as related and Renata
        does not name him back, so the published relation is asymmetric. The
        chemtech story is owned by Renata's page yet features Aatrox, the
        featured-only edge that the owner edge alone would miss.
    """
    return {
        "aatrox": universe_champion(
            "aatrox",
            related=("renataglasc",),
            stories=(("darkin-blade", ()), ("hollowspun", ("norra",))),
        ),
        "renataglasc": universe_champion(
            "renataglasc",
            stories=(("chemtech", ("aatrox",)),),
        ),
        "norra": universe_champion("norra", related=("aatrox",)),
    }


def ddragon_items() -> dict[str, Any]:
    """Build a Data Dragon item.json body covering tags and a doubled component.

    Returns:
        Payload whose "data" key maps item id to a record; item 3035 lists
        component 1036 twice and item 1029 publishes an empty tag list.
    """
    return {
        "data": {
            "1036": {"name": "Long Sword", "tags": ["Damage"]},
            "3035": {
                "name": "Last Whisper",
                "from": ["1036", "1036"],
                "tags": ["Damage", "ArmorPenetration"],
            },
            "1029": {"name": "Cloth Armor", "tags": []},
        }
    }


def roster_slugs() -> set[str]:
    """Return the champion slugs the loader will persist.

    Returns:
        The three roster slugs as a set, the shape the builders take.
    """
    return set(ROSTER_SLUGS)


def item_ids() -> set[str]:
    """Return the item ids the loader will persist.

    Returns:
        The three fixture item ids as a set, the shape the builders take.
    """
    return set(ITEM_IDS)


# ---------- champion_role tests ----------


def test_build_champion_roles_keys_rows_on_the_universe_slug() -> None:
    """Roles hang off the Universe slug, so the Renata override is applied."""
    edges = build_champion_roles(ddragon_champion_list(), roster_slugs())

    assert edges.rows == [
        {"champion_slug": "aatrox", "role_slug": "fighter"},
        {"champion_slug": "aatrox", "role_slug": "tank"},
        {"champion_slug": "renataglasc", "role_slug": "support"},
        {"champion_slug": "renataglasc", "role_slug": "mage"},
    ]
    assert edges.dropped == 0


def test_build_champion_roles_deduplicates_a_repeated_tag() -> None:
    """A champion publishing one tag twice yields one row, as the primary key needs."""
    payload = ddragon_champion_list()
    payload["data"]["Aatrox"]["tags"] = ["Fighter", "Fighter"]

    edges = build_champion_roles(payload, roster_slugs())

    assert [row for row in edges.rows if row["champion_slug"] == "aatrox"] == [
        {"champion_slug": "aatrox", "role_slug": "fighter"}
    ]


def test_build_champion_roles_gives_a_lore_only_character_no_roles() -> None:
    """Norra has no Data Dragon entry, so no tags become roles and nothing is dropped."""
    edges = build_champion_roles(ddragon_champion_list(), roster_slugs())

    assert [row for row in edges.rows if row["champion_slug"] == "norra"] == []
    assert edges.dropped == 0


def test_build_champion_roles_drops_a_champion_outside_the_roster() -> None:
    """A Data Dragon champion with no Universe page contributes no role rows."""
    payload = ddragon_champion_list()
    payload["data"]["Ahri"] = {"id": "Ahri", "key": "103", "tags": ["Mage", "Assassin"]}

    edges = build_champion_roles(payload, roster_slugs())

    assert [row for row in edges.rows if row["champion_slug"] == "ahri"] == []
    assert edges.dropped == 2


# ---------- champion_related tests ----------


def test_build_champion_related_stores_the_published_direction_only() -> None:
    """Aatrox names Renata, Renata does not name him, and no reverse row is invented."""
    edges = build_champion_related(universe_champions(), roster_slugs())

    assert {"champion_slug": "aatrox", "related_slug": "renataglasc"} in edges.rows
    assert {"champion_slug": "renataglasc", "related_slug": "aatrox"} not in edges.rows


def test_build_champion_related_reads_only_the_top_level_array() -> None:
    """The featured-video module and the explore carousel are not relation sources."""
    edges = build_champion_related(universe_champions(), roster_slugs())

    assert edges.rows == [
        {"champion_slug": "aatrox", "related_slug": "renataglasc"},
        {"champion_slug": "norra", "related_slug": "aatrox"},
    ]


def test_build_champion_related_deduplicates_a_repeated_entry() -> None:
    """A champion named twice by one page yields one row."""
    payloads = universe_champions()
    payloads["aatrox"]["related-champions"] = [{"slug": "renataglasc"}, {"slug": "renataglasc"}]

    edges = build_champion_related(payloads, roster_slugs())

    assert edges.rows.count({"champion_slug": "aatrox", "related_slug": "renataglasc"}) == 1


def test_build_champion_related_drops_a_target_outside_the_roster() -> None:
    """An edge pointing at a champion that will not exist is dropped and counted."""
    payloads = universe_champions()
    payloads["aatrox"]["related-champions"].append({"slug": "unknown"})

    edges = build_champion_related(payloads, roster_slugs())

    assert all(row["related_slug"] != "unknown" for row in edges.rows)
    assert edges.dropped == 1


# ---------- story_champion tests ----------


def test_build_story_champions_unions_the_owner_and_featured_edges() -> None:
    """Every story reaches both the page that carries it and the champions it features."""
    edges = build_story_champions(universe_champions(), roster_slugs())

    assert sorted((row["story_slug"], row["champion_slug"]) for row in edges.rows) == [
        ("chemtech", "aatrox"),
        ("chemtech", "renataglasc"),
        ("darkin-blade", "aatrox"),
        ("hollowspun", "aatrox"),
        ("hollowspun", "norra"),
    ]
    assert edges.dropped == 0


def test_build_story_champions_keeps_a_featured_only_edge() -> None:
    """Aatrox is featured in a story owned by another page, an edge the owner walk misses."""
    edges = build_story_champions(universe_champions(), roster_slugs())
    owners = {
        (story_slug, slug)
        for slug, payload in universe_champions().items()
        for module in payload["modules"]
        if (story_slug := module.get("story-slug"))
    }

    assert ("chemtech", "aatrox") not in owners
    assert {"story_slug": "chemtech", "champion_slug": "aatrox"} in edges.rows


def test_build_story_champions_ignores_modules_without_a_story_slug() -> None:
    """The featured-video module carries champions but no story, so it produces nothing."""
    payloads = {"aatrox": universe_champion("aatrox")}

    edges = build_story_champions(payloads, roster_slugs())

    assert edges.rows == []
    assert edges.dropped == 0


def test_build_story_champions_deduplicates_the_owner_and_featured_edge() -> None:
    """A page featuring its own champion yields one row, as the primary key needs."""
    payloads = {"aatrox": universe_champion("aatrox", stories=(("darkin-blade", ("aatrox",)),))}

    edges = build_story_champions(payloads, roster_slugs())

    assert edges.rows == [{"story_slug": "darkin-blade", "champion_slug": "aatrox"}]


def test_build_story_champions_drops_a_featured_champion_outside_the_roster() -> None:
    """A featured champion that will not exist is dropped and counted."""
    payloads = {"aatrox": universe_champion("aatrox", stories=(("darkin-blade", ("unknown",)),))}

    edges = build_story_champions(payloads, roster_slugs())

    assert edges.rows == [{"story_slug": "darkin-blade", "champion_slug": "aatrox"}]
    assert edges.dropped == 1


# ---------- item_tag tests ----------


def test_build_item_tags_emits_one_row_per_tag() -> None:
    """Tags are stored verbatim and an item with an empty tag list contributes nothing."""
    edges = build_item_tags(ddragon_items(), item_ids())

    assert edges.rows == [
        {"item_id": "1036", "tag": "Damage"},
        {"item_id": "3035", "tag": "Damage"},
        {"item_id": "3035", "tag": "ArmorPenetration"},
    ]
    assert edges.dropped == 0


def test_build_item_tags_deduplicates_a_repeated_tag() -> None:
    """An item publishing one tag twice yields one row, as the primary key needs."""
    payload = ddragon_items()
    payload["data"]["1036"]["tags"] = ["Damage", "Damage"]

    edges = build_item_tags(payload, item_ids())

    assert edges.rows.count({"item_id": "1036", "tag": "Damage"}) == 1


def test_build_item_tags_drops_an_item_outside_the_catalogue() -> None:
    """Tags of an item the loader never persists are dropped and counted."""
    payload = ddragon_items()
    payload["data"]["9999"] = {"name": "Removed", "tags": ["Damage", "Health"]}

    edges = build_item_tags(payload, item_ids())

    assert all(row["item_id"] != "9999" for row in edges.rows)
    assert edges.dropped == 2


# ---------- item_components tests ----------


def test_build_item_components_counts_a_repeated_component() -> None:
    """A recipe listing the same component twice yields one row with quantity two."""
    edges = build_item_components(ddragon_items(), item_ids())

    assert edges.rows == [{"item_id": "3035", "component_id": "1036", "quantity": 2}]
    assert edges.dropped == 0


def test_build_item_components_skips_items_without_a_recipe() -> None:
    """An item with no "from" list contributes no association row."""
    payload = ddragon_items()
    del payload["data"]["3035"]["from"]

    assert build_item_components(payload, item_ids()).rows == []


def test_build_item_components_drops_a_component_outside_the_catalogue() -> None:
    """A recipe naming a component that will not exist loses that edge, not the row set."""
    payload = ddragon_items()
    payload["data"]["3035"]["from"] = ["1036", "9999"]

    edges = build_item_components(payload, item_ids())

    assert edges.rows == [{"item_id": "3035", "component_id": "1036", "quantity": 1}]
    assert edges.dropped == 1


# ---------- orchestrator harness ----------

EXPECTED_STATS = AssociationStats(
    champion_roles=4,
    champion_related=2,
    story_champions=5,
    item_tags=3,
    item_maps=0,
    item_components=1,
    dropped_edges=0,
)

ASSOCIATION_TABLES = {
    "champion_role": champion_role,
    "champion_related": champion_related,
    "story_champion": story_champion,
    "item_tag": item_tag,
    "item_components": item_components,
}

EXPECTED_COUNTS = {
    "champion_role": 4,
    "champion_related": 2,
    "story_champion": 5,
    "item_tag": 3,
    "item_components": 1,
}


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
    """Insert the entity rows every association edge points at.

    Args:
        session: Open Session the rows are added to.

    Returns:
        None. The rows are flushed so the association writes satisfy their
        foreign keys.
    """
    session.add(Faction(slug="unaffiliated", name="Unaffiliated"))
    session.flush()
    for slug in ROSTER_SLUGS:
        session.add(
            Champion(
                slug=slug,
                name=slug.capitalize(),
                title=f"the {slug.capitalize()}",
                faction_slug="unaffiliated",
                bio_full="Bio.",
                bio_full_text="Bio.",
                playable=slug != "norra",
            )
        )
    for slug in ROLE_SLUGS:
        session.add(Role(slug=slug, name=slug.capitalize()))
    for slug in STORY_SLUGS:
        session.add(
            Story(
                slug=slug,
                title=slug,
                word_count=2,
                subsection_count=1,
                content="A tale.",
                content_text="A tale.",
            )
        )
    for item_id, record in ddragon_items()["data"].items():
        session.add(
            Item(
                ddragon_id=item_id,
                name=record["name"],
                description="Item.",
                description_text="Item.",
                gold_total=1200,
                gold_base=500,
                purchasable=True,
                in_store=True,
            )
        )
    session.flush()


def count_rows(session: Session, table: Table) -> int:
    """Count the rows currently stored in one association table.

    Args:
        session: Open Session the count is read through.
        table: Association table to count.

    Returns:
        The number of stored rows.
    """
    return session.execute(select(func.count()).select_from(table)).scalar_one()


def stored_counts(session: Session) -> dict[str, int]:
    """Count every association table the loader writes.

    Args:
        session: Open Session the counts are read through.

    Returns:
        Mapping of table name to row count.
    """
    return {name: count_rows(session, table) for name, table in ASSOCIATION_TABLES.items()}


# ---------- orchestrator tests ----------


def test_load_associations_persists_every_table(db_session: Session) -> None:
    """One run lands every fixture edge and reports matching counts."""
    seed_entities(db_session)

    stats = load_associations(
        db_session, ddragon_champion_list(), ddragon_items(), universe_champions()
    )

    assert stats == EXPECTED_STATS
    assert stored_counts(db_session) == EXPECTED_COUNTS


def test_load_associations_run_twice_leaves_the_same_rows(db_session: Session) -> None:
    """Delete-then-insert makes a repeated run change nothing."""
    seed_entities(db_session)
    first = load_associations(
        db_session, ddragon_champion_list(), ddragon_items(), universe_champions()
    )

    second = load_associations(
        db_session, ddragon_champion_list(), ddragon_items(), universe_champions()
    )

    assert second == first
    assert stored_counts(db_session) == EXPECTED_COUNTS


def test_load_associations_removes_an_edge_the_source_dropped(db_session: Session) -> None:
    """An edge the source stops publishing leaves the table, which upserting would not do."""
    seed_entities(db_session)
    load_associations(db_session, ddragon_champion_list(), ddragon_items(), universe_champions())

    payload = ddragon_items()
    del payload["data"]["3035"]["from"]
    load_associations(db_session, ddragon_champion_list(), payload, universe_champions())

    assert count_rows(db_session, item_components) == 0


def test_load_associations_stores_the_component_quantity(db_session: Session) -> None:
    """The doubled component lands as one row carrying quantity two."""
    seed_entities(db_session)

    load_associations(db_session, ddragon_champion_list(), ddragon_items(), universe_champions())

    rows = db_session.execute(
        select(
            item_components.c.item_id,
            item_components.c.component_id,
            item_components.c.quantity,
        )
    ).all()
    assert rows == [("3035", "1036", 2)]


def test_load_associations_counts_a_dropped_edge_without_failing(db_session: Session) -> None:
    """A dangling edge is counted rather than crashing the load or reaching the database."""
    seed_entities(db_session)
    payloads = universe_champions()
    payloads["aatrox"]["related-champions"].append({"slug": "unknown"})

    stats = load_associations(db_session, ddragon_champion_list(), ddragon_items(), payloads)

    assert stats.dropped_edges == 1
    assert stats.champion_related == 2
    assert count_rows(db_session, champion_related) == 2


def test_load_associations_leaves_the_roles_relationship_readable(db_session: Session) -> None:
    """Champion.roles is viewonly, so reads still traverse the rows this loader wrote."""
    seed_entities(db_session)

    load_associations(db_session, ddragon_champion_list(), ddragon_items(), universe_champions())
    db_session.expire_all()

    aatrox = db_session.get(Champion, "aatrox")
    assert aatrox is not None
    assert sorted(role.slug for role in aatrox.roles) == ["fighter", "tank"]
