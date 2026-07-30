import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Table, delete, insert
from sqlalchemy.orm import Session

from lolrag.db.models import (
    champion_related,
    champion_role,
    item_components,
    item_tag,
    story_champion,
)
from lolrag.ingest.identifiers import universe_slug

logger = logging.getLogger(__name__)

STORY_SLUG_KEY = "story-slug"
FEATURED_CHAMPIONS_KEY = "featured-champions"
RELATED_CHAMPIONS_KEY = "related-champions"


# ---------- builders ----------


@dataclass(frozen=True)
class Edges:
    """Association rows built from one source, with the dangling edges counted.

    Args:
        rows: One mapping per association row, keyed by column name, in source
            order and free of duplicate primary keys.
        dropped: Number of source edges discarded because an endpoint resolves
            to no entity row.
    """

    rows: list[dict[str, Any]]
    dropped: int


def _log_dangling(table: str, edge: tuple[str, str], missing: str) -> None:
    """Log one association edge discarded because an endpoint will not exist.

    Args:
        table: Name of the association table the edge would have been written to.
        edge: The two endpoints of the dropped edge, in column order.
        missing: The endpoint that resolves to no entity row.

    Returns:
        None. Callers count the drop themselves.
    """
    logger.warning("dropping %s edge %s: %s is absent from the corpus", table, edge, missing)


def build_champion_roles(champion_list: dict, champion_slugs: set[str]) -> Edges:
    """Build the champion_role rows from the Data Dragon class tags.

    Args:
        champion_list: Parsed Data Dragon champion.json body, whose "data" key
            maps champion id to a summary carrying a "tags" list.
        champion_slugs: Universe slugs of the champions that will exist, used to
            drop edges pointing at a champion the loader never persists.

    Returns:
        Edges holding one row per distinct (champion slug, role slug) pair, the
        role slug being the lowercased tag. The champions table is keyed on the
        Universe slug, so every Data Dragon id is mapped through universe_slug;
        a lore-only character has no Data Dragon entry and so gets no roles.
    """
    edges: dict[tuple[str, str], None] = {}
    dropped = 0
    for ddragon_id, entry in champion_list["data"].items():
        slug = universe_slug(ddragon_id)
        tags = list(dict.fromkeys(entry.get("tags") or []))
        if slug not in champion_slugs:
            _log_dangling("champion_role", (slug, ",".join(tags)), slug)
            dropped += len(tags)
            continue
        for tag in tags:
            edges[(slug, tag.lower())] = None
    rows = [{"champion_slug": slug, "role_slug": role} for slug, role in edges]
    return Edges(rows=rows, dropped=dropped)


def build_champion_related(
    universe_champions: Mapping[str, Mapping[str, Any]], champion_slugs: set[str]
) -> Edges:
    """Build the champion_related rows exactly as Riot publishes them.

    Args:
        universe_champions: Mapping of Universe champion slug to that champion's
            parsed Universe payload, whose top-level "related-champions" array
            carries one "slug" per entry.
        champion_slugs: Universe slugs of the champions that will exist, used to
            drop edges pointing at a champion the loader never persists.

    Returns:
        Edges holding one row per distinct (champion, related champion) pair.
        The relation is stored directed and is deliberately not symmetrized:
        fewer than two thirds of the published edges have a published reverse,
        and inventing the rest would put edges in the corpus that Riot never
        published.
    """
    edges: dict[tuple[str, str], None] = {}
    dropped = 0
    for slug, payload in universe_champions.items():
        for entry in payload.get(RELATED_CHAMPIONS_KEY) or []:
            related_slug = entry["slug"]
            if related_slug not in champion_slugs:
                _log_dangling("champion_related", (slug, related_slug), related_slug)
                dropped += 1
                continue
            edges[(slug, related_slug)] = None
    rows = [{"champion_slug": slug, "related_slug": related} for slug, related in edges]
    return Edges(rows=rows, dropped=dropped)


def build_story_champions(
    universe_champions: Mapping[str, Mapping[str, Any]], champion_slugs: set[str]
) -> Edges:
    """Build the story_champion rows from the champion pages' story modules.

    Args:
        universe_champions: Mapping of Universe champion slug to that champion's
            parsed Universe payload, whose "modules" list holds the story
            previews.
        champion_slugs: Universe slugs of the champions that will exist, used to
            drop edges pointing at a champion the loader never persists.

    Returns:
        Edges holding the union of two published edges: the owner edge, from a
        story preview to the champion whose page carries it, and the featured
        edge, from the same preview to every champion it lists under
        "featured-champions". Both are needed because a handful of stories are
        featured on a page other than their owner's. The story payload itself
        publishes no champion list, so this is the only source of the relation.
        Every story slug reached here is one the loader fetches, so the story
        side of the edge cannot dangle.
    """
    edges: dict[tuple[str, str], None] = {}
    dropped = 0
    for slug, payload in universe_champions.items():
        for module in payload.get("modules") or []:
            story_slug = module.get(STORY_SLUG_KEY)
            if not story_slug:
                continue
            featured = [entry["slug"] for entry in module.get(FEATURED_CHAMPIONS_KEY) or []]
            for champion_slug in (slug, *featured):
                if champion_slug not in champion_slugs:
                    _log_dangling("story_champion", (story_slug, champion_slug), champion_slug)
                    dropped += 1
                    continue
                edges[(story_slug, champion_slug)] = None
    rows = [{"story_slug": story, "champion_slug": champion} for story, champion in edges]
    return Edges(rows=rows, dropped=dropped)


def build_item_tags(item_payload: dict, item_ids: set[str]) -> Edges:
    """Build the item_tag rows from the Data Dragon item tags.

    Args:
        item_payload: Parsed Data Dragon item.json body, whose "data" key maps
            item id to a record carrying an optional "tags" list.
        item_ids: Ids of the items that will exist, used to drop edges pointing
            at an item the loader never persists.

    Returns:
        Edges holding one row per distinct (item id, tag) pair. The tag is
        stored verbatim rather than slugged, there being no roles table for it
        to key against. Items publishing no tags contribute nothing.
    """
    edges: dict[tuple[str, str], None] = {}
    dropped = 0
    for item_id, record in item_payload["data"].items():
        tags = list(dict.fromkeys(record.get("tags") or []))
        if item_id not in item_ids:
            _log_dangling("item_tag", (item_id, ",".join(tags)), item_id)
            dropped += len(tags)
            continue
        for tag in tags:
            edges[(item_id, tag)] = None
    rows = [{"item_id": item_id, "tag": tag} for item_id, tag in edges]
    return Edges(rows=rows, dropped=dropped)


def build_item_components(item_payload: dict, item_ids: set[str]) -> Edges:
    """Build the item_components rows, counting repeated components.

    Args:
        item_payload: Parsed Data Dragon item.json body, whose "data" key maps
            item id to a record carrying an optional "from" list of component
            ids.
        item_ids: Ids of the items that will exist, used to drop edges pointing
            at an item the loader never persists.

    Returns:
        Edges holding one row per distinct (item id, component id) pair, with
        quantity counting how many times the recipe lists that component, so a
        recipe consuming two copies of one component is not flattened to one.
    """
    rows: list[dict[str, Any]] = []
    dropped = 0
    for item_id, record in item_payload["data"].items():
        counts = Counter(record.get("from") or [])
        if item_id not in item_ids:
            _log_dangling("item_components", (item_id, ",".join(counts)), item_id)
            dropped += len(counts)
            continue
        for component_id, quantity in counts.items():
            if component_id not in item_ids:
                _log_dangling("item_components", (item_id, component_id), component_id)
                dropped += 1
                continue
            rows.append({"item_id": item_id, "component_id": component_id, "quantity": quantity})
    return Edges(rows=rows, dropped=dropped)


# ---------- orchestration ----------


@dataclass(frozen=True)
class AssociationStats:
    """Counts describing one association load run.

    Args:
        champion_roles: Number of champion_role rows persisted.
        champion_related: Number of champion_related rows persisted.
        story_champions: Number of story_champion rows persisted.
        item_tags: Number of item_tag rows persisted.
        item_components: Number of item_components rows persisted.
        dropped_edges: Edges discarded across all five tables because an
            endpoint resolved to no entity row.
    """

    champion_roles: int
    champion_related: int
    story_champions: int
    item_tags: int
    item_components: int
    dropped_edges: int


def _replace(session: Session, table: Table, rows: Sequence[Mapping[str, Any]]) -> None:
    """Rewrite one association table with the given rows.

    Args:
        session: Open Session the statements are executed on.
        table: Association table to rewrite.
        rows: Mappings keyed by column name, already deduplicated on the
            primary key.

    Returns:
        None. The table is emptied and refilled rather than upserted: these
        tables are derived data with no surrogate key and nothing referencing
        them, so a rewrite makes the row count equal len(rows) by construction
        and leaves no stale edge behind when a source drops one.
    """
    session.execute(delete(table))
    if not rows:
        return
    session.execute(insert(table).values(list(rows)))


def load_associations(
    session: Session,
    champion_list: dict,
    item_payload: dict,
    universe_champions: dict[str, dict],
) -> AssociationStats:
    """Rewrite every association table from payloads the caller already holds.

    Args:
        session: Open Session the statements are executed on. Changes are
            flushed but never committed, so the caller decides whether the run
            is kept. The entity loader must have run first, because every
            endpoint of every edge is a real foreign key.
        champion_list: Parsed Data Dragon champion.json body, supplying the
            class tags.
        item_payload: Parsed Data Dragon item.json body, supplying the item tags
            and build recipes.
        universe_champions: Mapping of Universe champion slug to that champion's
            parsed Universe payload, supplying the related-champion edges and
            the story modules. Its keys are the authoritative champion roster.

    Returns:
        AssociationStats with one count per table plus the total number of
        dangling edges dropped.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If any row violates the schema.
    """
    champion_slugs = set(universe_champions)
    item_ids = set(item_payload["data"])

    roles = build_champion_roles(champion_list, champion_slugs)
    related = build_champion_related(universe_champions, champion_slugs)
    stories = build_story_champions(universe_champions, champion_slugs)
    tags = build_item_tags(item_payload, item_ids)
    components = build_item_components(item_payload, item_ids)

    _replace(session, champion_role, roles.rows)
    _replace(session, champion_related, related.rows)
    _replace(session, story_champion, stories.rows)
    _replace(session, item_tag, tags.rows)
    _replace(session, item_components, components.rows)
    session.flush()

    stats = AssociationStats(
        champion_roles=len(roles.rows),
        champion_related=len(related.rows),
        story_champions=len(stories.rows),
        item_tags=len(tags.rows),
        item_components=len(components.rows),
        dropped_edges=roles.dropped
        + related.dropped
        + stories.dropped
        + tags.dropped
        + components.dropped,
    )
    logger.info("loaded associations: %s", stats)
    return stats
