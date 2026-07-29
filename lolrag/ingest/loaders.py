import asyncio
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from lolrag.config import Settings
from lolrag.db.models import (
    Ability,
    Base,
    Champion,
    Faction,
    Item,
    Role,
    Rune,
    RunePath,
    Story,
    SummonerSpell,
)
from lolrag.fetch import ddragon, universe
from lolrag.fetch.client import FetchClient
from lolrag.ingest.markup import clean_markup, clean_optional_markup

logger = logging.getLogger(__name__)

UNAFFILIATED_SLUG = "unaffiliated"
UNAFFILIATED_NAME = "Unaffiliated"

SPELL_SLOTS = ("Q", "W", "E", "R")
PASSIVE_SLOT = "P"

UNIVERSE_SLUG_OVERRIDES = {"Renata": "renataglasc"}

PLACEHOLDER_SPELL_SUFFIX = "Placeholder"

STORY_BLOCK_SEPARATOR = "\n\n"


# ---------- helpers ----------


def universe_slug(ddragon_id: str) -> str:
    """Return the Riot Universe slug for a Data Dragon champion id.

    Args:
        ddragon_id: Data Dragon champion id string, e.g. "Aatrox" or
            "MonkeyKing".

    Returns:
        The Universe slug, which is the lowercased id for every champion
        except the ones listed in UNIVERSE_SLUG_OVERRIDES.
    """
    return UNIVERSE_SLUG_OVERRIDES.get(ddragon_id, ddragon_id.lower())


def parse_release_date(value: str | None) -> datetime | None:
    """Parse a Riot ISO 8601 release date into a naive UTC datetime.

    Args:
        value: Release date string such as "2013-06-13T19:43:26.000Z", or
            None/empty when the source publishes no date.

    Returns:
        The instant as a naive UTC datetime, matching the TIMESTAMP WITHOUT
        TIME ZONE columns, or None when value is absent or unparseable.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("unparseable release date %r", value)
        return None
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(UTC).replace(tzinfo=None)


# ---------- builders ----------


def build_factions(payloads: list[dict]) -> list[Faction]:
    """Build the faction rows, led by the synthetic "unaffiliated" faction.

    Args:
        payloads: Parsed Riot Universe faction payloads, each carrying a
            "faction" object with slug, name and an "overview" object.

    Returns:
        Faction rows with the synthetic "unaffiliated" row first, so callers
        that persist in list order satisfy the champions.faction_slug foreign
        key for champions that belong to no published faction.
    """
    factions = [Faction(slug=UNAFFILIATED_SLUG, name=UNAFFILIATED_NAME)]
    for payload in payloads:
        faction = payload["faction"]
        slug = faction["slug"]
        if slug == UNAFFILIATED_SLUG:
            continue
        overview = (faction.get("overview") or {}).get("short")
        factions.append(
            Faction(
                slug=slug,
                name=faction["name"],
                overview=overview,
                overview_text=clean_optional_markup(overview),
            )
        )
    return factions


def build_roles(champion_list: dict) -> list[Role]:
    """Build one role row per distinct Data Dragon champion tag.

    Args:
        champion_list: Parsed Data Dragon champion.json body, whose "data" key
            maps champion id to a summary carrying a "tags" list.

    Returns:
        Role rows in alphabetical slug order, the slug being the lowercased
        tag.
    """
    names = {tag for entry in champion_list["data"].values() for tag in entry.get("tags", [])}
    return [Role(slug=name.lower(), name=name) for name in sorted(names)]


def build_champions(champion_list: dict, universe_payloads: dict[str, dict]) -> list[Champion]:
    """Build champion rows from the Universe roster, enriched by Data Dragon.

    Args:
        champion_list: Parsed Data Dragon champion.json body, supplying the
            Data Dragon id string for playable champions.
        universe_payloads: Mapping of Universe champion slug to that
            champion's parsed Universe payload. This is the authoritative
            roster: characters present here but absent from champion_list are
            lore-only.

    Returns:
        One Champion per Universe payload, with ddragon_key set to the Data
        Dragon id string and playable False for lore-only characters. The
        faction falls back to the synthetic "unaffiliated" slug whenever the
        payload publishes no faction. Roles are not attached here: champion_role
        is written by the association loader, which owns it outright.
    """
    data = champion_list["data"]
    ddragon_id_by_slug = {universe_slug(ddragon_id): ddragon_id for ddragon_id in data}

    champions: list[Champion] = []
    for slug, payload in universe_payloads.items():
        record = payload["champion"]
        biography = record["biography"]
        bio_full = biography["full"]
        bio_short = biography.get("short")
        ddragon_key = ddragon_id_by_slug.get(slug)
        champions.append(
            Champion(
                slug=slug,
                ddragon_key=ddragon_key,
                name=record["name"],
                title=record["title"],
                faction_slug=record.get("associated-faction-slug") or UNAFFILIATED_SLUG,
                bio_full=bio_full,
                bio_full_text=clean_markup(bio_full),
                bio_short=bio_short,
                bio_short_text=clean_optional_markup(bio_short),
                playable=ddragon_key is not None,
                release_date=parse_release_date(record.get("release-date")),
            )
        )
    return champions


def build_abilities(details: dict[str, dict]) -> list[Ability]:
    """Build the passive and Q/W/E/R ability rows for every champion detail.

    Args:
        details: Mapping of Data Dragon champion id to that champion's parsed
            detail body, whose "data" key holds the record with "passive" and
            "spells".

    Returns:
        Five Ability rows per champion: the passive in slot P with max_rank
        None, then the four spells in "spells" order as slots Q, W, E and R
        carrying their own maxrank.

    Raises:
        ValueError: If a champion publishes a number of spells other than the
            four Q/W/E/R slots.
    """
    abilities: list[Ability] = []
    for ddragon_id, payload in details.items():
        slug = universe_slug(ddragon_id)
        record = payload["data"][ddragon_id]
        passive = record["passive"]
        abilities.append(
            Ability(
                champion_slug=slug,
                slot=PASSIVE_SLOT,
                name=passive["name"],
                description=passive["description"],
                max_rank=None,
            )
        )
        for slot, spell in zip(SPELL_SLOTS, record["spells"], strict=True):
            tooltip = spell.get("tooltip")
            abilities.append(
                Ability(
                    champion_slug=slug,
                    slot=slot,
                    name=spell["name"],
                    description=spell["description"],
                    tooltip=tooltip,
                    tooltip_text=clean_optional_markup(tooltip),
                    max_rank=spell["maxrank"],
                )
            )
    return abilities


def build_items(payload: dict) -> list[Item]:
    """Build the item rows, without their build recipes.

    Args:
        payload: Parsed Data Dragon item.json body, whose "data" key maps item
            id to a record carrying name, description, plaintext, gold and an
            optional "from" list of component ids.

    Returns:
        One Item per record. Recipes are not attached here because a component
        link carries a quantity, which the plain many-to-many relationship
        cannot express; the association loader writes those rows instead.
    """
    return [
        Item(
            ddragon_id=item_id,
            name=record["name"],
            description=record["description"],
            description_text=clean_markup(record["description"]),
            plaintext=record.get("plaintext"),
            gold_total=record["gold"]["total"],
            gold_base=record["gold"]["base"],
            depth=record.get("depth"),
        )
        for item_id, record in payload["data"].items()
    ]


def build_rune_paths(payload: list[dict]) -> list[RunePath]:
    """Build rune path rows with their runes positioned by row and order.

    Args:
        payload: Parsed Data Dragon runesReforged.json body, a list of path
            records each carrying id, key, name and a "slots" list whose
            entries hold a "runes" list.

    Returns:
        One RunePath per record, keeping Data Dragon's own numeric ids for both
        the path and its runes, with row_index the slot position in "slots" and
        position_index the rune position within its slot.
    """
    paths: list[RunePath] = []
    for record in payload:
        path = RunePath(id=record["id"], key=record["key"], name=record["name"])
        for row_index, slot in enumerate(record["slots"]):
            for position_index, rune in enumerate(slot["runes"]):
                path.runes.append(
                    Rune(
                        id=int(rune["id"]),
                        key=rune["key"],
                        name=rune["name"],
                        short_desc=rune["shortDesc"],
                        short_desc_text=clean_markup(rune["shortDesc"]),
                        long_desc=rune["longDesc"],
                        long_desc_text=clean_markup(rune["longDesc"]),
                        row_index=row_index,
                        position_index=position_index,
                    )
                )
        paths.append(path)
    return paths


def build_summoner_spells(payload: dict) -> list[SummonerSpell]:
    """Build summoner spell rows from the Data Dragon summoner file.

    Args:
        payload: Parsed Data Dragon summoner.json body, whose "data" key maps
            spell id to a record carrying key, name, description, a "cooldown"
            list and "summonerLevel".

    Returns:
        One SummonerSpell per record whose id does not end in
        PLACEHOLDER_SPELL_SUFFIX, those being engine scaffolding rather than
        castable spells. cooldown is the first element of the cooldown list in
        seconds, or None when the list is empty.
    """
    spells: list[SummonerSpell] = []
    for spell_id, record in payload["data"].items():
        if spell_id.endswith(PLACEHOLDER_SPELL_SUFFIX):
            logger.debug("skipping placeholder summoner spell %s", spell_id)
            continue
        cooldown = record.get("cooldown") or []
        description = record["description"]
        spells.append(
            SummonerSpell(
                id=spell_id,
                key=record["key"],
                name=record["name"],
                description=description,
                description_text=clean_markup(description),
                cooldown=float(cooldown[0]) if cooldown else None,
                summoner_level=record.get("summonerLevel"),
            )
        )
    return spells


def build_stories(payloads: dict[str, dict]) -> list[Story]:
    """Build story rows by walking sections and their subsections.

    Args:
        payloads: Mapping of Universe story slug to that story's parsed
            payload, whose "story" object holds "story-sections", each of which
            holds "story-subsections" carrying the "content" blocks.

    Returns:
        One Story per payload; content joins every non-empty subsection block
        in source order, subsection_count counts those blocks and word_count is
        the whitespace-separated word count of the cleaned content. author is
        always None because no permitted source publishes it.
    """
    stories: list[Story] = []
    for slug, payload in payloads.items():
        record = payload["story"]
        blocks = [
            subsection["content"]
            for section in record.get("story-sections") or []
            for subsection in section.get("story-subsections") or []
            if subsection.get("content")
        ]
        content = STORY_BLOCK_SEPARATOR.join(blocks)
        content_text = clean_markup(content)
        stories.append(
            Story(
                slug=slug,
                title=record["title"],
                author=None,
                word_count=len(content_text.split()),
                subsection_count=len(blocks),
                content=content,
                content_text=content_text,
                release_date=parse_release_date(payload.get("release-date")),
            )
        )
    return stories


# ---------- orchestration ----------


@dataclass(frozen=True)
class LoadStats:
    """Counts describing one entity load run.

    Args:
        factions: Number of faction rows persisted, including the synthetic
            "unaffiliated" row.
        roles: Number of role rows persisted.
        champions: Number of champion rows persisted.
        abilities: Number of ability rows persisted.
        stories: Number of story rows persisted.
        items: Number of item rows persisted.
        rune_paths: Number of rune path rows persisted.
        runes: Number of rune rows persisted.
        summoner_spells: Number of summoner spell rows persisted.
        unaffiliated_champions: Champions whose faction fell back to the
            synthetic "unaffiliated" faction.
    """

    factions: int
    roles: int
    champions: int
    abilities: int
    stories: int
    items: int
    rune_paths: int
    runes: int
    summoner_spells: int
    unaffiliated_champions: int


def _merge_all(session: Session, rows: Iterable[Base]) -> None:
    """Upsert every row through Session.merge.

    Args:
        session: Open Session the rows are merged into.
        rows: ORM instances whose primary keys are already set, so merge
            updates the existing row instead of inserting a duplicate.

    Returns:
        None. Nothing is flushed; the caller controls the transaction.
    """
    for row in rows:
        session.merge(row)


def _assign_existing_ability_ids(session: Session, abilities: Sequence[Ability]) -> None:
    """Give each built ability the surrogate id its natural key already holds.

    Args:
        session: Open Session queried for the stored (champion_slug, slot)
            to id mapping.
        abilities: Built abilities whose id is filled in when a row for the
            same champion and slot is already stored.

    Returns:
        None. Abilities with no stored row keep id None and are inserted,
        which is what makes a repeated load idempotent despite abilities.id
        being a surrogate key.
    """
    stored = {
        (champion_slug, slot): ability_id
        for ability_id, champion_slug, slot in session.execute(
            select(Ability.id, Ability.champion_slug, Ability.slot)
        )
    }
    for ability in abilities:
        ability.id = stored.get((ability.champion_slug, ability.slot))


async def load_all(session: Session, client: FetchClient, settings: Settings) -> LoadStats:
    """Populate every entity table from the corpus, upserting on natural keys.

    Args:
        session: Open Session the rows are merged into. The transaction is
            left open and unflushed changes are flushed but never committed,
            so the caller decides whether the run is kept.
        client: Open FetchClient serving the corpus, from the on-disk cache
            when it is warm.
        settings: Application settings providing every ddragon_* and
            universe_* value plus cache_dir.

    Returns:
        LoadStats with one count per entity table plus the number of champions
        that fell back to the synthetic "unaffiliated" faction. The association
        tables are rewritten too, once every entity they reference exists, and
        their own counts are logged rather than returned.

    Raises:
        httpx.HTTPStatusError: If any request fails after its retries.
        sqlalchemy.exc.SQLAlchemyError: If any row violates the schema.
    """
    from lolrag.ingest.associations import load_associations

    champion_list, item_payload, rune_payload, spell_payload = await asyncio.gather(
        ddragon.fetch_champion_list(client, settings),
        ddragon.fetch_items(client, settings),
        ddragon.fetch_runes(client, settings),
        ddragon.fetch_summoner_spells(client, settings),
    )
    champion_ids = list(champion_list["data"])
    details = await ddragon.fetch_all_champion_details(client, settings, champion_ids)
    logger.info("loaded %d Data Dragon champion details", len(details))

    search_index = await universe.fetch_search_index(client, settings)
    champion_slugs = [entry["slug"] for entry in search_index["champions"]]
    faction_slugs = [entry["slug"] for entry in search_index["factions"]]
    universe_champions = await universe.fetch_all_champions(client, settings, champion_slugs)
    universe_factions = await universe.fetch_all_factions(client, settings, faction_slugs)

    story_slugs: list[str] = []
    seen: set[str] = set()
    for payload in universe_champions.values():
        for slug in universe.extract_story_slugs(payload):
            if slug not in seen:
                seen.add(slug)
                story_slugs.append(slug)
    universe_stories = await universe.fetch_all_stories(client, settings, story_slugs)
    logger.info(
        "loaded %d Universe champions, %d factions and %d stories",
        len(universe_champions),
        len(universe_factions),
        len(universe_stories),
    )

    factions = build_factions(list(universe_factions.values()))
    _merge_all(session, factions)
    session.flush()

    roles = build_roles(champion_list)
    _merge_all(session, roles)
    session.flush()

    champions = build_champions(champion_list, universe_champions)
    _merge_all(session, champions)
    session.flush()

    abilities = build_abilities(details)
    _assign_existing_ability_ids(session, abilities)
    _merge_all(session, abilities)
    session.flush()

    stories = build_stories(universe_stories)
    _merge_all(session, stories)
    session.flush()

    items = build_items(item_payload)
    _merge_all(session, items)
    session.flush()

    rune_paths = build_rune_paths(rune_payload)
    _merge_all(session, rune_paths)
    session.flush()

    summoner_spells = build_summoner_spells(spell_payload)
    _merge_all(session, summoner_spells)
    session.flush()

    load_associations(session, champion_list, item_payload, universe_champions)

    stats = LoadStats(
        factions=len(factions),
        roles=len(roles),
        champions=len(champions),
        abilities=len(abilities),
        stories=len(stories),
        items=len(items),
        rune_paths=len(rune_paths),
        runes=sum(len(path.runes) for path in rune_paths),
        summoner_spells=len(summoner_spells),
        unaffiliated_champions=sum(
            1 for champion in champions if champion.faction_slug == UNAFFILIATED_SLUG
        ),
    )
    logger.info("loaded entities: %s", stats)
    return stats
