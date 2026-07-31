"""Retrieval documents and their embedded chunks, built from the stored entities.

Every entity that carries prose gets exactly one document, keyed on the entity's
own identity rather than on a hash of its content: a content-addressed key would
orphan the stored row the moment the source text changed.

MAP_NAMES merges two first-party sources. Data Dragon map.json takes precedence
and supplies every id it names; Community Dragon v1/maps.json fills ids 33 and
35, which Data Dragon publishes as empty strings. The precedence is load-bearing
rather than arbitrary: Community Dragon calls map 12 "Random Map", which is
wrong, so it must never be used alone. The names are frozen here as a module
constant because they are seven strings that change on the timescale of years,
and fetching them would add two endpoints to every ingest for no new fact.
"""

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import delete, exists, func, insert, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.orm import Session, selectinload

from lolrag.config import Settings
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
    Story,
    SummonerSpell,
    item_map,
    item_tag,
)
from lolrag.embeddings import get_embeddings
from lolrag.ingest.formatting import SIGNIFICANT_DIGITS, STAT_NAMES, _round_significant
from lolrag.ingest.markup import clean_markup

logger = logging.getLogger(__name__)

COLLECTION_ABILITIES = "abilities"
COLLECTION_EQUIPMENT = "equipment"
COLLECTION_LORE = "lore"

MAP_NAMES = {
    11: "Summoner's Rift",
    12: "Howling Abyss",
    21: "Nexus Blitz",
    22: "Teamfight Tactics",
    30: "Arena",
    33: "Swarm",
    35: "The Bandlewood",
}

ENTITY_COLUMNS = (
    "champion_slug",
    "story_slug",
    "faction_slug",
    "ability_id",
    "item_id",
    "rune_id",
    "summoner_spell_id",
)

KIND_BY_LEVEL = "by_level"

KEYSTONE_ROW_INDEX = 0

PERCENT_SCALE = 100.0
MIN_CHARACTER_LEVEL = 1
MAX_CHARACTER_LEVEL = 18

RANK_SEPARATOR = "/"
BLOCK_SEPARATOR = "\n\n"
LINE_SEPARATOR = "\n"
LIST_SEPARATOR = ", "
VALUES_HEADER = "Values:"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
CHUNK_SEPARATORS = ["\n\n\n", "\n\n", "\n", " ", ""]

EMBED_BATCH_SIZE = 256
UPSERT_BATCH_SIZE = 500


# ---------- rendered numbers ----------


def _format_number(value: float) -> str:
    """Render one number without scientific notation or float32 noise.

    Args:
        value: The raw number.

    Returns:
        The number rounded to four significant digits and stripped of trailing
        zeros, so a stored 0.30000001192092896 reads as 0.3. A value that rounds
        to nothing negative renders as "0" rather than "-0".
    """
    text = f"{_round_significant(value, SIGNIFICANT_DIGITS):.10f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("", "-", "-0") else text


def _format_numbers(kind: str, values: Sequence[float], display_as_percent: bool) -> str:
    """Render one value row's numbers according to the shape its kind declares.

    Args:
        kind: Shape of the values array, one of per_rank, by_level, scalar,
            ratio.
        values: Numeric values in source order.
        display_as_percent: Whether the source displays these as percentages,
            in which case every number is multiplied by 100 and given a percent
            sign, so a stored 0.3 reads as "30%" instead of "0.3".

    Returns:
        The numbers as text: a by-level row as an explicit level-1 to level-18
        range, because its two entries are the endpoints of a linear
        interpolation and reading them as two bare numbers would state a fact
        the source never published; anything else as its entries joined with a
        slash, which collapses to one number for a single entry.
    """
    scale = PERCENT_SCALE if display_as_percent else 1.0
    suffix = "%" if display_as_percent else ""
    rendered = [f"{_format_number(value * scale)}{suffix}" for value in values]
    if not rendered:
        return ""
    if kind == KIND_BY_LEVEL:
        return (
            f"{rendered[0]} at level {MIN_CHARACTER_LEVEL}"
            f" to {rendered[-1]} at level {MAX_CHARACTER_LEVEL}"
        )
    return RANK_SEPARATOR.join(rendered)


def _stat_label(scaling_stat: str | None, stat_formula: str | None) -> str | None:
    """Name the champion stat a value scales with.

    Args:
        scaling_stat: Stored scaling stat code, or None when the value does not
            scale or the source enum is undecoded.
        stat_formula: Stored "total" or "bonus", or None when the source
            declared neither.

    Returns:
        The readable stat name preceded by "total" or "bonus", or None when
        there is no stat to name. A formula with no stat behind it renders
        nothing rather than a dangling "bonus": the corpus carries eleven such
        rows, and naming a stat that was never proven would be a guess.
    """
    if scaling_stat is None:
        return None
    name = STAT_NAMES.get(scaling_stat, scaling_stat)
    if stat_formula is None:
        return name
    return f"{stat_formula} {name}"


def render_value(
    name: str,
    kind: str,
    values: Sequence[float],
    *,
    display_as_percent: bool = False,
    scaling_stat: str | None = None,
    stat_formula: str | None = None,
    damage_type: str | None = None,
) -> str:
    """Render one stored numeric value row as a single document line.

    Args:
        name: Source value name, e.g. QBaseDamage or CooldownTime.
        kind: Shape of the values array, one of per_rank, by_level, scalar,
            ratio.
        values: Numeric values in source order.
        display_as_percent: Whether the source displays these as percentages.
        scaling_stat: Champion stat the value is a coefficient of, or None.
        stat_formula: Which amount of that stat, one of total, bonus, or None.
        damage_type: Damage type the value contributes, one of magic, physical,
            true, or None when the source declares none.

    Returns:
        The line "{name}: {numbers}", extended with " of {stat}" when the value
        is a stat coefficient and with ", {type} damage" when the source names a
        damage type. Every number is routed through four-significant-digit
        rounding, so the float32 noise the source carries never reaches the
        corpus.
    """
    line = f"{name}: {_format_numbers(kind, values, display_as_percent)}"
    label = _stat_label(scaling_stat, stat_formula)
    if label is not None:
        line = f"{line} of {label}"
    if damage_type is not None:
        line = f"{line}, {damage_type} damage"
    return line


def _ability_value_lines(values: Iterable[AbilityValue]) -> list[str]:
    """Render one ability's stored value rows in a stable order.

    Args:
        values: The ability's AbilityValue rows, in any order.

    Returns:
        One line per row, ordered by spell key and then value name so a rebuild
        over unchanged data produces byte-identical content and the upsert's
        change guard stays meaningful.
    """
    return [
        render_value(
            row.name,
            row.kind,
            row.values,
            display_as_percent=row.display_as_percent,
            scaling_stat=row.scaling_stat,
            stat_formula=row.stat_formula,
            damage_type=row.damage_type,
        )
        for row in sorted(values, key=lambda row: (row.spell_key, row.name))
    ]


def _item_value_lines(values: Iterable[ItemValue]) -> list[str]:
    """Render one item's stored value rows in a stable order.

    Args:
        values: The item's ItemValue rows, in any order.

    Returns:
        One line per row, ordered by value name. Item values carry no scaling
        stat or damage type, neither source publishing one for equipment.
    """
    return [
        render_value(row.name, row.kind, row.values, display_as_percent=row.display_as_percent)
        for row in sorted(values, key=lambda row: row.name)
    ]


# ---------- document builders ----------


@dataclass(frozen=True)
class BuiltDocument:
    """One document assembled in memory, before it reaches the database.

    Args:
        doc_key: Deterministic key derived from the entity's identity.
        collection: Logical collection, one of abilities, equipment, lore.
        entity_column: Name of the documents column holding this document's
            entity foreign key; the other six stay NULL.
        entity_id: Value of that foreign key.
        title: Document title, the same line the content opens with.
        source: Human-readable provenance string.
        content: Full document text, whose first line is title.
    """

    doc_key: str
    collection: str
    entity_column: str
    entity_id: str | int
    title: str
    source: str
    content: str


def _join_blocks(blocks: Iterable[str | None]) -> str:
    """Join document blocks with a blank line, dropping the empty ones.

    Args:
        blocks: Document blocks in order, any of which may be None or empty
            because the source publishes nothing for it.

    Returns:
        The surviving blocks joined by a blank line. Nothing is truncated; a
        block is dropped only when it carries no text at all.
    """
    return BLOCK_SEPARATOR.join(block.strip() for block in blocks if block and block.strip())


def _values_block(lines: Sequence[str]) -> str | None:
    """Render the trailing values block of a document.

    Args:
        lines: Rendered value lines, possibly empty.

    Returns:
        The values header followed by one line per value, or None when the
        entity publishes no numeric values and the header would head nothing.
    """
    if not lines:
        return None
    return LINE_SEPARATOR.join([VALUES_HEADER, *lines])


def build_ability_document(ability: Ability) -> BuiltDocument:
    """Build the document for one champion ability.

    Args:
        ability: Stored Ability with its champion and values loaded.

    Returns:
        A BuiltDocument in the abilities collection. The description is cleaned
        here rather than read from a stored column because abilities publish no
        markup-stripped description, and 133 of the 865 carry markup. The
        resolved tooltip block is omitted entirely when NULL, which is every
        passive and every spell whose substitution a token blocked.
    """
    champion = ability.champion
    title = f"{champion.name} {ability.slot}: {ability.name}"
    header = LINE_SEPARATOR.join([title, f"Champion: {champion.name}, {champion.title}"])
    content = _join_blocks(
        [
            header,
            clean_markup(ability.description),
            ability.tooltip_resolved,
            _values_block(_ability_value_lines(ability.values)),
        ]
    )
    return BuiltDocument(
        doc_key=f"ability:{ability.champion_slug}:{ability.slot}",
        collection=COLLECTION_ABILITIES,
        entity_column="ability_id",
        entity_id=ability.id,
        title=title,
        source=f"Data Dragon and Community Dragon ability {champion.name} {ability.slot}",
        content=content,
    )


def build_item_document(item: Item, tags: Sequence[str], map_ids: Sequence[int]) -> BuiltDocument:
    """Build the document for one purchasable item.

    Args:
        item: Stored Item with its values loaded.
        tags: The item's stored tags.
        map_ids: The map ids the item is available on, ascending.

    Returns:
        A BuiltDocument in the equipment collection, titled with the item name
        and the game modes it can be bought in.

    Raises:
        KeyError: If the item is available on a map id MAP_NAMES does not name,
            which means Riot shipped a new mode and the constant is stale.
    """
    modes = LIST_SEPARATOR.join(MAP_NAMES[map_id] for map_id in map_ids)
    title = f"{item.name} ({modes})"
    cost = [f"Cost: {item.gold_total} gold ({item.gold_base} to combine)"]
    if tags:
        cost.append(f"Tags: {LIST_SEPARATOR.join(tags)}")
    content = _join_blocks(
        [
            title,
            item.description_text,
            LINE_SEPARATOR.join(cost),
            _values_block(_item_value_lines(item.values)),
        ]
    )
    return BuiltDocument(
        doc_key=f"item:{item.ddragon_id}",
        collection=COLLECTION_EQUIPMENT,
        entity_column="item_id",
        entity_id=item.ddragon_id,
        title=title,
        source=f"Data Dragon item {item.ddragon_id}",
        content=content,
    )


def build_rune_document(rune: Rune) -> BuiltDocument:
    """Build the document for one rune.

    Args:
        rune: Stored Rune with its path loaded.

    Returns:
        A BuiltDocument in the equipment collection. Row zero of every path
        holds the keystones, so the title names the rune a keystone there and a
        plain rune elsewhere. The long description is dropped when it repeats
        the short one verbatim, which adds no fact and would only crowd the
        chunk.
    """
    kind = "keystone" if rune.row_index == KEYSTONE_ROW_INDEX else "rune"
    title = f"{rune.name} ({rune.path.name} {kind})"
    long_desc = None if rune.long_desc_text == rune.short_desc_text else rune.long_desc_text
    content = _join_blocks([title, rune.short_desc_text, long_desc])
    return BuiltDocument(
        doc_key=f"rune:{rune.id}",
        collection=COLLECTION_EQUIPMENT,
        entity_column="rune_id",
        entity_id=rune.id,
        title=title,
        source=f"Data Dragon rune {rune.key}",
        content=content,
    )


def build_summoner_spell_document(spell: SummonerSpell) -> BuiltDocument:
    """Build the document for one summoner spell.

    Args:
        spell: Stored SummonerSpell.

    Returns:
        A BuiltDocument in the equipment collection. The cooldown and unlock
        level lines are omitted when the source publishes neither.
    """
    title = f"{spell.name} (summoner spell)"
    facts: list[str] = []
    if spell.cooldown is not None:
        facts.append(f"Cooldown: {_format_number(spell.cooldown)} seconds")
    if spell.summoner_level is not None:
        facts.append(f"Unlocked at summoner level {spell.summoner_level}")
    content = _join_blocks([title, spell.description_text, LINE_SEPARATOR.join(facts)])
    return BuiltDocument(
        doc_key=f"summoner_spell:{spell.id}",
        collection=COLLECTION_EQUIPMENT,
        entity_column="summoner_spell_id",
        entity_id=spell.id,
        title=title,
        source=f"Data Dragon summoner spell {spell.id}",
        content=content,
    )


def build_champion_document(champion: Champion) -> BuiltDocument:
    """Build the biography document for one champion.

    Args:
        champion: Stored Champion with its faction and roles loaded.

    Returns:
        A BuiltDocument in the lore collection carrying the short biography
        ahead of the full one. Roles are omitted for the lore-only characters,
        which have no Data Dragon entry and so no class tags.
    """
    title = f"{champion.name}, {champion.title}"
    header = [title, f"Faction: {champion.faction.name}"]
    if champion.roles:
        header.append(f"Roles: {LIST_SEPARATOR.join(sorted(role.name for role in champion.roles))}")
    if champion.release_date is not None:
        header.append(f"Released: {champion.release_date.date().isoformat()}")
    content = _join_blocks(
        [LINE_SEPARATOR.join(header), champion.bio_short_text, champion.bio_full_text]
    )
    return BuiltDocument(
        doc_key=f"champion:{champion.slug}",
        collection=COLLECTION_LORE,
        entity_column="champion_slug",
        entity_id=champion.slug,
        title=title,
        source=f"Riot Universe champion {champion.slug}",
        content=content,
    )


def build_story_document(story: Story) -> BuiltDocument:
    """Build the document for one long-form lore story.

    Args:
        story: Stored Story.

    Returns:
        A BuiltDocument in the lore collection. The content keeps the triple
        newlines the story loader put between subsections, so the chunker can
        still see a scene boundary no other separator can forge.
    """
    header = [story.title]
    if story.author:
        header.append(f"Author: {story.author}")
    content = _join_blocks([LINE_SEPARATOR.join(header), story.content_text])
    return BuiltDocument(
        doc_key=f"story:{story.slug}",
        collection=COLLECTION_LORE,
        entity_column="story_slug",
        entity_id=story.slug,
        title=story.title,
        source=f"Riot Universe story {story.slug}",
        content=content,
    )


def build_faction_document(faction: Faction) -> BuiltDocument:
    """Build the document for one lore faction.

    Args:
        faction: Stored Faction.

    Returns:
        A BuiltDocument in the lore collection. The synthetic "unaffiliated"
        faction publishes no overview, so its document is its title line alone;
        it still gets one, because every faction a champion points at must be
        retrievable.
    """
    title = f"Faction: {faction.name}"
    content = _join_blocks([title, faction.overview_text])
    return BuiltDocument(
        doc_key=f"faction:{faction.slug}",
        collection=COLLECTION_LORE,
        entity_column="faction_slug",
        entity_id=faction.slug,
        title=title,
        source=f"Riot Universe faction {faction.slug}",
        content=content,
    )


# ---------- entity selection ----------


def _grouped(session: Session, table: object, value_column: object) -> dict[str, list]:
    """Read one item association table into a per-item list.

    Args:
        session: Open Session the rows are read through.
        table: The association table to read.
        value_column: The column holding the associated value.

    Returns:
        Mapping of item id to that item's values, ascending, so the rendered
        order does not depend on the physical row order.
    """
    grouped: dict[str, list] = {}
    for item_id, value in session.execute(
        select(table.c.item_id, value_column).order_by(table.c.item_id, value_column)
    ):
        grouped.setdefault(item_id, []).append(value)
    return grouped


def build_documents(session: Session) -> list[BuiltDocument]:
    """Build one document for every entity that carries retrievable prose.

    Args:
        session: Open Session the entities are read through. The entity and
            value loaders must have run first.

    Returns:
        One BuiltDocument per ability, purchasable item, rune, summoner spell,
        champion, story and faction, in that order. Items are filtered to the
        ones a player can actually buy somewhere: purchasable, listed in the
        shop, and available on at least one map. The rows the filter passes over
        stay in the items table as graph nodes with their components intact,
        because this is a document-build filter and not a deletion.

    Raises:
        KeyError: If an item is available on a map id MAP_NAMES does not name.
    """
    abilities = (
        session.execute(
            select(Ability)
            .options(selectinload(Ability.champion), selectinload(Ability.values))
            .order_by(Ability.champion_slug, Ability.slot)
        )
        .scalars()
        .all()
    )
    items = (
        session.execute(
            select(Item)
            .where(
                Item.purchasable.is_(True),
                Item.in_store.is_(True),
                exists().where(item_map.c.item_id == Item.ddragon_id),
            )
            .options(selectinload(Item.values))
            .order_by(Item.ddragon_id)
        )
        .scalars()
        .all()
    )
    tags_by_item = _grouped(session, item_tag, item_tag.c.tag)
    maps_by_item = _grouped(session, item_map, item_map.c.map_id)
    runes = (
        session.execute(select(Rune).options(selectinload(Rune.path)).order_by(Rune.id))
        .scalars()
        .all()
    )
    spells = session.execute(select(SummonerSpell).order_by(SummonerSpell.id)).scalars().all()
    champions = (
        session.execute(
            select(Champion)
            .options(selectinload(Champion.faction), selectinload(Champion.roles))
            .order_by(Champion.slug)
        )
        .scalars()
        .all()
    )
    stories = session.execute(select(Story).order_by(Story.slug)).scalars().all()
    factions = session.execute(select(Faction).order_by(Faction.slug)).scalars().all()

    documents = [
        *(build_ability_document(ability) for ability in abilities),
        *(
            build_item_document(
                item, tags_by_item.get(item.ddragon_id, []), maps_by_item[item.ddragon_id]
            )
            for item in items
        ),
        *(build_rune_document(rune) for rune in runes),
        *(build_summoner_spell_document(spell) for spell in spells),
        *(build_champion_document(champion) for champion in champions),
        *(build_story_document(story) for story in stories),
        *(build_faction_document(faction) for faction in factions),
    ]
    logger.info("built %d documents", len(documents))
    return documents


# ---------- chunking ----------


def build_splitter() -> RecursiveCharacterTextSplitter:
    """Build the splitter every document is chunked with.

    Returns:
        A RecursiveCharacterTextSplitter that prefers scene boundaries. The
        triple newline leads the separator list because the markup cleaner
        collapses any run of three or more newlines inside a block down to two,
        so a triple newline in stored content can only be a subsection boundary
        the story loader put there.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )


def chunk_document(content: str, splitter: RecursiveCharacterTextSplitter) -> list[str]:
    """Split one document into embeddable chunks, each naming what it describes.

    Args:
        content: Full document text, whose first line is its title.
        splitter: The splitter to cut it with.

    Returns:
        One string per chunk. A document that fits in one chunk is returned
        unchanged, since it already opens with its title line. A document that
        splits gets that title line prefixed onto every chunk that does not
        already start with it, so a tail chunk of bare numbers still names its
        champion and ability once the retriever hands it to the model on its
        own.
    """
    pieces = splitter.split_text(content)
    if len(pieces) <= 1:
        return pieces
    title = content.split(LINE_SEPARATOR, 1)[0]
    return [
        piece if piece.startswith(title) else f"{title}{LINE_SEPARATOR}{piece}" for piece in pieces
    ]


# ---------- persistence ----------


def _document_row(document: BuiltDocument, indexed_at: datetime) -> dict[str, object]:
    """Render one built document as a documents insert row.

    Args:
        document: The built document.
        indexed_at: Timestamp stamped on rows this run inserts or changes.

    Returns:
        A mapping keyed by column name with all seven entity foreign keys
        present, exactly one of them non-null, which is what the table's
        check constraint requires.
    """
    row: dict[str, object] = dict.fromkeys(ENTITY_COLUMNS)
    row[document.entity_column] = document.entity_id
    row.update(
        doc_key=document.doc_key,
        collection=document.collection,
        title=document.title,
        source=document.source,
        content=document.content,
        indexed_at=indexed_at,
    )
    return row


def _upsert_documents(session: Session, documents: Sequence[BuiltDocument]) -> list[int]:
    """Insert or update the documents, skipping the ones whose content is unchanged.

    Args:
        session: Open Session the statements are executed on. Changes are
            flushed but never committed.
        documents: The built documents, keyed on entity identity.

    Returns:
        The ids of exactly the documents this run inserted or changed. The
        conflict clause updates only when the stored content differs, so an
        unchanged document is not rewritten and does not appear here, which is
        what lets the caller re-embed nothing on a re-run over a static corpus.
    """
    indexed_at = datetime.now(UTC).replace(tzinfo=None)
    changed: list[int] = []
    for start in range(0, len(documents), UPSERT_BATCH_SIZE):
        batch = documents[start : start + UPSERT_BATCH_SIZE]
        statement = postgres_insert(Document).values(
            [_document_row(document, indexed_at) for document in batch]
        )
        statement = statement.on_conflict_do_update(
            index_elements=[Document.doc_key],
            set_={
                "content": statement.excluded.content,
                "title": statement.excluded.title,
                "source": statement.excluded.source,
                "indexed_at": statement.excluded.indexed_at,
            },
            where=Document.content.is_distinct_from(statement.excluded.content),
        ).returning(Document.id)
        changed.extend(session.execute(statement).scalars())
    session.flush()
    return changed


def _write_chunks(session: Session, document_ids: Sequence[int], settings: Settings) -> int:
    """Re-chunk and re-embed the given documents, replacing their stored chunks.

    Args:
        session: Open Session the statements are executed on. Changes are
            flushed but never committed.
        document_ids: Ids of the documents to rebuild chunks for.
        settings: Application settings providing embedding_model_name.

    Returns:
        The number of chunk rows written. Embedding runs in batches through one
        cached model rather than once per chunk, a few thousand chunks taking
        minutes either way but two orders of magnitude more calls otherwise.
    """
    if not document_ids:
        return 0
    splitter = build_splitter()
    rows: list[dict[str, object]] = []
    for document_id, content in session.execute(
        select(Document.id, Document.content)
        .where(Document.id.in_(document_ids))
        .order_by(Document.id)
    ):
        rows.extend(
            {"document_id": document_id, "chunk_index": index, "content": text}
            for index, text in enumerate(chunk_document(content, splitter))
        )

    embeddings = get_embeddings(settings.embedding_model_name)
    for start in range(0, len(rows), EMBED_BATCH_SIZE):
        batch = rows[start : start + EMBED_BATCH_SIZE]
        vectors = embeddings.embed_documents([str(row["content"]) for row in batch])
        for row, vector in zip(batch, vectors, strict=True):
            row["embedding"] = vector
        logger.info("embedded %d of %d chunks", start + len(batch), len(rows))

    session.execute(insert(Chunk), rows)
    session.flush()
    return len(rows)


# ---------- orchestration ----------


@dataclass(frozen=True)
class DocumentLoadStats:
    """Counts describing one document and chunk load run.

    Args:
        documents_built: Documents assembled from the stored entities.
        documents_changed: Documents this run inserted or whose content it
            rewrote; only these are re-chunked and re-embedded.
        chunks_written: Chunk rows embedded and inserted this run.
        chunks_skipped: Chunk rows left untouched because the document they
            belong to was unchanged, so a re-run over a static corpus embeds
            nothing at all.
    """

    documents_built: int
    documents_changed: int
    chunks_written: int
    chunks_skipped: int


def load_documents(session: Session, settings: Settings) -> DocumentLoadStats:
    """Build every document from the stored entities and embed the ones that changed.

    Args:
        session: Open Session the rows are written through. Changes are flushed
            but never committed, so the caller decides whether the run is kept.
            The entity and value loaders must have run first, because every
            document hangs off an entity row by foreign key.
        settings: Application settings providing embedding_model_name.

    Returns:
        DocumentLoadStats counting what was built, what changed and how many
        chunks were written against how many were left alone.

    Raises:
        KeyError: If an item is available on a map id MAP_NAMES does not name.
        sqlalchemy.exc.SQLAlchemyError: If any row violates the schema.
    """
    documents = build_documents(session)
    changed = _upsert_documents(session, documents)
    stored_chunks = session.execute(select(func.count()).select_from(Chunk)).scalar_one()
    removed = 0
    if changed:
        removed = session.execute(delete(Chunk).where(Chunk.document_id.in_(changed))).rowcount
    written = _write_chunks(session, changed, settings)

    stats = DocumentLoadStats(
        documents_built=len(documents),
        documents_changed=len(changed),
        chunks_written=written,
        chunks_skipped=stored_chunks - removed,
    )
    logger.info("loaded documents: %s", stats)
    return stats
