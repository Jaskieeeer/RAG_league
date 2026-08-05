import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from lolrag.config import Settings
from lolrag.fetch.client import FetchClient
from lolrag.fetch.corpus import CorpusCacheStats, warm_cache
from lolrag.ingest.documents import DocumentLoadStats, load_documents
from lolrag.ingest.loaders import LoadStats, load_all, load_game_mode_names
from lolrag.ingest.values import ValueLoadStats, load_values

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestReport:
    """Everything one ingest run measured, one stats object per stage.

    Args:
        cache: Counts describing the cache-warming stage.
        entities: Counts describing the entity and association load stage.
        values: Counts describing the numeric value load stage.
        documents: Counts describing the document and chunk load stage.
    """

    cache: CorpusCacheStats
    entities: LoadStats
    values: ValueLoadStats
    documents: DocumentLoadStats


async def run_ingest(
    session: Session,
    client: FetchClient,
    settings: Settings,
    *,
    refresh: bool = False,
) -> IngestReport:
    """Warm the corpus cache and load every entity, association and value row.

    Args:
        session: Open Session the rows are merged into. Changes are flushed but
            never committed, so the caller decides whether the run is kept.
        client: Open FetchClient the loaders read the corpus through. The
            cache-warming stage opens its own client internally, so this one
            serves the two load stages alone.
        settings: Application settings providing every ddragon_*, cdragon_*,
            universe_*, riot_static_* and http_* value plus cache_dir.
        refresh: If True, refetch and overwrite every cache entry instead of
            reusing existing ones.

    Returns:
        IngestReport composing the stats of all four stages. The document stage
        runs last because it reads the entity and value rows the earlier stages
        wrote, and it embeds only the documents whose content actually changed.
        The game-mode names it renders summoner spell modes with are fetched
        here rather than stored, no entity in the schema referring to a mode.

    Raises:
        httpx.HTTPStatusError: If any request fails after its retries.
        sqlalchemy.exc.SQLAlchemyError: If any row violates the schema.
        ValueError: If a Data Dragon spell cannot be joined to the bin.
        KeyError: If an item is available on a map id MAP_NAMES does not name.
    """
    cache = await warm_cache(settings, refresh=refresh)
    entities = await load_all(session, client, settings)
    values = await load_values(session, client, settings)
    mode_names = await load_game_mode_names(client, settings)
    documents = load_documents(session, settings, mode_names)
    report = IngestReport(cache=cache, entities=entities, values=values, documents=documents)
    logger.info("ingest complete: %s", report)
    return report
