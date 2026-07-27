import asyncio
import logging
from dataclasses import dataclass

from lolrag.config import Settings
from lolrag.fetch import cdragon, cdragon_bin, ddragon, universe
from lolrag.fetch.client import FetchClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusCacheStats:
    """Counts describing one cache-warming run.

    Args:
        ddragon_champions: Number of Data Dragon champion detail files fetched.
        cdragon_champions: Number of Community Dragon champion records fetched.
        cdragon_champion_bins: Number of Community Dragon raw champion bin
            files fetched.
        cdragon_item_bin: Number of Community Dragon raw item bin files
            fetched, 1 once the single file is present.
        universe_champions: Number of Universe champion payloads fetched.
        universe_factions: Number of Universe faction payloads fetched.
        universe_stories: Number of distinct Universe stories fetched.
        cache_hits: Requests served from the on-disk cache.
        cache_misses: Requests that went to the network.
    """

    ddragon_champions: int
    cdragon_champions: int
    cdragon_champion_bins: int
    cdragon_item_bin: int
    universe_champions: int
    universe_factions: int
    universe_stories: int
    cache_hits: int
    cache_misses: int


async def warm_cache(settings: Settings, *, refresh: bool = False) -> CorpusCacheStats:
    """Fetch the full corpus into the on-disk cache.

    Args:
        settings: Application settings providing every ddragon_*, cdragon_*,
            universe_* and http_* value plus cache_dir.
        refresh: If True, refetch and overwrite every cache entry instead of
            reusing existing ones.

    Returns:
        CorpusCacheStats describing how many documents of each kind were
        fetched and how the requests split between cache and network.

    Raises:
        httpx.HTTPStatusError: If any request fails after its retries.
    """
    async with FetchClient(settings, refresh=refresh) as client:
        champion_list, _items, _runes, _spells = await asyncio.gather(
            ddragon.fetch_champion_list(client, settings),
            ddragon.fetch_items(client, settings),
            ddragon.fetch_runes(client, settings),
            ddragon.fetch_summoner_spells(client, settings),
        )
        logger.info("fetched Data Dragon bulk files")

        champion_ids = list(champion_list["data"].keys())
        champion_keys = [int(entry["key"]) for entry in champion_list["data"].values()]
        await ddragon.fetch_all_champion_details(client, settings, champion_ids)
        logger.info("fetched %d Data Dragon champion details", len(champion_ids))

        await cdragon.fetch_all_champions(client, settings, champion_keys)
        logger.info("fetched %d Community Dragon champions", len(champion_keys))

        await cdragon_bin.fetch_all_champion_bins(client, settings, champion_ids)
        await cdragon_bin.fetch_item_bin(client, settings)
        logger.info("fetched %d Community Dragon champion bins and the item bin", len(champion_ids))

        search_index = await universe.fetch_search_index(client, settings)
        champion_slugs = [entry["slug"] for entry in search_index["champions"]]
        faction_slugs = [entry["slug"] for entry in search_index["factions"]]
        logger.info(
            "Universe index lists %d champions and %d factions",
            len(champion_slugs),
            len(faction_slugs),
        )

        universe_champions = await universe.fetch_all_champions(client, settings, champion_slugs)
        await universe.fetch_all_factions(client, settings, faction_slugs)
        logger.info("fetched Universe champions and factions")

        story_slugs: list[str] = []
        seen: set[str] = set()
        for payload in universe_champions.values():
            for slug in universe.extract_story_slugs(payload):
                if slug not in seen:
                    seen.add(slug)
                    story_slugs.append(slug)

        await universe.fetch_all_stories(client, settings, story_slugs)
        logger.info("fetched %d Universe stories", len(story_slugs))

        return CorpusCacheStats(
            ddragon_champions=len(champion_ids),
            cdragon_champions=len(champion_keys),
            cdragon_champion_bins=len(champion_ids),
            cdragon_item_bin=1,
            universe_champions=len(champion_slugs),
            universe_factions=len(faction_slugs),
            universe_stories=len(story_slugs),
            cache_hits=client.hits,
            cache_misses=client.misses,
        )
