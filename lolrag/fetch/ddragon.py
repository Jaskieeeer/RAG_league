import asyncio
from collections.abc import Sequence

from lolrag.config import Settings
from lolrag.fetch.client import FetchClient


async def fetch_versions(client: FetchClient, settings: Settings) -> list[str]:
    """Fetch the Data Dragon version manifest.

    Args:
        client: Open FetchClient used for the request.
        settings: Application settings providing ddragon_base_url.

    Returns:
        All published Data Dragon versions, newest first, e.g. ["16.14.1", ...].
    """
    url = f"{settings.ddragon_base_url}/api/versions.json"
    return await client.get_json(url, "ddragon", "api", "versions.json")


async def fetch_champion_list(client: FetchClient, settings: Settings) -> dict:
    """Fetch the Data Dragon all-champions summary file.

    Args:
        client: Open FetchClient used for the request.
        settings: Application settings providing ddragon_base_url,
            ddragon_version, ddragon_locale.

    Returns:
        Parsed champion.json body, whose "data" key maps champion id to summary.
    """
    url = (
        f"{settings.ddragon_base_url}/cdn/{settings.ddragon_version}"
        f"/data/{settings.ddragon_locale}/champion.json"
    )
    return await client.get_json(
        url, "ddragon", settings.ddragon_version, settings.ddragon_locale, "champion.json"
    )


async def fetch_champion_detail(client: FetchClient, settings: Settings, champion_id: str) -> dict:
    """Fetch one champion's Data Dragon detail file.

    Args:
        client: Open FetchClient used for the request.
        settings: Application settings providing ddragon_base_url,
            ddragon_version, ddragon_locale.
        champion_id: Data Dragon champion id, e.g. "Aatrox".

    Returns:
        Parsed detail body, whose "data" key maps champion id to the full
        record including "lore", "passive" and "spells".
    """
    url = (
        f"{settings.ddragon_base_url}/cdn/{settings.ddragon_version}"
        f"/data/{settings.ddragon_locale}/champion/{champion_id}.json"
    )
    return await client.get_json(
        url,
        "ddragon",
        settings.ddragon_version,
        settings.ddragon_locale,
        "champion",
        f"{champion_id}.json",
    )


async def fetch_all_champion_details(
    client: FetchClient, settings: Settings, champion_ids: Sequence[str]
) -> dict[str, dict]:
    """Fetch detail files for every given champion id concurrently.

    Args:
        client: Open FetchClient used for the requests.
        settings: Application settings providing ddragon_base_url,
            ddragon_version, ddragon_locale.
        champion_ids: Data Dragon champion ids to fetch.

    Returns:
        Mapping of champion id to that champion's parsed detail body.
    """
    payloads = await asyncio.gather(
        *(fetch_champion_detail(client, settings, champion_id) for champion_id in champion_ids)
    )
    return dict(zip(champion_ids, payloads, strict=True))


async def fetch_items(client: FetchClient, settings: Settings) -> dict:
    """Fetch the Data Dragon item file.

    Args:
        client: Open FetchClient used for the request.
        settings: Application settings providing ddragon_base_url,
            ddragon_version, ddragon_locale.

    Returns:
        Parsed item.json body, whose "data" key maps item id to item record.
    """
    url = (
        f"{settings.ddragon_base_url}/cdn/{settings.ddragon_version}"
        f"/data/{settings.ddragon_locale}/item.json"
    )
    return await client.get_json(
        url, "ddragon", settings.ddragon_version, settings.ddragon_locale, "item.json"
    )


async def fetch_runes(client: FetchClient, settings: Settings) -> list[dict]:
    """Fetch the Data Dragon reforged runes file.

    Args:
        client: Open FetchClient used for the request.
        settings: Application settings providing ddragon_base_url,
            ddragon_version, ddragon_locale.

    Returns:
        One record per rune path; this endpoint returns a JSON list, not an object.
    """
    url = (
        f"{settings.ddragon_base_url}/cdn/{settings.ddragon_version}"
        f"/data/{settings.ddragon_locale}/runesReforged.json"
    )
    return await client.get_json(
        url, "ddragon", settings.ddragon_version, settings.ddragon_locale, "runesReforged.json"
    )


async def fetch_summoner_spells(client: FetchClient, settings: Settings) -> dict:
    """Fetch the Data Dragon summoner spell file.

    Args:
        client: Open FetchClient used for the request.
        settings: Application settings providing ddragon_base_url,
            ddragon_version, ddragon_locale.

    Returns:
        Parsed summoner.json body, whose "data" key maps spell id to spell record.
    """
    url = (
        f"{settings.ddragon_base_url}/cdn/{settings.ddragon_version}"
        f"/data/{settings.ddragon_locale}/summoner.json"
    )
    return await client.get_json(
        url, "ddragon", settings.ddragon_version, settings.ddragon_locale, "summoner.json"
    )
