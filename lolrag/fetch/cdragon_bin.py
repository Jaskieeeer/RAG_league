import asyncio
from collections.abc import Sequence

from lolrag.config import Settings
from lolrag.fetch.client import FetchClient


async def fetch_champion_bin(client: FetchClient, settings: Settings, ddragon_id: str) -> dict:
    """Fetch one champion's raw game-data bin file from Community Dragon.

    Args:
        client: Open FetchClient used for the request.
        settings: Application settings providing cdragon_base_url.
        ddragon_id: Data Dragon champion id, e.g. "MonkeyKing". Its lowercase
            form is the path slug, which appears in both the directory and the
            filename position of the URL.

    Returns:
        Parsed bin record carrying the numeric ability values that the Community
        Dragon plugin endpoint zeroes out.
    """
    slug = ddragon_id.lower()
    url = f"{settings.cdragon_base_url}/latest/game/data/characters/{slug}/{slug}.bin.json"
    return await client.get_json(url, "cdragon", "latest", "characters", f"{slug}.bin.json")


async def fetch_all_champion_bins(
    client: FetchClient, settings: Settings, ddragon_ids: Sequence[str]
) -> dict[str, dict]:
    """Fetch raw bin files for every given champion id concurrently.

    Args:
        client: Open FetchClient used for the requests.
        settings: Application settings providing cdragon_base_url.
        ddragon_ids: Data Dragon champion ids to fetch.

    Returns:
        Mapping of the original Data Dragon champion id, not the lowercase
        slug, to that champion's parsed bin record.
    """
    payloads = await asyncio.gather(
        *(fetch_champion_bin(client, settings, ddragon_id) for ddragon_id in ddragon_ids)
    )
    return dict(zip(ddragon_ids, payloads, strict=True))


async def fetch_item_bin(client: FetchClient, settings: Settings) -> dict:
    """Fetch the single raw game-data bin file describing every item.

    Args:
        client: Open FetchClient used for the request.
        settings: Application settings providing cdragon_base_url.

    Returns:
        Parsed bin record carrying the numeric item values for all items in one
        payload of roughly 14 MB.
    """
    url = f"{settings.cdragon_base_url}/latest/game/items.cdtb.bin.json"
    return await client.get_json(url, "cdragon", "latest", "items.cdtb.bin.json")
