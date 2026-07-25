import asyncio
from collections.abc import Sequence

from lolrag.config import Settings
from lolrag.fetch.client import FetchClient


async def fetch_champion(client: FetchClient, settings: Settings, champion_key: int) -> dict:
    """Fetch one champion's Community Dragon record.

    Args:
        client: Open FetchClient used for the request.
        settings: Application settings providing cdragon_base_url.
        champion_key: Numeric champion key, e.g. 266. Data Dragon exposes this
            as a string at champion.json data[<Name>]["key"] and it must be
            coerced to int before being passed here.

    Returns:
        Parsed champion record including tactical and playstyle metadata and
        full ability tooltips.
    """
    url = (
        f"{settings.cdragon_base_url}/latest/plugins/rcp-be-lol-game-data"
        f"/global/default/v1/champions/{champion_key}.json"
    )
    return await client.get_json(url, "cdragon", "latest", "champions", f"{champion_key}.json")


async def fetch_all_champions(
    client: FetchClient, settings: Settings, champion_keys: Sequence[int]
) -> dict[int, dict]:
    """Fetch Community Dragon records for every given champion key concurrently.

    Args:
        client: Open FetchClient used for the requests.
        settings: Application settings providing cdragon_base_url.
        champion_keys: Numeric champion keys to fetch.

    Returns:
        Mapping of numeric champion key to that champion's parsed record.
    """
    payloads = await asyncio.gather(
        *(fetch_champion(client, settings, champion_key) for champion_key in champion_keys)
    )
    return dict(zip(champion_keys, payloads, strict=True))
