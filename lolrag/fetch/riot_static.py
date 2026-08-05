from lolrag.config import Settings
from lolrag.fetch.client import FetchClient


async def fetch_game_modes(client: FetchClient, settings: Settings) -> list[dict]:
    """Fetch Riot's canonical game-mode enum list.

    Args:
        client: Open FetchClient used for the request.
        settings: Application settings providing riot_static_base_url.

    Returns:
        One record per documented game mode, each carrying a "gameMode" enum and
        a "description"; this endpoint returns a JSON list, not an object. It is
        not versioned by patch, so it is cached under the same static path on
        every run.
    """
    url = f"{settings.riot_static_base_url}/docs/lol/gameModes.json"
    return await client.get_json(url, "riot_static", "gameModes.json")
