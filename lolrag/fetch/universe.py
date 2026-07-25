import asyncio
from collections.abc import Sequence

from lolrag.config import Settings
from lolrag.fetch.client import FetchClient


async def fetch_search_index(client: FetchClient, settings: Settings) -> dict:
    """Fetch the Riot Universe search index.

    Args:
        client: Open FetchClient used for the request.
        settings: Application settings providing universe_base_url.

    Returns:
        Parsed index containing "champions" and "factions" arrays, each entry
        carrying a "slug". This is the only usable hub endpoint; the
        champions, factions and stories index endpoints return 403.
    """
    url = f"{settings.universe_base_url}/search/index.json"
    return await client.get_json(url, "universe", "search", "index.json")


async def fetch_champion(client: FetchClient, settings: Settings, slug: str) -> dict:
    """Fetch one champion's Riot Universe page payload.

    Args:
        client: Open FetchClient used for the request.
        settings: Application settings providing universe_base_url.
        slug: Universe champion slug, e.g. "aatrox".

    Returns:
        Parsed champion payload including biography and a "modules" list that
        references the champion's stories.
    """
    url = f"{settings.universe_base_url}/champions/{slug}/index.json"
    return await client.get_json(url, "universe", "champions", f"{slug}.json")


async def fetch_faction(client: FetchClient, settings: Settings, slug: str) -> dict:
    """Fetch one faction's Riot Universe page payload.

    Args:
        client: Open FetchClient used for the request.
        settings: Application settings providing universe_base_url.
        slug: Universe faction slug, e.g. "noxus".

    Returns:
        Parsed faction payload including the faction's long-form description.
    """
    url = f"{settings.universe_base_url}/factions/{slug}/index.json"
    return await client.get_json(url, "universe", "factions", f"{slug}.json")


async def fetch_story(client: FetchClient, settings: Settings, slug: str) -> dict:
    """Fetch one story's Riot Universe payload.

    Args:
        client: Open FetchClient used for the request.
        settings: Application settings providing universe_base_url.
        slug: Universe story slug, e.g. "the-blade-of-the-ruined-king".

    Returns:
        Parsed story payload including the full long-form text.
    """
    url = f"{settings.universe_base_url}/story/{slug}/index.json"
    return await client.get_json(url, "universe", "story", f"{slug}.json")


async def fetch_all_champions(
    client: FetchClient, settings: Settings, slugs: Sequence[str]
) -> dict[str, dict]:
    """Fetch Universe champion payloads for every given slug concurrently.

    Args:
        client: Open FetchClient used for the requests.
        settings: Application settings providing universe_base_url.
        slugs: Universe champion slugs to fetch.

    Returns:
        Mapping of slug to that champion's parsed payload.
    """
    payloads = await asyncio.gather(*(fetch_champion(client, settings, slug) for slug in slugs))
    return dict(zip(slugs, payloads, strict=True))


async def fetch_all_factions(
    client: FetchClient, settings: Settings, slugs: Sequence[str]
) -> dict[str, dict]:
    """Fetch Universe faction payloads for every given slug concurrently.

    Args:
        client: Open FetchClient used for the requests.
        settings: Application settings providing universe_base_url.
        slugs: Universe faction slugs to fetch.

    Returns:
        Mapping of slug to that faction's parsed payload.
    """
    payloads = await asyncio.gather(*(fetch_faction(client, settings, slug) for slug in slugs))
    return dict(zip(slugs, payloads, strict=True))


async def fetch_all_stories(
    client: FetchClient, settings: Settings, slugs: Sequence[str]
) -> dict[str, dict]:
    """Fetch Universe story payloads for every given slug concurrently.

    Args:
        client: Open FetchClient used for the requests.
        settings: Application settings providing universe_base_url.
        slugs: Universe story slugs to fetch.

    Returns:
        Mapping of slug to that story's parsed payload.
    """
    payloads = await asyncio.gather(*(fetch_story(client, settings, slug) for slug in slugs))
    return dict(zip(slugs, payloads, strict=True))


def extract_story_slugs(champion_payload: dict) -> list[str]:
    """Collect the story slugs referenced by a Universe champion payload.

    Args:
        champion_payload: Parsed Universe champion payload containing a
            "modules" list; modules that reference a story carry a
            "story-slug" key.

    Returns:
        Story slugs in the order they appear, with duplicates removed. Empty
        if the payload has no modules or no module references a story.
    """
    slugs: list[str] = []
    seen: set[str] = set()
    for module in champion_payload.get("modules", []):
        slug = module.get("story-slug")
        if slug and slug not in seen:
            seen.add(slug)
            slugs.append(slug)
    return slugs
