import httpx
from langchain_core.documents import Document

from lolrag.config import Settings

_SPELL_KEYS = ["Q", "W", "E", "R"]


def fetch_champion_json(version: str, locale: str, base_url: str) -> dict:
    """Download the Data Dragon all-champions summary file.

    Args:
        version: Pinned Data Dragon patch version, e.g. "16.14.1".
        locale: Data Dragon locale code, e.g. "en_US".
        base_url: Data Dragon CDN root URL.

    Returns:
        Parsed JSON body of champion.json as a dict.

    Raises:
        httpx.HTTPStatusError: If the request does not return a 2xx status.
    """
    url = f"{base_url}/cdn/{version}/data/{locale}/champion.json"
    response = httpx.get(url)
    response.raise_for_status()
    return response.json()


def fetch_champion_detail(champion_id: str, version: str, locale: str, base_url: str) -> dict:
    """Download the Data Dragon per-champion detail file.

    Args:
        champion_id: Data Dragon champion id, e.g. "Aatrox".
        version: Pinned Data Dragon patch version.
        locale: Data Dragon locale code.
        base_url: Data Dragon CDN root URL.

    Returns:
        The champion's detail dict, unwrapped from the response's "data" key,
        containing "lore", "passive", and "spells".

    Raises:
        httpx.HTTPStatusError: If the request does not return a 2xx status.
    """
    url = f"{base_url}/cdn/{version}/data/{locale}/champion/{champion_id}.json"
    response = httpx.get(url)
    response.raise_for_status()
    return response.json()["data"][champion_id]


def fetch_all_champion_details(
    champion_ids: list[str], version: str, locale: str, base_url: str
) -> dict[str, dict]:
    """Download per-champion detail files for every given champion id.

    Args:
        champion_ids: Data Dragon champion ids to fetch details for.
        version: Pinned Data Dragon patch version.
        locale: Data Dragon locale code.
        base_url: Data Dragon CDN root URL.

    Returns:
        Mapping of champion id to that champion's detail dict.
    """
    return {
        champion_id: fetch_champion_detail(champion_id, version, locale, base_url)
        for champion_id in champion_ids
    }


def _format_abilities(detail: dict) -> str:
    """Build the natural-language abilities section for one champion.

    Args:
        detail: Per-champion detail dict containing "passive" and "spells".

    Returns:
        Newline-separated passive and spell name/description lines, spells
        labeled Q, W, E, R in Data Dragon's fixed ordering.
    """
    passive = detail["passive"]
    lines = [f"Passive - {passive['name']}: {passive['description']}"]
    for key, spell in zip(_SPELL_KEYS, detail["spells"], strict=True):
        lines.append(f"{key} - {spell['name']}: {spell['description']}")
    return "\n".join(lines)


def _format_page_content(champion: dict, tags: str, lore: str, abilities: str) -> str:
    """Build the natural-language page content for one champion.

    Args:
        champion: Single champion entry from champion.json["data"].
        tags: Comma-separated champion class tags.
        lore: Full champion lore text from the per-champion detail endpoint.
        abilities: Formatted passive and spell text from _format_abilities.

    Returns:
        Paragraph combining name, title, tags, partype, full lore, and abilities.
    """
    return (
        f"{champion['name']}, {champion['title']}, is a {tags} champion "
        f"with resource type {champion['partype']}.\n\n"
        f"{lore}\n\n"
        f"{abilities}"
    )


def champion_json_to_documents(
    champion_json: dict, champion_details: dict[str, dict], version: str
) -> list[Document]:
    """Transform Data Dragon summary and detail payloads into one Document per champion.

    Args:
        champion_json: Parsed champion.json body (summary), containing "data" and "version".
        champion_details: Mapping of champion id to that champion's detail dict, as
            returned by fetch_all_champion_details, providing "lore", "passive", "spells".
        version: Data Dragon version expected in champion_json["version"];
            stamped into each Document's metadata.

    Returns:
        One langchain_core.documents.Document per champion, with full lore and
        abilities folded into page_content.

    Raises:
        ValueError: If champion_json["version"] does not match version.
        KeyError: If a champion in champion_json has no matching entry in champion_details.
    """
    if champion_json["version"] != version:
        raise ValueError(
            f"champion_json version {champion_json['version']!r} does not match "
            f"expected version {version!r}"
        )

    documents = []
    for champion_id, champion in champion_json["data"].items():
        detail = champion_details[champion_id]
        tags = ", ".join(champion["tags"])
        abilities = _format_abilities(detail)
        documents.append(
            Document(
                page_content=_format_page_content(champion, tags, detail["lore"], abilities),
                metadata={
                    "source": f"ddragon:champion:{champion_id}:{version}",
                    "champion_id": champion_id,
                    "name": champion["name"],
                    "title": champion["title"],
                    "tags": tags,
                    "partype": champion["partype"],
                    "ddragon_version": version,
                },
            )
        )
    return documents


def ingest(settings: Settings) -> list[Document]:
    """Fetch and transform the full champion corpus for the configured version.

    Args:
        settings: Application settings providing ddragon_version, ddragon_locale,
            ddragon_base_url.

    Returns:
        List of Documents, one per champion, including full lore and abilities.
    """
    champion_json = fetch_champion_json(
        version=settings.ddragon_version,
        locale=settings.ddragon_locale,
        base_url=settings.ddragon_base_url,
    )
    champion_ids = list(champion_json["data"].keys())
    champion_details = fetch_all_champion_details(
        champion_ids,
        settings.ddragon_version,
        settings.ddragon_locale,
        settings.ddragon_base_url,
    )
    return champion_json_to_documents(champion_json, champion_details, settings.ddragon_version)
