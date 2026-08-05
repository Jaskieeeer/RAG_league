from pathlib import Path

import httpx
import pytest

from lolrag.config import get_settings
from lolrag.fetch import cdragon, cdragon_bin, ddragon, riot_static, universe
from lolrag.fetch.cache import cache_path
from lolrag.fetch.client import FetchClient
from tests.test_fetch_client import build_client, build_settings

CHAMPION_PAYLOAD = {
    "modules": [
        {"type": "story-preview", "story-slug": "the-blade-of-the-ruined-king"},
        {"type": "image"},
        {"type": "story-preview", "story-slug": "a-new-dawn"},
        {"type": "story-preview", "story-slug": "the-blade-of-the-ruined-king"},
        {"type": "story-preview", "story-slug": ""},
    ]
}


def ok_handler(calls: list[httpx.Request]) -> object:
    """Build a handler returning an empty JSON object and recording requests.

    Args:
        calls: List every received request is appended to.

    Returns:
        Handler suitable for httpx.MockTransport.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=b"{}")

    return handler


# ---------- ddragon ----------


@pytest.mark.parametrize(
    ("call", "expected_url", "expected_segments"),
    [
        (
            ddragon.fetch_versions,
            "https://ddragon.test/api/versions.json",
            ("ddragon", "api", "versions.json"),
        ),
        (
            ddragon.fetch_champion_list,
            "https://ddragon.test/cdn/16.14.1/data/en_US/champion.json",
            ("ddragon", "16.14.1", "en_US", "champion.json"),
        ),
        (
            ddragon.fetch_items,
            "https://ddragon.test/cdn/16.14.1/data/en_US/item.json",
            ("ddragon", "16.14.1", "en_US", "item.json"),
        ),
        (
            ddragon.fetch_runes,
            "https://ddragon.test/cdn/16.14.1/data/en_US/runesReforged.json",
            ("ddragon", "16.14.1", "en_US", "runesReforged.json"),
        ),
        (
            ddragon.fetch_summoner_spells,
            "https://ddragon.test/cdn/16.14.1/data/en_US/summoner.json",
            ("ddragon", "16.14.1", "en_US", "summoner.json"),
        ),
    ],
)
async def test_ddragon_bulk_endpoints_use_expected_url_and_cache_path(
    tmp_path: Path, call: object, expected_url: str, expected_segments: tuple[str, ...]
) -> None:
    """Each Data Dragon bulk fetcher requests the documented URL and caches it in place."""
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []

    async with build_client(settings, ok_handler(calls)) as client:
        await call(client, settings)

    assert str(calls[0].url) == expected_url
    assert cache_path(tmp_path, *expected_segments).exists()


async def test_ddragon_champion_detail_uses_expected_url_and_cache_path(tmp_path: Path) -> None:
    """The per-champion detail fetcher nests the champion file under a champion directory."""
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []

    async with build_client(settings, ok_handler(calls)) as client:
        await ddragon.fetch_champion_detail(client, settings, "Aatrox")

    assert str(calls[0].url) == "https://ddragon.test/cdn/16.14.1/data/en_US/champion/Aatrox.json"
    assert cache_path(tmp_path, "ddragon", "16.14.1", "en_US", "champion", "Aatrox.json").exists()


async def test_ddragon_fetch_all_champion_details_keys_by_champion_id(tmp_path: Path) -> None:
    """Gathered detail payloads come back keyed by the requested champion ids."""
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []

    async with build_client(settings, ok_handler(calls)) as client:
        details = await ddragon.fetch_all_champion_details(client, settings, ["Aatrox", "Ahri"])

    assert set(details) == {"Aatrox", "Ahri"}
    assert len(calls) == 2


# ---------- cdragon ----------


async def test_cdragon_champion_uses_expected_url_and_cache_path(tmp_path: Path) -> None:
    """The Community Dragon fetcher addresses a champion by its numeric key."""
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []

    async with build_client(settings, ok_handler(calls)) as client:
        await cdragon.fetch_champion(client, settings, 266)

    assert str(calls[0].url) == (
        "https://cdragon.test/latest/plugins/rcp-be-lol-game-data"
        "/global/default/v1/champions/266.json"
    )
    assert cache_path(tmp_path, "cdragon", "latest", "champions", "266.json").exists()


async def test_cdragon_fetch_all_champions_keys_by_numeric_key(tmp_path: Path) -> None:
    """Gathered Community Dragon payloads come back keyed by numeric champion key."""
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []

    async with build_client(settings, ok_handler(calls)) as client:
        champions = await cdragon.fetch_all_champions(client, settings, [266, 103])

    assert set(champions) == {266, 103}
    assert len(calls) == 2


# ---------- cdragon bin ----------


async def test_cdragon_champion_bin_lowercases_the_slug_in_both_path_positions(
    tmp_path: Path,
) -> None:
    """The bin fetcher addresses a champion by its lowercase slug, used twice in the path."""
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []

    async with build_client(settings, ok_handler(calls)) as client:
        await cdragon_bin.fetch_champion_bin(client, settings, "MonkeyKing")

    assert str(calls[0].url) == (
        "https://cdragon.test/latest/game/data/characters/monkeyking/monkeyking.bin.json"
    )


async def test_cdragon_champion_bin_caches_and_serves_the_second_call(tmp_path: Path) -> None:
    """The bin payload lands at its cache path and a repeat call issues no second request."""
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []

    async with build_client(settings, ok_handler(calls)) as client:
        await cdragon_bin.fetch_champion_bin(client, settings, "MonkeyKing")
        await cdragon_bin.fetch_champion_bin(client, settings, "MonkeyKing")

    assert cache_path(tmp_path, "cdragon", "latest", "characters", "monkeyking.bin.json").exists()
    assert len(calls) == 1


async def test_cdragon_fetch_all_champion_bins_keys_by_original_id(tmp_path: Path) -> None:
    """Gathered bin payloads come back keyed by the requested ids, not by their slugs."""
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []

    async with build_client(settings, ok_handler(calls)) as client:
        bins = await cdragon_bin.fetch_all_champion_bins(client, settings, ["MonkeyKing", "KaiSa"])

    assert set(bins) == {"MonkeyKing", "KaiSa"}
    assert len(calls) == 2


async def test_cdragon_item_bin_uses_expected_url_and_cache_path(tmp_path: Path) -> None:
    """The item bin is a single file fetched without a champion segment."""
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []

    async with build_client(settings, ok_handler(calls)) as client:
        await cdragon_bin.fetch_item_bin(client, settings)

    assert str(calls[0].url) == "https://cdragon.test/latest/game/items.cdtb.bin.json"
    assert cache_path(tmp_path, "cdragon", "latest", "items.cdtb.bin.json").exists()


# ---------- riot static docs ----------


async def test_game_modes_use_expected_url_and_cache_path(tmp_path: Path) -> None:
    """The game-mode list is a single unversioned file cached under its own root."""
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []

    async with build_client(settings, ok_handler(calls)) as client:
        await riot_static.fetch_game_modes(client, settings)

    assert str(calls[0].url) == "https://riotstatic.test/docs/lol/gameModes.json"
    assert cache_path(tmp_path, "riot_static", "gameModes.json").exists()


# ---------- universe ----------


@pytest.mark.parametrize(
    ("name", "argument", "expected_path", "expected_segments"),
    [
        ("fetch_search_index", None, "search/index.json", ("universe", "search", "index.json")),
        (
            "fetch_champion",
            "aatrox",
            "champions/aatrox/index.json",
            ("universe", "champions", "aatrox.json"),
        ),
        (
            "fetch_faction",
            "noxus",
            "factions/noxus/index.json",
            ("universe", "factions", "noxus.json"),
        ),
        (
            "fetch_story",
            "a-new-dawn",
            "story/a-new-dawn/index.json",
            ("universe", "story", "a-new-dawn.json"),
        ),
    ],
)
async def test_universe_endpoints_use_expected_url_and_cache_path(
    tmp_path: Path,
    name: str,
    argument: str | None,
    expected_path: str,
    expected_segments: tuple[str, ...],
) -> None:
    """Each Universe fetcher requests the documented URL and caches it in place."""
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []
    call = getattr(universe, name)

    async with build_client(settings, ok_handler(calls)) as client:
        if argument is None:
            await call(client, settings)
        else:
            await call(client, settings, argument)

    assert str(calls[0].url) == f"https://universe.test/v1/en_us/{expected_path}"
    assert cache_path(tmp_path, *expected_segments).exists()


async def test_universe_fetch_all_stories_keys_by_slug(tmp_path: Path) -> None:
    """Gathered story payloads come back keyed by the requested slugs."""
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []

    async with build_client(settings, ok_handler(calls)) as client:
        stories = await universe.fetch_all_stories(client, settings, ["a-new-dawn", "rise"])

    assert set(stories) == {"a-new-dawn", "rise"}
    assert len(calls) == 2


def test_extract_story_slugs_preserves_order_and_deduplicates() -> None:
    """Story slugs are collected in order, once each, skipping modules without one."""
    assert universe.extract_story_slugs(CHAMPION_PAYLOAD) == [
        "the-blade-of-the-ruined-king",
        "a-new-dawn",
    ]


def test_extract_story_slugs_handles_payload_without_modules() -> None:
    """A payload with no modules yields no story slugs."""
    assert universe.extract_story_slugs({}) == []


# ---------- integration ----------


@pytest.mark.integration
async def test_fetch_versions_against_live_ddragon(tmp_path: Path) -> None:
    """The real Data Dragon version manifest returns a non-empty list of version strings."""
    settings = get_settings().model_copy(update={"cache_dir": str(tmp_path)})

    async with FetchClient(settings) as client:
        versions = await ddragon.fetch_versions(client, settings)

    assert versions
    assert all(isinstance(version, str) for version in versions)
