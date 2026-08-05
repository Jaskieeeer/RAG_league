from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from lolrag.config import Settings
from lolrag.fetch import corpus
from lolrag.fetch.client import FetchClient
from lolrag.fetch.corpus import CorpusCacheStats, warm_cache
from tests.test_fetch_client import build_settings

CHAMPION_IDS = ("Aatrox", "MonkeyKing")
CHAMPION_KEYS = ("266", "62")
FACTION_SLUGS = ("noxus", "ionia")
SHARED_STORY_SLUG = "a-new-dawn"
UNIQUE_STORY_SLUGS = ("the-blade-of-the-ruined-king", "the-monkey-king")

DDRAGON_DATA_PATH = "/cdn/16.14.1/data/en_US"
CDRAGON_CHAMPION_PATH = "/latest/plugins/rcp-be-lol-game-data/global/default/v1/champions"
CDRAGON_BIN_PATH = "/latest/game/data/characters"
CDRAGON_ITEM_BIN_PATH = "/latest/game/items.cdtb.bin.json"
UNIVERSE_PATH = "/v1/en_us"
RIOT_GAME_MODES_PATH = "/docs/lol/gameModes.json"

TOTAL_REQUESTS = 20


# ---------- fixture corpus ----------


def universe_champion_payload(unique_story_slug: str) -> dict[str, Any]:
    """Build a Universe champion payload referencing the shared story and one unique one.

    Args:
        unique_story_slug: Story slug referenced by this champion alone.

    Returns:
        Payload shaped like a real Universe champion response, whose modules
        list carries story-preview entries plus one module without a slug.
    """
    return {
        "modules": [
            {"type": "story-preview", "story-slug": SHARED_STORY_SLUG},
            {"type": "image"},
            {"type": "story-preview", "story-slug": unique_story_slug},
        ]
    }


def build_routes() -> dict[str, Any]:
    """Build the URL-path to response-body map covering every endpoint warm_cache touches.

    Returns:
        Mapping of URL path to the JSON body served for it, spanning two
        champions, two factions and three distinct stories.
    """
    routes: dict[str, Any] = {
        f"{DDRAGON_DATA_PATH}/champion.json": {
            "data": {
                champion_id: {"key": champion_key}
                for champion_id, champion_key in zip(CHAMPION_IDS, CHAMPION_KEYS, strict=True)
            }
        },
        f"{DDRAGON_DATA_PATH}/item.json": {"data": {}},
        f"{DDRAGON_DATA_PATH}/runesReforged.json": [],
        f"{DDRAGON_DATA_PATH}/summoner.json": {"data": {}},
        CDRAGON_ITEM_BIN_PATH: {"entries": {}},
        RIOT_GAME_MODES_PATH: [{"gameMode": "CLASSIC", "description": "Classic games"}],
        f"{UNIVERSE_PATH}/search/index.json": {
            "champions": [{"slug": champion_id.lower()} for champion_id in CHAMPION_IDS],
            "factions": [{"slug": slug} for slug in FACTION_SLUGS],
        },
    }
    for champion_id, champion_key, story_slug in zip(
        CHAMPION_IDS, CHAMPION_KEYS, UNIQUE_STORY_SLUGS, strict=True
    ):
        slug = champion_id.lower()
        routes[f"{DDRAGON_DATA_PATH}/champion/{champion_id}.json"] = {"data": {champion_id: {}}}
        routes[f"{CDRAGON_CHAMPION_PATH}/{champion_key}.json"] = {"id": int(champion_key)}
        routes[f"{CDRAGON_BIN_PATH}/{slug}/{slug}.bin.json"] = {"entries": {}}
        routes[f"{UNIVERSE_PATH}/champions/{slug}/index.json"] = universe_champion_payload(
            story_slug
        )
    for slug in FACTION_SLUGS:
        routes[f"{UNIVERSE_PATH}/factions/{slug}/index.json"] = {"slug": slug}
    for slug in (SHARED_STORY_SLUG, *UNIQUE_STORY_SLUGS):
        routes[f"{UNIVERSE_PATH}/story/{slug}/index.json"] = {"slug": slug}
    return routes


ROUTES = build_routes()


# ---------- harness ----------


def corpus_handler(calls: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    """Build a handler serving the fixture corpus and recording every request.

    Args:
        calls: List every received request is appended to.

    Returns:
        Handler suitable for httpx.MockTransport. Any path outside the fixture
        corpus gets a 404 so a wrong URL fails loudly instead of passing.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        body = ROUTES.get(request.url.path)
        if body is None:
            return httpx.Response(404, json={"unrouted": request.url.path})
        return httpx.Response(200, json=body)

    return handler


def _patch_client(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    """Replace FetchClient inside the corpus module with a mock-transport factory.

    Args:
        monkeypatch: Fixture performing the reversible attribute swap.
        handler: Callable invoked per request, returning the response to serve.

    Returns:
        None. warm_cache builds its own client, so the module attribute is the
        only seam that keeps every request off the network without changing
        production code.
    """

    def factory(settings: Settings, *, refresh: bool = False) -> FetchClient:
        return FetchClient(settings, refresh=refresh, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(corpus, "FetchClient", factory)


def paths_under(calls: list[httpx.Request], prefix: str) -> list[str]:
    """Return the recorded request paths starting with a prefix, in order.

    Args:
        calls: Requests recorded by the handler.
        prefix: URL path prefix selecting one endpoint family.

    Returns:
        Matching URL paths, duplicates included so repeat fetches are visible.
    """
    return [request.url.path for request in calls if request.url.path.startswith(prefix)]


# ---------- tests ----------


async def test_warm_cache_counts_every_document_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full run reports one count per document kind matching the fixture corpus."""
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []
    _patch_client(monkeypatch, corpus_handler(calls))

    stats = await warm_cache(settings)

    assert stats == CorpusCacheStats(
        ddragon_champions=2,
        cdragon_champions=2,
        cdragon_champion_bins=2,
        cdragon_item_bin=1,
        universe_champions=2,
        universe_factions=2,
        universe_stories=3,
        riot_game_modes=1,
        cache_hits=0,
        cache_misses=TOTAL_REQUESTS,
    )
    assert len(calls) == TOTAL_REQUESTS


async def test_champion_bins_are_addressed_by_lowercase_id_never_by_numeric_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bin fetcher receives champion ids, so no bin URL carries a numeric champion key."""
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []
    _patch_client(monkeypatch, corpus_handler(calls))

    await warm_cache(settings)

    bin_paths = paths_under(calls, CDRAGON_BIN_PATH)
    assert sorted(bin_paths) == [
        f"{CDRAGON_BIN_PATH}/aatrox/aatrox.bin.json",
        f"{CDRAGON_BIN_PATH}/monkeyking/monkeyking.bin.json",
    ]
    assert not [path for path in bin_paths for key in CHAMPION_KEYS if key in path]


async def test_story_slug_shared_by_two_champions_is_fetched_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both champions reference one common story, which is requested a single time."""
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []
    _patch_client(monkeypatch, corpus_handler(calls))

    stats = await warm_cache(settings)

    story_paths = paths_under(calls, f"{UNIVERSE_PATH}/story/")
    assert story_paths.count(f"{UNIVERSE_PATH}/story/{SHARED_STORY_SLUG}/index.json") == 1
    assert sorted(story_paths) == sorted(
        f"{UNIVERSE_PATH}/story/{slug}/index.json"
        for slug in (SHARED_STORY_SLUG, *UNIQUE_STORY_SLUGS)
    )
    assert stats.universe_stories == 3


async def test_second_run_against_a_warm_cache_issues_no_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rerunning against the same cache directory is served entirely from disk."""
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []
    _patch_client(monkeypatch, corpus_handler(calls))
    await warm_cache(settings)
    first_run_calls = len(calls)

    stats = await warm_cache(settings)

    assert len(calls) == first_run_calls
    assert stats.cache_misses == 0
    assert stats.cache_hits == TOTAL_REQUESTS


async def test_refresh_refetches_the_whole_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """refresh=True ignores a populated cache and requests every document again."""
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []
    _patch_client(monkeypatch, corpus_handler(calls))
    await warm_cache(settings)
    first_run_calls = len(calls)

    stats = await warm_cache(settings, refresh=True)

    assert len(calls) == first_run_calls + TOTAL_REQUESTS
    assert stats.cache_hits == 0
    assert stats.cache_misses == TOTAL_REQUESTS
