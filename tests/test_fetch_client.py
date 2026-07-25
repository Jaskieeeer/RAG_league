from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from lolrag.config import Settings
from lolrag.fetch import client as client_module
from lolrag.fetch.cache import cache_path, write_cached
from lolrag.fetch.client import FetchClient

VERSIONS_URL = "https://ddragon.test/api/versions.json"
VERSIONS_SEGMENTS = ("ddragon", "api", "versions.json")


def build_settings(cache_dir: Path, **overrides: Any) -> Settings:
    """Build hermetic Settings pointing every base URL at a fake host.

    Args:
        cache_dir: Directory used as the cache root, normally pytest's tmp_path.
        **overrides: Field values overriding the test defaults.

    Returns:
        Settings with no network delay and a three-attempt retry budget.
    """
    values: dict[str, Any] = {
        "ddragon_version": "16.14.1",
        "ddragon_locale": "en_US",
        "ddragon_base_url": "https://ddragon.test",
        "cdragon_base_url": "https://cdragon.test",
        "universe_base_url": "https://universe.test/v1/en_us",
        "cache_dir": str(cache_dir),
        "http_delay_seconds": 0.0,
        "http_timeout_seconds": 1.0,
        "http_max_retries": 3,
    }
    values.update(overrides)
    return Settings(**values)


def build_client(
    settings: Settings,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    refresh: bool = False,
) -> FetchClient:
    """Build a FetchClient whose transport is an in-memory mock.

    Args:
        settings: Settings the client reads its cache and retry policy from.
        handler: Callable invoked per request, returning the response to serve.
        refresh: Whether the client ignores existing cache entries.

    Returns:
        FetchClient with its httpx.AsyncClient replaced by one backed by
        httpx.MockTransport, so no test ever reaches the network.
    """
    client = FetchClient(settings, refresh=refresh)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"User-Agent": settings.http_user_agent},
        timeout=settings.http_timeout_seconds,
    )
    return client


def json_handler(
    calls: list[httpx.Request], *statuses: int, body: bytes = b'["16.14.1"]'
) -> Callable[[httpx.Request], httpx.Response]:
    """Build a handler returning the given statuses in order, recording requests.

    Args:
        calls: List every received request is appended to.
        *statuses: Status codes to return, one per call; the last is reused
            once exhausted.
        body: Response body returned with every status.

    Returns:
        Handler suitable for httpx.MockTransport.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        status = statuses[min(len(calls) - 1, len(statuses) - 1)]
        return httpx.Response(status, content=body)

    return handler


async def test_cache_hit_short_circuits_the_network(tmp_path: Path) -> None:
    """A pre-seeded cache entry is returned without the transport being invoked."""
    settings = build_settings(tmp_path)
    write_cached(cache_path(tmp_path, *VERSIONS_SEGMENTS), b'["16.14.1", "16.13.1"]')

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("a cache hit must not reach the network")

    async with build_client(settings, handler) as client:
        payload = await client.get_json(VERSIONS_URL, *VERSIONS_SEGMENTS)

        assert payload == ["16.14.1", "16.13.1"]
        assert client.hits == 1
        assert client.misses == 0


async def test_refresh_bypasses_an_existing_cache_entry(tmp_path: Path) -> None:
    """refresh=True refetches and overwrites a cache entry that already exists."""
    settings = build_settings(tmp_path)
    path = cache_path(tmp_path, *VERSIONS_SEGMENTS)
    write_cached(path, b'["stale"]')
    calls: list[httpx.Request] = []

    async with build_client(
        settings, json_handler(calls, 200, body=b'["16.14.1"]'), refresh=True
    ) as client:
        payload = await client.get_json(VERSIONS_URL, *VERSIONS_SEGMENTS)

        assert payload == ["16.14.1"]
        assert client.hits == 0
        assert client.misses == 1

    assert len(calls) == 1
    assert path.read_bytes() == b'["16.14.1"]'


async def test_retry_succeeds_after_two_server_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two 503s followed by a 200 yield the payload after exactly three calls."""
    monkeypatch.setattr(client_module, "_BACKOFF_BASE_SECONDS", 0.0)
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []

    async with build_client(settings, json_handler(calls, 503, 503, 200)) as client:
        payload = await client.get_json(VERSIONS_URL, *VERSIONS_SEGMENTS)

    assert payload == ["16.14.1"]
    assert len(calls) == 3


async def test_retry_exhaustion_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A permanently failing endpoint raises once the retry budget is spent."""
    monkeypatch.setattr(client_module, "_BACKOFF_BASE_SECONDS", 0.0)
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []

    async with build_client(settings, json_handler(calls, 503)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_json(VERSIONS_URL, *VERSIONS_SEGMENTS)

    assert len(calls) == settings.http_max_retries


async def test_transport_error_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transport failure is retried and a later success is returned."""
    monkeypatch.setattr(client_module, "_BACKOFF_BASE_SECONDS", 0.0)
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, content=b'["16.14.1"]')

    async with build_client(settings, handler) as client:
        payload = await client.get_json(VERSIONS_URL, *VERSIONS_SEGMENTS)

    assert payload == ["16.14.1"]
    assert len(calls) == 2


async def test_not_found_raises_immediately_without_retry(tmp_path: Path) -> None:
    """A 404 is final: it raises after exactly one call."""
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []

    async with build_client(settings, json_handler(calls, 404)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_json(VERSIONS_URL, *VERSIONS_SEGMENTS)

    assert len(calls) == 1


async def test_forbidden_raises_immediately_without_retry(tmp_path: Path) -> None:
    """A 403, as some Universe hub endpoints return, raises after exactly one call."""
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []

    async with build_client(settings, json_handler(calls, 403)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_json(VERSIONS_URL, *VERSIONS_SEGMENTS)

    assert len(calls) == 1


async def test_failed_fetch_writes_nothing_to_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither a final error nor an intermediate one leaves a cache entry behind."""
    monkeypatch.setattr(client_module, "_BACKOFF_BASE_SECONDS", 0.0)
    settings = build_settings(tmp_path)
    calls: list[httpx.Request] = []

    async with build_client(settings, json_handler(calls, 503)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_json(VERSIONS_URL, *VERSIONS_SEGMENTS)

    assert not cache_path(tmp_path, *VERSIONS_SEGMENTS).exists()
    assert list(tmp_path.rglob("*")) == []


async def test_user_agent_header_comes_from_settings(tmp_path: Path) -> None:
    """Every outgoing request carries the configured User-Agent."""
    settings = build_settings(tmp_path, http_user_agent="lolrag-test/1.0")
    calls: list[httpx.Request] = []

    async with build_client(settings, json_handler(calls, 200)) as client:
        await client.get_json(VERSIONS_URL, *VERSIONS_SEGMENTS)

    assert calls[0].headers["User-Agent"] == "lolrag-test/1.0"


def test_retry_after_header_overrides_backoff() -> None:
    """A 429 carrying Retry-After waits for that value instead of the backoff."""
    response = httpx.Response(429, headers={"Retry-After": "7"})

    assert client_module._retry_delay_seconds(response, attempt=3) == 7.0


def test_backoff_is_exponential_without_retry_after() -> None:
    """Without a Retry-After header the delay doubles per attempt."""
    response = httpx.Response(503)

    delays = [client_module._retry_delay_seconds(response, attempt) for attempt in (1, 2, 3)]
    assert delays == [1.0, 2.0, 4.0]
