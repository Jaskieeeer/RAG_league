import asyncio
import json
import logging
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx

from lolrag.config import Settings
from lolrag.fetch.cache import cache_path, read_cached, write_cached

logger = logging.getLogger(__name__)

_RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_BACKOFF_BASE_SECONDS = 1.0


class FetchClient:
    """Async HTTP client that serves corpus requests from an immutable on-disk cache.

    Args:
        settings: Application settings providing cache_dir, http_user_agent,
            http_concurrency, http_delay_seconds, http_timeout_seconds,
            http_max_retries.
        refresh: If True, ignore existing cache entries and refetch every URL,
            overwriting the cache. Cache entries are otherwise never stale.
        transport: Transport the underlying httpx.AsyncClient sends through,
            letting tests supply an in-memory transport so they never reach the
            network. None is httpx's own default, meaning the real network.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        refresh: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._refresh = refresh
        self._cache_dir = Path(settings.cache_dir)
        self._semaphore = asyncio.Semaphore(settings.http_concurrency)
        self._client = httpx.AsyncClient(
            headers={"User-Agent": settings.http_user_agent},
            timeout=settings.http_timeout_seconds,
            transport=transport,
        )
        self.hits = 0
        self.misses = 0

    async def __aenter__(self) -> "FetchClient":
        """Enter the async context, returning the client itself.

        Returns:
            This FetchClient, with its underlying httpx.AsyncClient open.
        """
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the underlying httpx.AsyncClient on context exit.

        Args:
            exc_type: Exception class raised in the context, or None.
            exc: Exception instance raised in the context, or None.
            tb: Traceback of the raised exception, or None.

        Returns:
            None. Exceptions are never suppressed.
        """
        await self._client.__aexit__(exc_type, exc, tb)

    async def get_json(self, url: str, *cache_segments: str) -> Any:
        """Return the parsed JSON body for url, from cache when available.

        Args:
            url: Absolute URL to fetch.
            *cache_segments: Path segments locating this URL's cache entry
                under settings.cache_dir, mirroring the URL's structure.

        Returns:
            The parsed JSON body, typically a dict or a list depending on the
            endpoint. Riot payloads are returned unmodelled and unmodified.

        Raises:
            httpx.HTTPStatusError: If the final attempt returns a non-2xx status.
            httpx.TransportError: If every attempt fails at the transport level.
            json.JSONDecodeError: If the body is not valid JSON.
        """
        path = cache_path(self._cache_dir, *cache_segments)
        if not self._refresh:
            cached = read_cached(path)
            if cached is not None:
                self.hits += 1
                logger.debug("cache hit %s", path)
                return json.loads(cached)

        self.misses += 1
        response = await self._fetch(url)
        if response.status_code == 200:
            write_cached(path, response.content)
        return json.loads(response.content)

    async def _fetch(self, url: str) -> httpx.Response:
        """Fetch url over the network with rate limiting and bounded retries.

        Args:
            url: Absolute URL to fetch.

        Returns:
            The first 2xx response. Only a 200 is written to the cache by the
            caller, so a transient failure is never frozen into the corpus.

        Raises:
            httpx.HTTPStatusError: If the final attempt returns a non-2xx status.
            httpx.TransportError: If every attempt fails at the transport level.
            RuntimeError: If http_max_retries is below 1, leaving no attempts.
        """
        async with self._semaphore:
            for attempt in range(1, self._settings.http_max_retries + 1):
                last_attempt = attempt == self._settings.http_max_retries
                try:
                    await asyncio.sleep(self._settings.http_delay_seconds)
                    response = await self._client.get(url)
                except httpx.TransportError as error:
                    if last_attempt:
                        raise
                    logger.warning(
                        "transport error on %s (attempt %d/%d): %s",
                        url,
                        attempt,
                        self._settings.http_max_retries,
                        error,
                    )
                    await asyncio.sleep(_backoff_seconds(attempt))
                    continue

                if response.status_code in _RETRY_STATUS_CODES and not last_attempt:
                    logger.warning(
                        "status %d on %s (attempt %d/%d)",
                        response.status_code,
                        url,
                        attempt,
                        self._settings.http_max_retries,
                    )
                    await asyncio.sleep(_retry_delay_seconds(response, attempt))
                    continue

                response.raise_for_status()
                return response

        raise RuntimeError(f"retry loop exited without a result for {url}")


def _backoff_seconds(attempt: int) -> float:
    """Return the exponential backoff delay for a failed attempt.

    Args:
        attempt: 1-based attempt number that just failed.

    Returns:
        Seconds to wait before the next attempt: 1, 2, 4, ...
    """
    return _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))


def _retry_delay_seconds(response: httpx.Response, attempt: int) -> float:
    """Return the delay before retrying a retryable response.

    Args:
        response: The retryable response, possibly carrying a Retry-After header.
        attempt: 1-based attempt number that just failed.

    Returns:
        The Retry-After value in seconds when the server sent a parseable one on
        a 429, otherwise the exponential backoff delay.
    """
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return float(retry_after)
            except ValueError:
                logger.warning("unparseable Retry-After header %r", retry_after)
    return _backoff_seconds(attempt)
