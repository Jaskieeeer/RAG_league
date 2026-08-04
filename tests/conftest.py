"""Fixtures shared by every test, and the TLS setup the live-model tests need.

truststore is injected here for the same reason the CLI injects it: the Gemini
client verifies against Python's own certificate store, which does not hold this
machine's trust roots, so an eval-marked run would fail on TLS before it reached
the model. The CLI got this at import of __main__; pytest has no such entry
point, so it goes here.
"""

import asyncio
from collections.abc import Iterator

import pytest
import truststore
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from lolrag.config import get_settings
from lolrag.fetch.client import FetchClient
from lolrag.ingest.run import run_ingest

truststore.inject_into_ssl()


async def _ingest(session: Session) -> None:
    """Load the full corpus into an open session.

    Args:
        session: Session the rows are written through; nothing is committed.
    """
    settings = get_settings()
    async with FetchClient(settings) as client:
        await run_ingest(session, client, settings)


@pytest.fixture
def corpus_session() -> Iterator[Session]:
    """Yield a Session holding one full corpus ingest, rolled back after the test.

    The scope is deliberately per-test rather than per-session. A longer-lived
    transaction would still hold its uncommitted rows while the other
    corpus-marked tests ran, and their inserts would block on the unique keys it
    has not released yet. Nothing is committed here either, so the database is
    left exactly as empty as it was found.

    Returns:
        A Session against the configured database with the full corpus loaded.
    """
    engine = create_engine(get_settings().database_url)
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    try:
        asyncio.run(_ingest(session))
        yield session
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()
        engine.dispose()
