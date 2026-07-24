from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from lolrag.config import Settings


@lru_cache
def get_engine(database_url: str) -> Engine:
    """Return a process-wide cached SQLAlchemy engine for database_url.

    Args:
        database_url: SQLAlchemy connection URL, e.g.
            "postgresql+psycopg://lolrag:lolrag@localhost:5432/lolrag".

    Returns:
        Engine for database_url, created once per process and reused on
        subsequent calls with the same database_url.
    """
    return create_engine(database_url)


@lru_cache
def get_session_factory(database_url: str) -> sessionmaker[Session]:
    """Return a process-wide cached session factory bound to database_url.

    Args:
        database_url: SQLAlchemy connection URL the returned factory's
            sessions connect through.

    Returns:
        sessionmaker bound to the engine for database_url, created once per
        process and reused on subsequent calls with the same database_url.
    """
    return sessionmaker(bind=get_engine(database_url))


def get_session(settings: Settings) -> Session:
    """Open a new Session against the database configured in settings.

    Args:
        settings: Application settings providing database_url. Settings is
            not hashable, so caching happens on settings.database_url rather
            than on settings itself.

    Returns:
        A new Session; the caller is responsible for closing it, typically
        via a `with` block.
    """
    return get_session_factory(settings.database_url)()
