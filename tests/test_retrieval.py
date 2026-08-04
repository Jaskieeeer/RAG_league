import math
from collections.abc import Iterator
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from lolrag import retrieval
from lolrag.config import Settings, get_settings
from lolrag.db.models import Chunk, Document, Faction
from lolrag.retrieval import PgVectorRetriever

EMBEDDING_DIMENSIONS = 384

INDEXED_AT = datetime(2026, 1, 1)


@pytest.fixture
def db_session() -> Iterator[Session]:
    """Yield a Session bound to a transaction that is rolled back after the test.

    Returns:
        A Session against the configured database; every change made through it
        is discarded when the test finishes, so the database stays empty.
    """
    engine = create_engine(get_settings().database_url)
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    try:
        yield session
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()
        engine.dispose()


def vector(angle: float) -> list[float]:
    """Build a unit embedding rotated by an angle in the first two dimensions.

    Args:
        angle: Angle in radians away from the query vector, which lies on the
            first axis.

    Returns:
        A 384-float unit vector whose cosine distance from vector(0.0) is
        exactly 1 - cos(angle), so the expected ranking is arithmetic rather
        than a property of any embedding model.
    """
    values = [0.0] * EMBEDDING_DIMENSIONS
    values[0] = math.cos(angle)
    values[1] = math.sin(angle)
    return values


class StubEmbeddings:
    """Embedding model that maps a query to a fixed vector.

    Args:
        embedding: The vector every embed_query call returns.
    """

    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding

    def embed_query(self, text: str) -> list[float]:
        """Return the fixed vector regardless of the query text.

        Args:
            text: Query text, ignored.

        Returns:
            The vector this stub was built with.
        """
        return self.embedding


def seed(session: Session) -> None:
    """Insert three lore documents whose chunks sit at known angles from the query.

    Args:
        session: Open Session the rows are written through; nothing is
            committed.

    Returns:
        None. The corpus is four chunks over three documents, two of them
        belonging to the same document so a ranking without deduplication is
        distinguishable from one with it.
    """
    seeds = [
        ("near", 0, 0.0),
        ("middle", 0, 0.5),
        ("middle", 1, 0.2),
        ("far", 0, 1.2),
    ]
    for slug in ("near", "middle", "far"):
        session.add(Faction(slug=slug, name=slug.title(), overview=None, overview_text=None))
    session.flush()

    documents = {}
    for slug in ("near", "middle", "far"):
        document = Document(
            doc_key=f"faction:{slug}",
            collection="lore",
            faction_slug=slug,
            title=f"Faction: {slug.title()}",
            source=f"Riot Universe faction {slug}",
            content=f"Overview of {slug}.",
            indexed_at=INDEXED_AT,
        )
        session.add(document)
        documents[slug] = document
    session.flush()

    for slug, chunk_index, angle in seeds:
        session.add(
            Chunk(
                document_id=documents[slug].id,
                chunk_index=chunk_index,
                content=f"{slug} chunk {chunk_index}",
                embedding=vector(angle),
            )
        )
    session.flush()


def retriever(session: Session, settings: Settings, k: int) -> PgVectorRetriever:
    """Build a retriever bound to the given session.

    Args:
        session: Open Session the ranking query runs through.
        settings: Application settings passed to the retriever.
        k: Number of chunks to return.

    Returns:
        A PgVectorRetriever bound to the session.
    """
    return PgVectorRetriever(session=session, settings=settings, k=k)


# ---------- ranking ----------


def test_retriever_ranks_chunks_by_ascending_cosine_distance(
    db_session: Session, monkeypatch
) -> None:
    """The nearest chunk leads and the farthest trails, in one pass over every chunk."""
    seed(db_session)
    monkeypatch.setattr(retrieval, "get_embeddings", lambda name: StubEmbeddings(vector(0.0)))

    documents = retriever(db_session, get_settings(), 4).invoke("anything")

    assert [doc.page_content for doc in documents] == [
        "near chunk 0",
        "middle chunk 1",
        "middle chunk 0",
        "far chunk 0",
    ]
    distances = [doc.metadata["distance"] for doc in documents]
    assert distances == sorted(distances)
    assert distances[0] == pytest.approx(0.0, abs=1e-6)


def test_retriever_returns_two_chunks_of_the_same_document_without_deduplicating(
    db_session: Session, monkeypatch
) -> None:
    """Pure top-k is the naive baseline; dedupe by document is an eval-gated later change."""
    seed(db_session)
    monkeypatch.setattr(retrieval, "get_embeddings", lambda name: StubEmbeddings(vector(0.3)))

    documents = retriever(db_session, get_settings(), 3).invoke("anything")

    assert [doc.metadata["doc_key"] for doc in documents].count("faction:middle") == 2


def test_retriever_honours_k(db_session: Session, monkeypatch) -> None:
    """The cutoff is the retriever's own k, not the whole corpus."""
    seed(db_session)
    monkeypatch.setattr(retrieval, "get_embeddings", lambda name: StubEmbeddings(vector(0.0)))

    assert len(retriever(db_session, get_settings(), 2).invoke("anything")) == 2


def test_retriever_searches_every_collection(db_session: Session, monkeypatch) -> None:
    """No collection filter exists, so an equipment chunk competes with a lore chunk."""
    seed(db_session)
    document = Document(
        doc_key="faction:equipment-stand-in",
        collection="equipment",
        faction_slug="near",
        title="Equipment stand-in",
        source="test",
        content="Equipment content.",
        indexed_at=INDEXED_AT,
    )
    db_session.add(document)
    db_session.flush()
    db_session.add(
        Chunk(document_id=document.id, chunk_index=0, content="closest", embedding=vector(0.1))
    )
    db_session.flush()
    monkeypatch.setattr(retrieval, "get_embeddings", lambda name: StubEmbeddings(vector(0.1)))

    documents = retriever(db_session, get_settings(), 5).invoke("anything")

    assert {doc.metadata["collection"] for doc in documents} == {"lore", "equipment"}
    assert documents[0].metadata["doc_key"] == "faction:equipment-stand-in"


# ---------- metadata ----------


def test_retriever_carries_the_document_metadata_onto_every_chunk(
    db_session: Session, monkeypatch
) -> None:
    """A chunk handed to the model on its own must still name what it came from."""
    seed(db_session)
    monkeypatch.setattr(retrieval, "get_embeddings", lambda name: StubEmbeddings(vector(0.0)))

    document = retriever(db_session, get_settings(), 1).invoke("anything")[0]

    assert document.page_content == "near chunk 0"
    assert document.metadata["doc_key"] == "faction:near"
    assert document.metadata["title"] == "Faction: Near"
    assert document.metadata["collection"] == "lore"
    assert document.metadata["source"] == "Riot Universe faction near"
    assert document.metadata["chunk_index"] == 0
    assert "distance" in document.metadata


def test_retriever_returns_nothing_when_the_corpus_is_empty(
    db_session: Session, monkeypatch
) -> None:
    """An unindexed database is an empty result, not an error."""
    monkeypatch.setattr(retrieval, "get_embeddings", lambda name: StubEmbeddings(vector(0.0)))

    assert retriever(db_session, get_settings(), 4).invoke("anything") == []


# ---------- statement shape ----------


def test_search_statement_binds_the_embedding_rather_than_interpolating_it() -> None:
    """The query vector must reach Postgres as a parameter, never as inlined SQL."""
    compiled = retrieval.build_search_statement(vector(0.0), 4).compile(
        dialect=postgresql.dialect()
    )

    assert "%(embedding_1)s" in str(compiled)
    assert "0.99500416" not in str(compiled)
    assert "LIMIT" in str(compiled)
    assert "ORDER BY distance" in str(compiled)


def test_search_statement_ranks_with_the_operator_the_hnsw_index_was_built_for() -> None:
    """The index is vector_cosine_ops, so any operator but <=> silently scans all 7650 rows."""
    compiled = str(
        retrieval.build_search_statement(vector(0.0), 4).compile(dialect=postgresql.dialect())
    )

    assert "<=>" in compiled
    assert "<->" not in compiled
    assert "<#>" not in compiled
