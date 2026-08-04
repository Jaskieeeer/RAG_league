"""Dense retrieval over the stored chunks, ranked by cosine distance in Postgres.

The query is hand-written with SQLAlchemy Core rather than delegated to a vector
store wrapper, so the ranking is one readable SELECT that can be explained and
profiled. Cosine distance is not interchangeable here: the chunks index is an
HNSW index built with vector_cosine_ops, and any other operator would silently
fall back to a sequential scan over every chunk in the corpus.
"""

import logging

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select

from lolrag.config import Settings
from lolrag.db.models import Chunk
from lolrag.db.models import Document as DocumentRow
from lolrag.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def build_search_statement(embedding: list[float], k: int) -> Select:
    """Build the ranked nearest-neighbour SELECT over chunks joined to documents.

    Args:
        embedding: Query embedding, 384 floats from the same model the chunks
            were embedded with.
        k: Number of chunks to return.

    Returns:
        A Select yielding doc_key, title, collection, source, chunk_index, chunk
        content and cosine distance, ordered nearest first and limited to k. The
        embedding travels as a bound parameter, never as interpolated SQL.
    """
    distance = Chunk.embedding.cosine_distance(embedding).label("distance")
    return (
        select(
            DocumentRow.doc_key,
            DocumentRow.title,
            DocumentRow.collection,
            DocumentRow.source,
            Chunk.chunk_index,
            Chunk.content,
            distance,
        )
        .select_from(Chunk)
        .join(DocumentRow, DocumentRow.id == Chunk.document_id)
        .order_by(distance)
        .limit(k)
    )


class PgVectorRetriever(BaseRetriever):
    """Retriever returning the k chunks nearest a query, by cosine distance.

    The ranking is pure top-k over every chunk in the corpus: no deduplication
    by document and no collection filter, so several chunks of the same document
    can and do occupy several ranks. That is the naive v1 baseline the roadmap
    calls for, and both behaviours are eval-gated changes rather than defaults.

    Args:
        session: Open Session the query is executed through. The retriever never
            opens or closes a session of its own, so the caller controls the
            transaction the read happens in.
        settings: Application settings providing embedding_model_name.
        k: Number of chunks to return per query.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: Session = Field(description="Open Session the ranking query runs through.")
    settings: Settings = Field(description="Application settings providing the embedding model.")
    k: int = Field(description="Number of chunks to return per query.")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Embed the query and return the k nearest chunks, nearest first.

        Args:
            query: Natural-language query text.
            run_manager: Callback manager LangChain passes for this run.

        Returns:
            One Document per matched chunk, page_content set to the chunk text
            and metadata carrying doc_key, title, collection, source,
            chunk_index and distance.
        """
        embeddings = get_embeddings(self.settings.embedding_model_name)
        statement = build_search_statement(embeddings.embed_query(query), self.k)
        rows = self.session.execute(statement).all()
        logger.info("retrieved %d chunks for a query of %d characters", len(rows), len(query))
        return [
            Document(
                page_content=row.content,
                metadata={
                    "doc_key": row.doc_key,
                    "title": row.title,
                    "collection": row.collection,
                    "source": row.source,
                    "chunk_index": row.chunk_index,
                    "distance": row.distance,
                },
            )
            for row in rows
        ]
