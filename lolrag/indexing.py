import hashlib
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from lolrag.config import Settings

_ID_LENGTH = 32


def compute_document_id(source: str, content: str) -> str:
    """Derive a deterministic Chroma document id from a document's source and content.

    Args:
        source: Document metadata "source" value, e.g. "ddragon:champion:Aatrox:16.14.1".
        content: Document page_content.

    Returns:
        First 32 hex characters of the sha256 digest of "{source}:{content}". Stable
        across repeated calls with identical inputs, so re-ingesting unchanged
        content upserts in place instead of creating a duplicate entry.
    """
    digest = hashlib.sha256(f"{source}:{content}".encode()).hexdigest()
    return digest[:_ID_LENGTH]


@lru_cache
def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Return a process-wide cached embedding model instance.

    Args:
        model_name: HuggingFace embedding model identifier.

    Returns:
        HuggingFaceEmbeddings for model_name, loaded once per process and reused
        on subsequent calls with the same model_name.
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def get_vector_store(settings: Settings) -> Chroma:
    """Open or create the persistent Chroma collection, without adding documents.

    Args:
        settings: Application settings providing chroma_collection_name,
            chroma_persist_dir, embedding_model_name.

    Returns:
        Chroma vector store backed by the configured persistent directory and
        collection, ready for querying or upserting.
    """
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(settings.embedding_model_name),
        persist_directory=settings.chroma_persist_dir,
    )


def build_index(documents: list[Document], settings: Settings) -> Chroma:
    """Upsert documents into the persistent Chroma collection.

    Args:
        documents: Documents to index, each with a "source" metadata key.
        settings: Application settings providing chroma_collection_name,
            chroma_persist_dir, embedding_model_name.

    Returns:
        The Chroma vector store the documents were upserted into.
    """
    vector_store = get_vector_store(settings)
    ids = [compute_document_id(doc.metadata["source"], doc.page_content) for doc in documents]
    vector_store.add_documents(documents, ids=ids)
    return vector_store
