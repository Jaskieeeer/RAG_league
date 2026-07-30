from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


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
