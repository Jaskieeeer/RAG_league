from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from lolrag.config import Settings
from lolrag.indexing import get_vector_store

_SYSTEM_PROMPT = (
    "You are a League of Legends knowledge assistant. Answer the question using "
    "only the context provided below. If the context does not contain the answer, "
    "say so explicitly. Do not use any outside knowledge.\n\n"
    "Context:\n{context}"
)


class SourceDocument(BaseModel):
    """A single retrieved source cited alongside a generated answer.

    Args:
        champion_id: Data Dragon champion id, e.g. "Aatrox", if the source
            document's metadata carries one.
        name: Source document display name, if the source document's metadata
            carries one.
        source: Document metadata "source" value the answer was grounded in.
    """

    champion_id: str | None = Field(
        default=None, description="Data Dragon champion id, if the source document has one."
    )
    name: str | None = Field(
        default=None, description="Source document display name, if it has one."
    )
    source: str = Field(description="Identifier of the document the answer was grounded in.")


class RagResponse(BaseModel):
    """The full result of answering a question against the RAG pipeline.

    Args:
        answer: Generated answer text, grounded in the retrieved sources.
        sources: Documents retrieved and used to ground the answer.
    """

    answer: str = Field(description="Generated answer text, grounded in the retrieved sources.")
    sources: list[SourceDocument] = Field(
        description="Documents retrieved and used to ground the answer."
    )


def retrieve(question: str, settings: Settings) -> list[Document]:
    """Retrieve the most relevant documents for a question from the vector store.

    Args:
        question: User question to retrieve context for.
        settings: Application settings providing retriever_k and vector store
            configuration.

    Returns:
        Up to settings.retriever_k Documents most relevant to question.
    """
    retriever = get_vector_store(settings).as_retriever(search_kwargs={"k": settings.retriever_k})
    return retriever.invoke(question)


def _format_context(documents: list[Document]) -> str:
    """Join retrieved documents into a single text block for the prompt context.

    Args:
        documents: Retrieved documents, each with a "source" metadata key and
            optionally a "name" metadata key.

    Returns:
        Documents rendered as "{label}: {page_content}", separated by blank
        lines, where label is the document's "name" metadata if present and
        its "source" metadata otherwise.
    """
    lines = []
    for doc in documents:
        label = doc.metadata.get("name", doc.metadata["source"])
        lines.append(f"{label}: {doc.page_content}")
    return "\n\n".join(lines)


def _build_prompt() -> ChatPromptTemplate:
    """Build the grounded chat prompt used for answer generation.

    Returns:
        ChatPromptTemplate with a system message carrying grounding
        instructions and the {context} placeholder, and a human message
        template containing only the literal {question}.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )


@lru_cache
def get_llm(model_name: str, fallback_model_name: str, temperature: float, api_key: str | None) -> Runnable:
    """Return a process-wide cached chat model with a fallback configured.

    Args:
        model_name: Primary Gemini chat model identifier.
        fallback_model_name: Gemini chat model identifier used if the primary fails.
        temperature: Sampling temperature applied to both models.

    Returns:
        Runnable that invokes model_name and falls back to fallback_model_name
        on failure, cached per unique (model_name, fallback_model_name, temperature).
    """
    primary = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key)
    fallback = ChatGoogleGenerativeAI(model=fallback_model_name, temperature=temperature, google_api_key=api_key)
    return primary.with_fallbacks([fallback])


def generate(question: str, documents: list[Document], settings: Settings) -> str:
    """Generate a grounded answer from retrieved documents.

    Args:
        question: User question to answer.
        documents: Retrieved documents to ground the answer in.
        settings: Application settings providing llm_model_name,
            llm_fallback_model_name, llm_temperature.

    Returns:
        Generated answer text.
    """
    prompt = _build_prompt()
    messages = prompt.format_messages(context=_format_context(documents), question=question)
    llm = get_llm(
        settings.llm_model_name,
        settings.llm_fallback_model_name,
        settings.llm_temperature,
        settings.google_api_key.get_secret_value() if settings.google_api_key else None,
    )
    response = llm.invoke(messages)
    return str(response.content)


def answer_question(question: str, settings: Settings) -> RagResponse:
    """Answer a question end to end: retrieve context, generate, cite sources.

    Args:
        question: User question to answer.
        settings: Application settings for retrieval and generation.

    Returns:
        RagResponse with the generated answer and the sources it was grounded in.
    """
    documents = retrieve(question, settings)
    answer = generate(question, documents, settings)
    sources = [
        SourceDocument(
            champion_id=doc.metadata.get("champion_id"),
            name=doc.metadata.get("name"),
            source=doc.metadata["source"],
        )
        for doc in documents
    ]
    return RagResponse(answer=answer, sources=sources)
