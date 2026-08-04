from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from lolrag.config import Settings
from lolrag.retrieval import PgVectorRetriever

_SYSTEM_PROMPT = (
    "You are a League of Legends knowledge assistant. Answer the question using "
    "only the context provided below. If the context does not contain the answer, "
    "say so explicitly. Do not use any outside knowledge.\n\n"
    "Context:\n{context}"
)


class SourceDocument(BaseModel):
    """A single retrieved chunk cited alongside a generated answer.

    The cosine distance the chunk ranked at is deliberately absent: it is a
    property of one retriever's scoring, not of the source, and exposing it
    would invite callers to compare numbers across retrievers that do not share
    a scale. It travels in the retrieved Document's metadata instead, where
    evaluation and debugging can still read it.

    Args:
        doc_key: Deterministic key of the document the chunk belongs to, e.g.
            "ability:aatrox:Q".
        title: Title of that document, e.g. "Aatrox Q: The Darkin Blade".
        collection: Collection the document belongs to, one of abilities,
            champion_stats, equipment, lore.
        source: Human-readable provenance string for the document.
        chunk_index: Position within the document of the chunk that matched.
        content: Full text of the matched chunk.
    """

    doc_key: str = Field(description="Deterministic key of the document the chunk belongs to.")
    title: str = Field(description="Title of the document the chunk belongs to.")
    collection: str = Field(description="Collection the document belongs to.")
    source: str = Field(description="Human-readable provenance string for the document.")
    chunk_index: int = Field(description="Position within the document of the chunk that matched.")
    content: str = Field(description="Full text of the matched chunk.")


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


# ---------- retrieval ----------


def retrieve(question: str, session: Session, settings: Settings) -> list[Document]:
    """Retrieve the chunks nearest a question from the stored corpus.

    Args:
        question: User question to retrieve context for.
        session: Open Session the ranking query runs through.
        settings: Application settings providing embedding_model_name and
            retriever_k.

    Returns:
        Up to settings.retriever_k Documents, nearest first, each carrying the
        matched chunk's text and its document's metadata.
    """
    retriever = PgVectorRetriever(session=session, settings=settings, k=settings.retriever_k)
    return retriever.invoke(question)


# ---------- generation ----------


def format_context(documents: list[Document]) -> str:
    """Join retrieved chunks into a single text block for the prompt context.

    Args:
        documents: Retrieved chunks, each carrying a "title" metadata key.

    Returns:
        Chunks rendered as "{title}: {page_content}", separated by blank lines.

    Raises:
        KeyError: If a document carries no "title" metadata, which means it did
            not come from the retriever and the caller has built it by hand.
    """
    lines = []
    for doc in documents:
        lines.append(f"{doc.metadata['title']}: {doc.page_content}")
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
def get_llm(
    model_name: str, fallback_model_name: str, temperature: float, api_key: str | None
) -> Runnable:
    """Return a process-wide cached chat model with a fallback configured.

    Args:
        model_name: Primary Gemini chat model identifier.
        fallback_model_name: Gemini chat model identifier used if the primary fails.
        temperature: Sampling temperature applied to both models.
        api_key: Gemini API key passed to both models, or None to defer to the
            GOOGLE_API_KEY environment variable.

    Returns:
        Runnable that invokes model_name and falls back to fallback_model_name
        on failure, cached per unique argument combination.
    """
    primary = ChatGoogleGenerativeAI(
        model=model_name, temperature=temperature, google_api_key=api_key
    )
    fallback = ChatGoogleGenerativeAI(
        model=fallback_model_name, temperature=temperature, google_api_key=api_key
    )
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
    messages = prompt.format_messages(context=format_context(documents), question=question)
    llm = get_llm(
        settings.llm_model_name,
        settings.llm_fallback_model_name,
        settings.llm_temperature,
        settings.google_api_key.get_secret_value() if settings.google_api_key else None,
    )
    response = llm.invoke(messages)
    return str(response.text)


# ---------- orchestration ----------


def answer_question(question: str, session: Session, settings: Settings) -> RagResponse:
    """Answer a question against the corpus and cite every chunk it was grounded in.

    Args:
        question: User question to answer.
        session: Open Session the retrieval query runs through.
        settings: Application settings providing embedding_model_name,
            retriever_k and the llm_* values.

    Returns:
        RagResponse whose sources hold one SourceDocument per retrieved chunk,
        in the retriever's ranked order, so the caller can read the citations
        back in the order they were weighted.
    """
    documents = retrieve(question, session, settings)
    answer = generate(question, documents, settings)
    return RagResponse(
        answer=answer,
        sources=[
            SourceDocument(
                doc_key=document.metadata["doc_key"],
                title=document.metadata["title"],
                collection=document.metadata["collection"],
                source=document.metadata["source"],
                chunk_index=document.metadata["chunk_index"],
                content=document.page_content,
            )
            for document in documents
        ],
    )
