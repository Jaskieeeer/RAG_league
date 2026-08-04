import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from lolrag import pipeline
from lolrag.config import get_settings
from lolrag.pipeline import RagResponse, SourceDocument, answer_question, format_context


def chunk(doc_key: str, title: str, content: str, chunk_index: int = 0) -> Document:
    """Build a retrieved chunk carrying the metadata the retriever produces.

    Args:
        doc_key: Deterministic key of the owning document.
        title: Title of the owning document.
        content: Chunk text.
        chunk_index: Position of the chunk within its document.

    Returns:
        A Document shaped exactly like one PgVectorRetriever returns.
    """
    return Document(
        page_content=content,
        metadata={
            "doc_key": doc_key,
            "title": title,
            "collection": "lore",
            "source": f"Riot Universe {doc_key}",
            "chunk_index": chunk_index,
            "distance": 0.1,
        },
    )


# ---------- context formatting ----------


def test_format_context_includes_document_content():
    documents = [chunk("champion:aatrox", "Aatrox, the Darkin Blade", "Aatrox is a darkin blade.")]

    context = format_context(documents)

    assert "Aatrox is a darkin blade." in context


def test_format_context_labels_each_chunk_with_its_document_title():
    documents = [chunk("champion:ahri", "Ahri, the Nine-Tailed Fox", "Ahri is a nine-tailed fox.")]

    context = format_context(documents)

    assert context == "Ahri, the Nine-Tailed Fox: Ahri is a nine-tailed fox."


def test_format_context_handles_multiple_documents_distinctly():
    documents = [
        chunk("champion:aatrox", "Aatrox, the Darkin Blade", "Aatrox content."),
        chunk("champion:ahri", "Ahri, the Nine-Tailed Fox", "Ahri content."),
    ]

    context = format_context(documents)

    assert "Aatrox, the Darkin Blade: Aatrox content." in context
    assert "Ahri, the Nine-Tailed Fox: Ahri content." in context


def test_format_context_rejects_a_document_carrying_no_title():
    with pytest.raises(KeyError):
        format_context([Document(page_content="Some lore text.", metadata={"source": "s1"})])


# ---------- source documents ----------


def test_source_document_round_trips_every_field():
    source_document = SourceDocument(
        doc_key="ability:aatrox:Q",
        title="Aatrox Q: The Darkin Blade",
        collection="abilities",
        source="Data Dragon and Community Dragon ability Aatrox Q",
        chunk_index=1,
        content="Aatrox slams his greatsword down.",
    )

    assert source_document.doc_key == "ability:aatrox:Q"
    assert source_document.title == "Aatrox Q: The Darkin Blade"
    assert source_document.collection == "abilities"
    assert source_document.chunk_index == 1
    assert source_document.content == "Aatrox slams his greatsword down."


@pytest.mark.parametrize(
    "missing",
    ["doc_key", "title", "collection", "source", "chunk_index", "content"],
)
def test_source_document_requires_every_field(missing: str):
    fields = {
        "doc_key": "ability:aatrox:Q",
        "title": "Aatrox Q: The Darkin Blade",
        "collection": "abilities",
        "source": "Data Dragon and Community Dragon ability Aatrox Q",
        "chunk_index": 0,
        "content": "Aatrox slams his greatsword down.",
    }
    del fields[missing]

    with pytest.raises(ValidationError):
        SourceDocument(**fields)


def test_source_document_exposes_no_distance():
    assert "distance" not in SourceDocument.model_fields


def test_rag_response_round_trips_answer_and_sources():
    sources = [
        SourceDocument(
            doc_key="champion:aatrox",
            title="Aatrox, the Darkin Blade",
            collection="lore",
            source="Riot Universe champion aatrox",
            chunk_index=0,
            content="Aatrox is a darkin blade.",
        )
    ]

    response = RagResponse(answer="Aatrox is a darkin blade.", sources=sources)

    assert response.answer == "Aatrox is a darkin blade."
    assert response.sources == sources


# ---------- orchestration ----------


def test_answer_question_cites_one_source_per_chunk_in_ranked_order(monkeypatch):
    documents = [
        chunk("story:the-darkin-blade", "The Darkin Blade", "Scene one.", chunk_index=0),
        chunk("story:the-darkin-blade", "The Darkin Blade", "Scene two.", chunk_index=1),
        chunk("champion:aatrox", "Aatrox, the Darkin Blade", "Aatrox bio."),
    ]
    monkeypatch.setattr(pipeline, "retrieve", lambda question, session, settings: documents)
    monkeypatch.setattr(pipeline, "generate", lambda question, docs, settings: "A grounded answer.")

    response = answer_question("Who is Aatrox?", session=None, settings=get_settings())

    assert response.answer == "A grounded answer."
    assert [source.doc_key for source in response.sources] == [
        "story:the-darkin-blade",
        "story:the-darkin-blade",
        "champion:aatrox",
    ]
    assert [source.chunk_index for source in response.sources] == [0, 1, 0]
    assert [source.content for source in response.sources] == [
        "Scene one.",
        "Scene two.",
        "Aatrox bio.",
    ]


def test_answer_question_passes_the_retrieved_chunks_to_generation(monkeypatch):
    documents = [chunk("champion:ahri", "Ahri, the Nine-Tailed Fox", "Ahri content.")]
    seen: dict[str, object] = {}
    monkeypatch.setattr(pipeline, "retrieve", lambda question, session, settings: documents)

    def fake_generate(question, docs, settings):
        seen["question"] = question
        seen["docs"] = docs
        return "answer"

    monkeypatch.setattr(pipeline, "generate", fake_generate)

    answer_question("Who is Ahri?", session=None, settings=get_settings())

    assert seen["question"] == "Who is Ahri?"
    assert seen["docs"] == documents
