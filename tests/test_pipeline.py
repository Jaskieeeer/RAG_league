import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from lolrag.config import Settings
from lolrag.indexing import build_index
from lolrag.pipeline import RagResponse, SourceDocument, _format_context, answer_question, retrieve
from tests.test_ingestion import build_documents


def test_format_context_includes_document_content():
    documents = [
        Document(
            page_content="Aatrox is a darkin blade.",
            metadata={"name": "Aatrox", "source": "s1"},
        )
    ]

    context = _format_context(documents)

    assert "Aatrox is a darkin blade." in context


def test_format_context_includes_champion_name():
    documents = [
        Document(
            page_content="Ahri is a nine-tailed fox.",
            metadata={"name": "Ahri", "source": "s1"},
        )
    ]

    context = _format_context(documents)

    assert "Ahri" in context


def test_format_context_handles_single_document():
    documents = [
        Document(page_content="Jinx loves chaos.", metadata={"name": "Jinx", "source": "s1"})
    ]

    context = _format_context(documents)

    assert context == "Jinx: Jinx loves chaos."


def test_format_context_handles_multiple_documents_distinctly():
    documents = [
        Document(page_content="Aatrox content.", metadata={"name": "Aatrox", "source": "s1"}),
        Document(page_content="Ahri content.", metadata={"name": "Ahri", "source": "s2"}),
    ]

    context = _format_context(documents)

    assert "Aatrox content." in context
    assert "Ahri content." in context


def test_format_context_falls_back_to_source_when_name_missing():
    documents = [Document(page_content="Some lore text.", metadata={"source": "wiki:lore:1"})]

    context = _format_context(documents)

    assert context == "wiki:lore:1: Some lore text."


def test_source_document_round_trips_optional_and_required_fields():
    source_document = SourceDocument(champion_id="Aatrox", name="Aatrox", source="s1")

    assert source_document.champion_id == "Aatrox"
    assert source_document.name == "Aatrox"
    assert source_document.source == "s1"


def test_source_document_allows_missing_champion_fields():
    source_document = SourceDocument(source="wiki:lore:1")

    assert source_document.champion_id is None
    assert source_document.name is None
    assert source_document.source == "wiki:lore:1"


def test_source_document_missing_source_raises_validation_error():
    with pytest.raises(ValidationError):
        SourceDocument(champion_id="Aatrox", name="Aatrox")


def test_rag_response_round_trips_answer_and_sources():
    sources = [
        SourceDocument(champion_id="Aatrox", name="Aatrox", source="s1"),
        SourceDocument(champion_id="Ahri", name="Ahri", source="s2"),
    ]

    response = RagResponse(answer="Aatrox is a darkin blade.", sources=sources)

    assert response.answer == "Aatrox is a darkin blade."
    assert response.sources == sources


@pytest.mark.integration
def test_retrieve_returns_relevant_champion(tmp_path):
    settings = Settings(
        ddragon_version="16.14.1",
        chroma_persist_dir=str(tmp_path / "chroma"),
    )
    build_index(build_documents(), settings)

    documents = retrieve("Tell me about Jinx, the Loose Cannon.", settings)

    assert documents
    assert any(doc.metadata["champion_id"] == "Jinx" for doc in documents)


@pytest.mark.integration
def test_answer_question_returns_grounded_response(tmp_path):
    settings = Settings(
        ddragon_version="16.14.1",
        chroma_persist_dir=str(tmp_path / "chroma"),
    )
    build_index(build_documents(), settings)

    response = answer_question("Tell me about Jinx, the Loose Cannon.", settings)

    assert isinstance(response, RagResponse)
    assert response.answer
    assert response.sources
