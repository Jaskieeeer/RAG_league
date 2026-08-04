from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from lolrag.config import get_settings
from lolrag.db.models import Chunk
from lolrag.retrieval import PgVectorRetriever

CACHE_DIR = Path(get_settings().cache_dir)

pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(not CACHE_DIR.is_dir(), reason=f"no warm corpus cache at {CACHE_DIR}"),
]

EXPECTED_CHUNKS = 7650

ABILITY_QUESTION = "How much damage does Aatrox's Q, The Darkin Blade, deal?"
ABILITY_EXPECTED_DOC_KEY = "ability:aatrox:Q"

LORE_QUESTION = "What is the story of Viego, the Ruined King?"
LORE_EXPECTED_DOC_KEY = "champion:viego"

EQUIPMENT_QUESTION = "Which item reflects damage back at the attacker with armor?"
EQUIPMENT_EXPECTED_COLLECTION = "equipment"


def test_retriever_finds_the_right_documents_in_the_full_corpus(
    corpus_session: Session,
) -> None:
    """Against all 7650 chunks the naive dense retriever puts the answering document in top k.

    Retrieval only: no LLM is called, so the assertion is about the ranking and
    nothing else. Every question is answered from a different collection, since
    the retriever applies no collection filter and all four compete in one pass.
    """
    settings = get_settings()

    assert corpus_session.execute(select(func.count()).select_from(Chunk)).scalar_one() == (
        EXPECTED_CHUNKS
    )

    retriever = PgVectorRetriever(session=corpus_session, settings=settings, k=settings.retriever_k)

    ability_hits = retriever.invoke(ABILITY_QUESTION)
    assert len(ability_hits) == settings.retriever_k
    assert ABILITY_EXPECTED_DOC_KEY in [doc.metadata["doc_key"] for doc in ability_hits]
    assert ability_hits[0].metadata["collection"] == "abilities"

    lore_hits = retriever.invoke(LORE_QUESTION)
    assert LORE_EXPECTED_DOC_KEY in [doc.metadata["doc_key"] for doc in lore_hits]
    assert lore_hits[0].metadata["collection"] == "lore"

    equipment_hits = retriever.invoke(EQUIPMENT_QUESTION)
    assert equipment_hits[0].metadata["collection"] == EQUIPMENT_EXPECTED_COLLECTION

    distances = [doc.metadata["distance"] for doc in ability_hits]
    assert distances == sorted(distances)
    assert all(doc.page_content for doc in ability_hits)
