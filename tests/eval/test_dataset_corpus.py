from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from lolrag.config import get_settings
from lolrag.db.models import Document
from lolrag.eval.dataset import load_golden_dataset

CACHE_DIR = Path(get_settings().cache_dir)

pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(not CACHE_DIR.is_dir(), reason=f"no warm corpus cache at {CACHE_DIR}"),
]


def test_every_expected_doc_key_resolves_against_the_corpus(corpus_session: Session) -> None:
    """Every doc key and collection the golden dataset names exists in the documents table.

    A mistyped key cannot fail loudly on its own: no chunk would ever carry it,
    so the question would score as a permanent retrieval miss and read as a
    retriever that cannot find an easy document. The collection is checked in
    the same pass because the dataset declares one so that a later routing
    upgrade can be scored on it, and that is worth nothing unless it agrees with
    the corpus today. Both assertions share one ingest, which is the expensive
    part of this gate.
    """
    dataset = load_golden_dataset()
    stored = dict(corpus_session.execute(select(Document.doc_key, Document.collection)).all())
    expected = {key for question in dataset.questions for key in question.expected_doc_keys}

    assert expected
    assert sorted(expected - set(stored)) == []

    mismatches = [
        f"{question.id}: {key} is in {stored[key]}, dataset says {question.collection}"
        for question in dataset.questions
        if question.collection is not None
        for key in question.expected_doc_keys
        if stored[key] != question.collection
    ]

    assert mismatches == []
