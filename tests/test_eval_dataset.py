from collections import Counter

import pytest
from pydantic import ValidationError

from lolrag.eval.dataset import GoldenDataset, GoldenQuestion, load_golden_dataset

EXPECTED_QUESTIONS = 164
EXPECTED_SCORING = {"retrieval": 120, "refusal": 30, "limitation": 14}
EXPECTED_CHECKS = {"numeric": 46, "entailment": 74, "refusal": 30, "none": 14}
EXPECTED_COLLECTIONS = {"abilities", "champion_stats", "equipment", "lore", None}


def question(**overrides: object) -> dict[str, object]:
    """Build a valid golden question payload with fields overridden.

    Args:
        overrides: Fields to replace in the base payload.

    Returns:
        A mapping ready to pass to GoldenQuestion.
    """
    payload: dict[str, object] = {
        "id": "q01",
        "question": "What does Aatrox's Q do?",
        "category": "ability-mechanics",
        "scoring": "retrieval",
        "check": "entailment",
        "expected_doc_keys": ["ability:aatrox:Q"],
        "expected_answer": "Three sword swings.",
        "collection": "abilities",
        "failure_mode": "prose description retrieval",
    }
    payload.update(overrides)
    return payload


# ---------- packaged dataset ----------


def test_packaged_dataset_holds_every_authored_question():
    dataset = load_golden_dataset()

    assert dataset.version == 3
    assert len(dataset.questions) == EXPECTED_QUESTIONS


def test_packaged_dataset_scoring_counts():
    dataset = load_golden_dataset()

    counts = Counter(question.scoring for question in dataset.questions)

    assert dict(counts) == EXPECTED_SCORING


def test_packaged_dataset_check_counts():
    dataset = load_golden_dataset()

    counts = Counter(question.check for question in dataset.questions)

    assert dict(counts) == EXPECTED_CHECKS


def test_packaged_dataset_ids_are_unique():
    dataset = load_golden_dataset()

    ids = [question.id for question in dataset.questions]

    assert len(ids) == len(set(ids))


def test_packaged_dataset_doc_keys_are_namespaced():
    dataset = load_golden_dataset()

    keys = [key for question in dataset.questions for key in question.expected_doc_keys]

    assert keys
    assert all(":" in key for key in keys)


def test_packaged_dataset_collections_are_known():
    dataset = load_golden_dataset()

    assert {question.collection for question in dataset.questions} <= EXPECTED_COLLECTIONS


def test_packaged_dataset_every_question_carries_a_reference_answer():
    dataset = load_golden_dataset()

    assert all(question.expected_answer.strip() for question in dataset.questions)


def test_packaged_dataset_every_question_names_a_failure_mode():
    dataset = load_golden_dataset()

    assert all(question.failure_mode.strip() for question in dataset.questions)


def test_by_scoring_selects_only_that_block():
    dataset = load_golden_dataset()

    block = dataset.by_scoring("refusal")

    assert len(block) == EXPECTED_SCORING["refusal"]
    assert all(question.scoring == "refusal" for question in block)


def test_dropped_fields_are_gone_from_the_schema():
    assert "expected_champion_ids" not in GoldenQuestion.model_fields
    assert "notes" not in GoldenQuestion.model_fields


# ---------- schema ----------


def test_question_rejects_an_unknown_scoring_mode():
    with pytest.raises(ValidationError):
        GoldenQuestion(**question(scoring="qualitative"))


def test_question_rejects_an_unknown_check():
    with pytest.raises(ValidationError):
        GoldenQuestion(**question(check="vibes"))


def test_question_rejects_a_check_its_scoring_mode_forbids():
    with pytest.raises(ValidationError):
        GoldenQuestion(**question(scoring="retrieval", check="refusal"))


def test_question_rejects_a_retrieval_question_with_no_doc_keys():
    with pytest.raises(ValidationError):
        GoldenQuestion(**question(expected_doc_keys=[]))


def test_question_rejects_a_refusal_question_carrying_doc_keys():
    with pytest.raises(ValidationError):
        GoldenQuestion(
            **question(scoring="refusal", check="refusal", expected_doc_keys=["ability:aatrox:Q"])
        )


def test_question_accepts_a_limitation_question_with_no_keys_and_no_collection():
    parsed = GoldenQuestion(
        **question(scoring="limitation", check="none", expected_doc_keys=[], collection=None)
    )

    assert parsed.collection is None


def test_dataset_rejects_duplicate_question_ids():
    with pytest.raises(ValidationError):
        GoldenDataset(
            description="two of the same",
            version=2,
            questions=[GoldenQuestion(**question()), GoldenQuestion(**question())],
        )
