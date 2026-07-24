import pytest
from pydantic import ValidationError

from lolrag.eval.dataset import load_golden_dataset
from lolrag.eval.judge import FaithfulnessVerdict
from lolrag.eval.metrics import hit_at_k, reciprocal_rank


def test_hit_at_k_true_when_relevant_within_k():
    assert hit_at_k(["Ahri", "Jinx", "Garen"], {"Jinx"}, 3) is True


def test_hit_at_k_false_when_relevant_absent():
    assert hit_at_k(["Ahri", "Garen", "Lux"], {"Jinx"}, 3) is False


def test_hit_at_k_false_when_relevant_beyond_k():
    assert hit_at_k(["Ahri", "Garen", "Jinx"], {"Jinx"}, 2) is False


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["Jinx", "Ahri"], {"Jinx"}) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["Ahri", "Garen", "Jinx"], {"Jinx"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_none_present():
    assert reciprocal_rank(["Ahri", "Garen"], {"Jinx"}) == 0.0


def test_golden_dataset_has_questions():
    dataset = load_golden_dataset()

    assert dataset.questions


def test_golden_dataset_ids_are_unique():
    dataset = load_golden_dataset()

    ids = [question.id for question in dataset.questions]

    assert len(ids) == len(set(ids))


def test_golden_dataset_expected_champion_ids_non_empty():
    dataset = load_golden_dataset()

    assert all(question.expected_champion_ids for question in dataset.questions)


def test_golden_dataset_categories_are_known():
    dataset = load_golden_dataset()

    assert all(question.category in {"factual", "refusal"} for question in dataset.questions)


def test_faithfulness_verdict_rejects_score_below_range():
    with pytest.raises(ValidationError):
        FaithfulnessVerdict(score=0, reasoning="out of range")


def test_faithfulness_verdict_rejects_score_above_range():
    with pytest.raises(ValidationError):
        FaithfulnessVerdict(score=6, reasoning="out of range")


def test_faithfulness_verdict_accepts_mid_range_score():
    verdict = FaithfulnessVerdict(score=3, reasoning="partially supported")

    assert verdict.score == 3
