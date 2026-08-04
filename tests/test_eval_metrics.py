from typing import get_args

import pytest
from pydantic import ValidationError

from lolrag.eval.judge import (
    ENTAILMENT_SCORES,
    EntailmentVerdict,
    FaithfulnessVerdict,
    RefusalVerdict,
)
from lolrag.eval.metrics import hit_at_k, reciprocal_rank

# ---------- retrieval metrics ----------


def test_hit_at_k_true_when_relevant_within_k():
    assert hit_at_k(["ability:ahri:Q", "ability:jinx:Q", "ability:garen:Q"], {"ability:jinx:Q"}, 3)


def test_hit_at_k_false_when_relevant_absent():
    assert not hit_at_k(["ability:ahri:Q", "ability:garen:Q"], {"ability:jinx:Q"}, 3)


def test_hit_at_k_false_when_relevant_beyond_k():
    assert not hit_at_k(
        ["ability:ahri:Q", "ability:garen:Q", "ability:jinx:Q"], {"ability:jinx:Q"}, 2
    )


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["ability:jinx:Q", "ability:ahri:Q"], {"ability:jinx:Q"}) == 1.0


def test_reciprocal_rank_third_position():
    retrieved = ["ability:ahri:Q", "ability:garen:Q", "ability:jinx:Q"]

    assert reciprocal_rank(retrieved, {"ability:jinx:Q"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_none_present():
    assert reciprocal_rank(["ability:ahri:Q", "ability:garen:Q"], {"ability:jinx:Q"}) == 0.0


# ---------- verdict schemas ----------


def test_faithfulness_verdict_rejects_score_below_range():
    with pytest.raises(ValidationError):
        FaithfulnessVerdict(score=0, reasoning="out of range")


def test_faithfulness_verdict_rejects_score_above_range():
    with pytest.raises(ValidationError):
        FaithfulnessVerdict(score=6, reasoning="out of range")


def test_faithfulness_verdict_accepts_mid_range_score():
    verdict = FaithfulnessVerdict(score=3, reasoning="partially supported")

    assert verdict.score == 3


def test_entailment_verdict_rejects_an_unknown_label():
    with pytest.raises(ValidationError):
        EntailmentVerdict(verdict="mostly", reasoning="not a label")


def test_entailment_scores_run_from_one_to_zero():
    labels = set(get_args(EntailmentVerdict.model_fields["verdict"].annotation))

    assert ENTAILMENT_SCORES == {"entails": 1.0, "partial": 0.5, "misses": 0.0}
    assert set(ENTAILMENT_SCORES) == labels


def test_refusal_verdict_is_a_single_boolean_axis():
    verdict = RefusalVerdict(declined=True, reasoning="declined for lack of context")

    assert verdict.declined is True
