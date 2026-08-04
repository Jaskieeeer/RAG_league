import pytest
from pydantic import ValidationError

from lolrag.eval.agreement import (
    EntailmentSample,
    compute_agreement,
    dump_entailment_sample,
    load_entailment_sample,
    select_sample,
)


def sample(id_: str, judge: str, human: str | None = None) -> EntailmentSample:
    """Build one entailment judgement record.

    Args:
        id_: Golden question identifier.
        judge: The judge's verdict.
        human: The hand-assigned verdict, or None when unscored.

    Returns:
        An EntailmentSample carrying fixture text.
    """
    return EntailmentSample(
        id=id_,
        system="pipeline",
        question=f"question {id_}",
        reference=f"reference {id_}",
        answer=f"answer {id_}",
        judge_verdict=judge,
        human_verdict=human,
    )


# ---------- sampling ----------


def test_a_sample_smaller_than_the_request_is_returned_whole():
    candidates = [sample("a", "entails"), sample("b", "misses")]

    assert len(select_sample(candidates, 10)) == 2


def test_the_draw_is_deterministic_for_a_seed():
    candidates = [sample(f"q{index:02d}", "entails") for index in range(20)]

    first = select_sample(candidates, 5, seed=7)
    second = select_sample(candidates, 5, seed=7)

    assert [item.id for item in first] == [item.id for item in second]
    assert len(first) == 5


def test_the_sample_is_ordered_by_question_id():
    candidates = [sample("c", "entails"), sample("a", "misses"), sample("b", "partial")]

    assert [item.id for item in select_sample(candidates, 3)] == ["a", "b", "c"]


# ---------- round trip ----------


def test_a_dumped_sample_reads_back_unchanged(tmp_path):
    samples = [sample("a", "entails"), sample("b", "partial")]

    path = dump_entailment_sample(samples, tmp_path / "nested" / "entailment_sample.json")

    assert load_entailment_sample(path) == samples


def test_a_dumped_sample_starts_unscored(tmp_path):
    path = dump_entailment_sample([sample("a", "entails")], tmp_path / "sample.json")

    assert all(item.human_verdict is None for item in load_entailment_sample(path))


def test_hand_written_verdicts_are_validated(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text(
        '[{"id": "a", "system": "pipeline", "question": "q", "reference": "r", '
        '"answer": "a", "judge_verdict": "entails", "human_verdict": "yes"}]',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_entailment_sample(path)


# ---------- agreement ----------


def test_agreement_counts_only_hand_scored_records():
    samples = [
        sample("a", "entails", "entails"),
        sample("b", "partial", "misses"),
        sample("c", "misses"),
    ]

    agreement = compute_agreement(samples)

    assert agreement.num_hand_scored == 2
    assert agreement.num_agreed == 1
    assert agreement.agreement_rate == 0.5
    assert agreement.disagreement_ids == ["b"]


def test_agreement_over_nothing_is_zero_and_not_a_crash():
    agreement = compute_agreement([sample("a", "entails")])

    assert agreement.num_hand_scored == 0
    assert agreement.agreement_rate == 0.0
