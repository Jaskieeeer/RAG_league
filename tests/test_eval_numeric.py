from decimal import Decimal

from lolrag.eval.dataset import load_golden_dataset
from lolrag.eval.numeric import (
    extract_numbers,
    render_token,
    required_tokens,
    score_numeric,
)

GAREN_REFERENCE = (
    "BaseDamage 125/200/275, plus ExecuteDamage 0.25/0.3/0.35 scaling on the target's missing "
    "health. The execute fraction is stored unflagged, so the corpus does not itself mark it as "
    "a percentage."
)

THORNMAIL_REFERENCE = "2450 gold. The Arena version is a different item at 2500."


def tokens(expected: str) -> list[str]:
    """Render the required tokens of a reference answer as text.

    Args:
        expected: Reference answer to extract from.

    Returns:
        One rendered token per requirement, series joined by slashes.
    """
    return [render_token(token) for token in required_tokens(expected)]


# ---------- extraction ----------


def test_extract_numbers_reads_integers_and_decimals():
    assert extract_numbers("70 damage and 0.9 ratio") == [Decimal("70"), Decimal("0.9")]


def test_extract_numbers_strips_thousands_separators():
    assert extract_numbers("costs 1,200 gold") == [Decimal("1200")]


def test_extract_numbers_keeps_list_commas_apart():
    assert extract_numbers("10, 25, 40") == [Decimal("10"), Decimal("25"), Decimal("40")]


def test_extract_numbers_splits_a_slash_series():
    assert extract_numbers("120/100/80 seconds") == [
        Decimal("120"),
        Decimal("100"),
        Decimal("80"),
    ]


# ---------- requirement extraction ----------


def test_a_series_is_one_required_token():
    assert tokens("120/100/80 seconds.") == ["120/100/80"]


def test_only_the_first_sentence_supplies_requirements():
    assert tokens(THORNMAIL_REFERENCE) == ["2450"]


def test_a_reference_opening_without_numbers_falls_through_to_the_next_sentence():
    reference = "Attack damage. RocketTAD is 1.1 of total AD; no AP ratio is stored."

    assert tokens(reference) == ["1.1"]


def test_rank_and_level_indices_are_not_requirements():
    reference = "BleedDamagePerStack 13 at level 1 rising to 30 at level 18, stacking 5 times."

    assert tokens(reference) == ["13", "30"]


def test_a_scalar_repeated_by_a_series_is_not_required_twice():
    reference = "70 physical damage at rank 5, plus 0.9 of total AD (QBaseDamage 10/25/40/55/70,"
    reference += " QTotalADRatio 0.6/0.675/0.75/0.825/0.9)."

    assert tokens(reference) == ["10/25/40/55/70", "0.6/0.675/0.75/0.825/0.9"]


def test_two_series_in_one_sentence_are_two_requirements():
    assert tokens(GAREN_REFERENCE) == ["125/200/275", "0.25/0.3/0.35"]


def test_a_reference_stating_no_number_requires_nothing():
    assert tokens("Attack damage, with no ratio stored.") == []


# ---------- scoring ----------


def test_a_response_carrying_every_series_passes():
    response = "Garen's Demacian Justice deals 125/200/275 plus 0.25/0.3/0.35 of missing health."

    score = score_numeric(GAREN_REFERENCE, response)

    assert score.passed is True
    assert score.coverage == 1.0
    assert score.missing == []


def test_extra_numbers_around_the_right_series_do_not_fail_it():
    response = (
        "At rank 1 it is 125, rank 2 is 200 and rank 3 is 275 base damage, on a 120/100/80 "
        "second cooldown, plus 0.25/0.3/0.35 of the target's missing health at 4 ranks of "
        "nothing in particular."
    )

    score = score_numeric(GAREN_REFERENCE, response)

    assert score.passed is True


def test_a_partly_right_answer_fails_but_keeps_partial_coverage():
    response = "It deals 125/200/275 physical damage."

    score = score_numeric(GAREN_REFERENCE, response)

    assert score.passed is False
    assert score.matched == ["125/200/275"]
    assert score.missing == ["0.25/0.3/0.35"]
    assert score.coverage == 0.5


def test_one_wrong_member_fails_the_whole_series():
    score = score_numeric("BaseHeal 150/250/350.", "It heals 150/250/400.")

    assert score.passed is False
    assert score.coverage == 0.0


def test_trailing_zeros_do_not_change_a_number():
    assert score_numeric("APRatio 0.8 of total AP.", "the ratio is 0.80").passed is True


def test_a_thousands_separated_response_matches_a_plain_reference():
    assert score_numeric("1200 gold.", "It costs 1,200 gold.").passed is True


def test_a_percent_conversion_is_not_accepted_for_an_unflagged_fraction():
    reference = "HealthCost 0.15 at every rank, paid in health rather than mana."

    assert score_numeric(reference, "It costs 15 percent of current health.").passed is False


def test_a_contrast_value_alone_does_not_pass():
    assert score_numeric(THORNMAIL_REFERENCE, "Thornmail costs 2500 gold.").passed is False


def test_a_declining_answer_scores_zero():
    score = score_numeric("120/100/80 seconds.", "The context does not give a cooldown.")

    assert score.passed is False
    assert score.coverage == 0.0


# ---------- the packaged dataset ----------


def test_every_numeric_question_requires_at_least_one_number():
    dataset = load_golden_dataset()

    numeric = [question for question in dataset.questions if question.check == "numeric"]

    assert numeric
    assert all(required_tokens(question.expected_answer) for question in numeric)


def test_every_numeric_question_is_satisfied_by_its_own_reference_answer():
    dataset = load_golden_dataset()

    for question in dataset.questions:
        if question.check != "numeric":
            continue
        score = score_numeric(question.expected_answer, question.expected_answer)
        assert score.passed, f"{question.id} does not satisfy itself: {score.missing}"
